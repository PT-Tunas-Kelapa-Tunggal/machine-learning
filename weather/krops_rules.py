"""Krops L5 and L8 weather rules as pure functions: the reference spec for the backend and the Swift app.

No network, no files, no pandas. Every function takes plain values and returns plain values, so each
one can be ported line by line and checked against the same cases in `tests/rule_cases.json`.

Conventions
- Daily rain is a dict {date: mm}. A missing day is absent or None; it is never treated as 0 mm.
- A date that arrives more than once counts once if its values agree, and as missing if they don't
  (daily_from_pairs builds the dict that way from raw (date, mm) pairs).
- A month is a (year, month) tuple.
- Comparisons are strict where the rule says "over"/"under", exactly as written below.

Status of each rule: "proposal" = the 17 Sep 2026 proposal, "decided" = the 9 Oct 2026 decisions,
"proposed" = recommended by the weather notebooks, pending BMKG station data.
"""
from __future__ import annotations

from datetime import timedelta
from statistics import mean, median

# ============================================================
# THRESHOLDS
# ============================================================

P30_THRESHOLD_MM = 60.0     # L5: water_stress when P30 is under this (proposal)
P30_WINDOW_DAYS = 30
P30_MAX_AGE_DAYS = 7        # a window ending earlier than this many days ago is too old (proposed)
P30_SOURCE_ORDER = ("weatherkit-history", "nasa-power")  # proposed; on 9 Oct NASA POWER came first

SUSPECT_RATIO = 3.0         # NASA POWER monthly check (proposed)
NORMAL_PERIOD = (1991, 2020)

HEAVY_RAIN_MM = 25.0        # L8 Stage B: a day over this is heavy (proposal)
GATE_TUNDA_BELOW_MM = 60.0  # L8 Stage A: month average under this → tunda (proposal)
GATE_CEK_BELOW_MM = 100.0   # 60–100 → cek (decided)
GATE_TUNDA_ABOVE_MM = 300.0  # over this → tunda (proposal); 200–300 stays open (decided)
IN_BAND_MM = (100.0, 200.0)  # in-band share, logged only (decided)
CUT_WINDOW_AT_CLOSED_MONTH = True  # PENDING TEAM DECISION – not in proposal or brief


def _present(daily):
    return {d: float(v) for d, v in (daily or {}).items() if v is not None}


def daily_from_pairs(pairs):
    """{date: mm or None} from raw (date, mm) pairs, one entry per date.

    A date given more than once keeps its value when every value agrees, and becomes missing (None) when they
    disagree: there is no telling which one is right. A None next to a value isn't a disagreement.
    """
    values = {}
    for d, v in pairs:
        values.setdefault(d, set())
        if v is not None:
            values[d].add(float(v))
    return {d: (next(iter(v)) if len(v) == 1 else None) for d, v in sorted(values.items())}


# ============================================================
# L5 — P30 AND water_stress
# ============================================================

def missing_in_p30_window(daily):
    """Days missing inside the 30-day window ending on the latest day with a value (all 30 if no day has one).

    Missing days after that latest day aren't counted: they move the window back, and choose_p30's age limit
    decides whether it is still recent enough.
    """
    rain = _present(daily)
    if not rain:
        return P30_WINDOW_DAYS
    end = max(rain)
    return sum(1 for i in range(P30_WINDOW_DAYS) if end - timedelta(days=i) not in rain)


def compute_p30(daily, source):
    """P30 from one source: the 30 calendar days ending on the latest day with a value.

    All 30 days must be present; otherwise there is no P30 (None), rather than a sum of fewer days.
    """
    rain = _present(daily)
    if not rain:
        return None
    end = max(rain)
    window = [end - timedelta(days=i) for i in range(P30_WINDOW_DAYS - 1, -1, -1)]
    if any(d not in rain for d in window):
        return None
    p30_mm = round(sum(rain[d] for d in window), 1)
    return {
        "p30_mm": p30_mm,
        "p30_start_date": window[0],
        "p30_end_date": end,
        "p30_source": source,
        "water_stress": p30_mm < P30_THRESHOLD_MM,
    }


def choose_p30(sources, today, order=P30_SOURCE_ORDER, max_age_days=P30_MAX_AGE_DAYS):
    """The L5 data-contract record: the first source in `order` with a full window ending ≤ max_age_days ago.

    `sources` maps a source name to its daily rain (NASA POWER already filtered by `suspect_months`).
    p30_quality is "ok" for the first source in the order, "fallback" for a later one, "none" if none qualifies;
    with "none", p30_mm and water_stress are None (unknown), never a silent False. p30_missing_days gives, for every
    source in the order, the days missing inside its latest 30-day window (missing_in_p30_window), so a gap that
    sent P30 to the fallback shows in the record. A window is never completed from another source.
    """
    missing = {name: missing_in_p30_window(sources.get(name)) for name in order}
    for rank, name in enumerate(order):
        record = compute_p30(sources.get(name), name)
        if record and (today - record["p30_end_date"]).days <= max_age_days:
            return record | {"p30_quality": "ok" if rank == 0 else "fallback", "p30_missing_days": missing}
    return {"p30_mm": None, "p30_start_date": None, "p30_end_date": None,
            "p30_source": None, "water_stress": None, "p30_quality": "none", "p30_missing_days": missing}


# ============================================================
# NASA POWER MONTHLY CHECK
# ============================================================

def monthly_totals(daily):
    """{(year, month): total mm} over the days present."""
    totals = {}
    for d, v in _present(daily).items():
        totals[(d.year, d.month)] = totals.get((d.year, d.month), 0.0) + v
    return totals


def monthly_normals(daily, first=NORMAL_PERIOD[0], last=NORMAL_PERIOD[1]):
    """{calendar month: median monthly total} over the years first–last (default the 1991–2020 normal)."""
    by_month = {}
    for (y, m), total in monthly_totals(daily).items():
        if first <= y <= last:
            by_month.setdefault(m, []).append(total)
    return {m: median(v) for m, v in sorted(by_month.items())}


def suspect_months(nasa, normals, weatherkit=None, ratio=SUSPECT_RATIO):
    """NASA POWER months to drop: total over ratio × the normal AND over ratio × WeatherKit's total.

    A month WeatherKit has no data for is judged on the normal alone. Requiring both keeps a genuinely
    wet month, which WeatherKit also sees, from being thrown out.
    """
    wk = monthly_totals(weatherkit) if weatherkit else {}
    out = set()
    for (y, m), total in monthly_totals(nasa).items():
        if total <= ratio * normals[m]:
            continue
        if (y, m) in wk and total <= ratio * wk[(y, m)]:
            continue
        out.add((y, m))
    return out


def drop_months(daily, months):
    return {d: v for d, v in daily.items() if (d.year, d.month) not in months}


# ============================================================
# L8 — STAGE A MONTH GATE
# ============================================================

def month_table(daily, first, last):
    """Stage A table: {calendar month: {"avg_mm", "in_band_share"}} over the years first–last."""
    by_month = {}
    for (y, m), total in monthly_totals(daily).items():
        if first <= y <= last:
            by_month.setdefault(m, []).append(total)
    low, high = IN_BAND_MM
    return {m: {"avg_mm": mean(v), "in_band_share": sum(low <= t <= high for t in v) / len(v)}
            for m, v in sorted(by_month.items())}


def month_gate(avg_mm):
    """'tunda' under 60 or over 300 mm, 'cek' from 60 to under 100 mm, else 'open'."""
    if avg_mm < GATE_TUNDA_BELOW_MM or avg_mm > GATE_TUNDA_ABOVE_MM:
        return "tunda"
    if avg_mm < GATE_CEK_BELOW_MM:
        return "cek"
    return "open"


# ============================================================
# L8 — STAGE B DAY PICK AND THE FULL DECISION
# ============================================================

def stage_b_day_pick(forecast, today, heavy_mm=HEAVY_RAIN_MM):
    """Days of the cached forecast that are safe to spread on, and why the others were removed.

    Keeps days from today on that have an amount, then removes: days over heavy_mm (proposal); the day
    before each heavy day (decided, no source yet); the last cached day (no next-day forecast to check it).
    Each removed day is listed once, under the first reason that applies.
    """
    rain = {d: v for d, v in _present(forecast).items() if d >= today}
    dates = sorted(rain)
    heavy = {d for d in dates if rain[d] > heavy_mm}
    day_before = {d - timedelta(days=1) for d in heavy} & set(dates)
    last = set(dates[-1:])
    window = [d for d in dates if d not in heavy | day_before | last]
    removed = {
        "heavy": sorted(heavy),
        "day_before_heavy": sorted(day_before - heavy),
        "last_cached": sorted(last - heavy - day_before),
    }
    return window, removed


def l8_decision(forecast, table, today, heavy_mm=HEAVY_RAIN_MM, cut_at_closed_month=CUT_WINDOW_AT_CLOSED_MONTH):
    """The L8 output: Stage A on today's month, then Stage B on the cached forecast.

    `table` maps each calendar month to {"avg_mm": …} (the 1991–2020 table, proposed). With cut_at_closed_month,
    dates of a later month whose own gate isn't open are dropped from the window. That cut is pending a team
    decision: it is in neither the proposal nor the brief.
    """
    gates = {m: month_gate(row["avg_mm"]) for m, row in table.items()}
    result = {"month_gate": {"month": today.month, "avg_mm": table[today.month]["avg_mm"], "gate": gates[today.month]},
              "dates": [], "removed": None}
    if gates[today.month] == "tunda":
        return result | {"l8_state": "tunda"}
    if gates[today.month] == "cek":
        return result | {"l8_state": "cek"}
    if not _present(forecast):
        return result | {"l8_state": "cek_hujan"}
    window, removed = stage_b_day_pick(forecast, today, heavy_mm)
    if cut_at_closed_month:
        window = [d for d in window if gates.get(d.month) == "open"]
    return result | {"l8_state": "jendela_tabur" if window else "cek_hujan", "dates": window, "removed": removed}
