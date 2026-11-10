"""Jadwal Pemupukan: the year's fertilizer rounds for the app screen, from the farm's fertilizers and L8 Stage A.

In:  a request JSON with latitude/longitude, fertilizerFreq (rounds a year) and fertilizers[], where
     overallDoseWeightInKg is the kg of that product for the whole farm in a year: each round gets it ÷ freq.
     Only the listed fertilizers are scheduled; one with a null id can't be marked as spread.
     Optional applications[]: {fertilizerId, date} for the steps already spread ("Tandai Mulai Memupuk" or, when
     overdue, "Tandai Sudah Dipupuk").
Out: one round per application in the plan year: its window (start month and the next one), its status from today
     ("dimulai dalam X hari lagi" or "hari ke-N dari M hari"), and the fertilizers in order with the wait between them.
     Each round carries its card states (header, each fertilizer, each wait: README.md "Card states").
     A round under way or starting within the 10-day forecast also gets `daily`: the safe days, a date (or "not
     before", or the reason to wait) for each step.

The months come from the agreed weather rule: L8 Stage A, the 1991–2020 average rain of each calendar month
(krops_rules.py month_gate: open, cek or tunda), read through weather_sources.py and its cache in data/.
The rounds go in open months 12 ÷ freq apart, the set with the highest in-band (100–200 mm) share, as in
spreading.round_months on ml-integration (154ac1f). The days come from L8 Stage B (WeatherKit's forecast) and L5
(P30), through the same modules. Both are copies of the team's weather code, so this folder runs on its own
(README.md "Copied from"). Coordinates are never written to the output. README.md lists every datapoint.

Usage:
    python integration/v2/fertilization_timing/fertilization_timing.py \
        --request integration/v2/fertilization_timing/example_request.json \
        [--today 2026-11-08] [--year 2027] [--no-daily] [--out schedule.json]
"""
import argparse
import calendar
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "1.0"
TZ = ZoneInfo("Asia/Jakarta")
DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "weather_cache"
MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]  # in `wait` texts

ALLOWED_ROUNDS_PER_YEAR = (1, 2, 3, 4, 6)  # whole months apart (12 ÷ rounds), so not 5
ROUND_WINDOW_MONTHS = 2  # GUESS – from the app design: a round runs from its month through the next one
FORECAST_DAYS = 10  # WeatherKit's daily forecast: a round starting this close gets daily dates
MONTHS_ID = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
             "November", "Desember"]

# Days to wait before the next product, as (min, max); max None = any later day.
# Dolomite is a lime: the higher pH near its particles turns urea/ammonium N into ammonia gas (Minato et al., RBCS
# 2026: +1.7–2.2% NH3 loss from urea per t/ha of dolomitic lime). The number of days is practice, not PPKS/MPOB:
# pabrikpupukdolomit.com gives 21–28 days before urea, ZA or NPK, 14–21 before SP-36 and 14 before KCl.
DOLOMITE_WAIT_DAYS = (14, 28)
KIESERITE_WAIT_DAYS = (1, 1)  # GUESS – integration/krops_vari/config.py min_days_mg_to_n["kieserite_Mg_expensive"]
# Urea/ZA go on a day of their own, but no source sets a gap from NPK, KCl or SP-36 (a Bungo TM 15 trial, Rozi &
# Prastia 2019, spread urea with KCl a week after SP-36). Rain decides the urea day: L8 Stage B.
BEFORE_N_STRAIGHT_WAIT_DAYS = (1, None)
OTHER_WAIT_DAYS = (1, 7)  # GUESS – from the app design: "Tunggu 1–7 hari" between the other products

# Order in a round: Mg/lime first, then the compound, then the straights; N last, never mixed with the lime.
# (display name, order group, (min, max) days to wait before the next product)
N_STRAIGHT_GROUP = 3
PRODUCTS = {
    "dolomite": ("Dolomit", 0, DOLOMITE_WAIT_DAYS),
    "dolomit": ("Dolomit", 0, DOLOMITE_WAIT_DAYS),
    "kieserite": ("Kieserit", 0, KIESERITE_WAIT_DAYS),
    "kieserit": ("Kieserit", 0, KIESERITE_WAIT_DAYS),
    "npk": ("NPK", 1, OTHER_WAIT_DAYS),
    "sp-36": ("SP-36", 2, OTHER_WAIT_DAYS),
    "sp36": ("SP-36", 2, OTHER_WAIT_DAYS),
    "kcl": ("KCl", 2, OTHER_WAIT_DAYS),
    "mop": ("KCl", 2, OTHER_WAIT_DAYS),
    "kcl/mop": ("KCl", 2, OTHER_WAIT_DAYS),
    "kcl (mop)": ("KCl", 2, OTHER_WAIT_DAYS),
    "urea": ("Urea", N_STRAIGHT_GROUP, OTHER_WAIT_DAYS),
    "za": ("ZA", N_STRAIGHT_GROUP, OTHER_WAIT_DAYS),
}
UNKNOWN_PRODUCT = (None, 2, OTHER_WAIT_DAYS)  # after the compound, before the N straights


def rounds_per_year(request):
    """fertilizerFreq as a whole number of rounds a year; one of ALLOWED_ROUNDS_PER_YEAR."""
    freq = request.get("fertilizerFreq")
    if isinstance(freq, bool) or not isinstance(freq, (int, float)) or freq != int(freq):
        raise ValueError(f"fertilizerFreq must be a whole number, not {freq!r}")
    if int(freq) not in ALLOWED_ROUNDS_PER_YEAR:
        raise ValueError(f"fertilizerFreq must be one of {', '.join(map(str, ALLOWED_ROUNDS_PER_YEAR))}, "
                         f"not {int(freq)}")
    return int(freq)


def round_months(stage_a, rounds):
    """(chosen months, every open set with its combined in-band share), months 1–12; (None, []) if no set is open.

    A set is one month per round, 12 ÷ rounds apart: with 3 rounds e.g. (1, 5, 9). Every month in it must be open.
    Ties go to the set starting earliest in the year (shares are rounded first, so float noise can't break a tie).
    """
    gap = 12 // rounds
    sets = sorted({tuple(sorted((m - 1 + k * gap) % 12 + 1 for k in range(rounds))) for m in range(1, 13)})
    open_sets = [(s, round(sum(stage_a[m]["in_band_share"] for m in s), 3)) for s in sets
                 if all(stage_a.get(m, {}).get("gate") == "open" for m in s)]
    if not open_sets:
        return None, []
    return max(open_sets, key=lambda item: item[1])[0], open_sets


def round_window(year, month):
    """(first day, last day) of a round starting in `month` of `year` and running ROUND_WINDOW_MONTHS months."""
    end_index = month - 1 + ROUND_WINDOW_MONTHS - 1
    end_year, end_month = year + end_index // 12, end_index % 12 + 1
    return date(year, month, 1), date(end_year, end_month, calendar.monthrange(end_year, end_month)[1])


def round_status(start, end, today):
    """What the app's round header shows: upcoming (days until start), ongoing (day N of M) or done."""
    total_days = (end - start).days + 1
    if today < start:
        return {"status": "upcoming", "days_until_start": (start - today).days, "day_of_round": None,
                "total_days": total_days}
    if today <= end:
        return {"status": "ongoing", "days_until_start": 0, "day_of_round": (today - start).days + 1,
                "total_days": total_days}
    return {"status": "done", "days_until_start": None, "day_of_round": None, "total_days": total_days}


def _npk_display(entry):
    """"NPK 15-15-15" from the composition, with "-Mg" added when the label has Mg."""
    comp = entry.get("composition") or {}
    if not all(k in comp for k in ("N", "P", "K")):
        return "NPK"
    grade = [comp["N"], comp["P"], comp["K"]] + ([comp["Mg"]] if comp.get("Mg") else [])
    return "NPK " + "-".join(f"{float(v):g}" for v in grade)


def fertilizer_steps(fertilizers, rounds):
    """(steps of one round in spreading order, flags). The wait sits on the step it follows, as on the screen.

    dose_kg is the round's share: the yearly overallDoseWeightInKg ÷ rounds."""
    flags, ordered = [], []
    for index, entry in enumerate(fertilizers):
        name = str(entry.get("name", "")).strip()
        display, group, wait = PRODUCTS.get(name.casefold(), UNKNOWN_PRODUCT)
        if display is None:
            display = name or "Unknown product"
            flags.append(f"{display}: no spreading rule for this product; placed after NPK, before Urea/ZA")
        elif display == "NPK":
            display = _npk_display(entry)
        dose = entry.get("overallDoseWeightInKg")
        if dose is None:
            flags.append(f"{display}: overallDoseWeightInKg is missing")
        if entry.get("id") is None:
            flags.append(f"{display}: no id, so it can't be marked as spread")
        ordered.append((group, index, entry, display, dose, wait))
    ordered.sort(key=lambda item: item[:2])  # stable: the user's order inside a group
    mg_sources = {display for group, _, _, display, _, _ in ordered if group == 0}
    if len(mg_sources) > 1:
        flags.append(f"both {' and '.join(sorted(mg_sources))} listed: they are alternative Mg sources, pick one")
    steps = []
    for number, (group, _, entry, display, dose, wait) in enumerate(ordered, start=1):
        last = number == len(ordered)
        if not last and group != 0 and ordered[number][0] == N_STRAIGHT_GROUP:
            wait = BEFORE_N_STRAIGHT_WAIT_DAYS  # the lime's own wait still holds before N
        steps.append({
            "step": number,
            "fertilizer_id": entry.get("id"),
            "name": entry.get("name"),
            "display_name": display,
            "dose_kg": float(dose) / rounds if dose is not None else None,
            "annual_dose_kg": float(dose) if dose is not None else None,
            "bag_kg": entry.get("bagKg"),
            "wait_after_days": None if last else {"min": wait[0], "max": wait[1]},
        })
    return steps, flags


def default_plan_year(chosen, today):
    """This year while one of its rounds hasn't ended yet, else next year (November 2026 → 2027, January 2027 → 2027)."""
    if any(round_window(today.year, month)[1] >= today for month in chosen or ()):
        return today.year
    return today.year + 1


def is_active(status):
    """A round gets daily dates when it is under way or starts within the forecast."""
    return status["status"] == "ongoing" or (status["status"] == "upcoming"
                                            and status["days_until_start"] <= FORECAST_DAYS)


def applications_by_round(applications, windows, year):
    """({round number: {fertilizer id: date spread}}, flags) from the app's "Tandai Mulai Memupuk" and "Tandai Sudah
    Dipupuk" records. A date belongs to the round it falls in or after, up to the next round's start (the last round:
    up to the first round's start next year), so a step marked after its window closed still counts."""
    done, flags = {}, []
    starts = sorted((start, number) for number, (start, _) in windows.items())
    if not starts:
        return done, flags
    bounds = [(start, number, nxt) for (start, number), (nxt, _) in
              zip(starts, starts[1:] + [(starts[0][0].replace(year=starts[0][0].year + 1), None)])]
    for item in applications or []:
        fertilizer_id, text = item.get("fertilizerId"), item.get("date")
        try:
            day = date.fromisoformat(text)
        except (TypeError, ValueError):
            flags.append(f"application {fertilizer_id}: date {text!r} is not YYYY-MM-DD; ignored")
            continue
        number = next((n for start, n, nxt in bounds if start <= day < nxt), None)
        if number is not None:
            done.setdefault(number, {})[fertilizer_id] = day
        elif day.year == year:
            flags.append(f"application {fertilizer_id} on {day.isoformat()} is before the year's first round; ignored")
    return done, flags


def card_states(start, end, steps, done, today):
    """(header, steps with their card state) for the round card, from the dates alone: no weather.

    A step can be marked from the previous step's day + its minimum wait (the first step: the round's start). Its
    deadline is that day + the maximum wait, or the round's end when that comes first, there is no maximum, or it is
    the first step; after the deadline it is overdue. A step after one not yet marked is locked.
    """
    status = round_status(start, end, today)
    markable = [s for s in steps if s["fertilizer_id"] is not None]  # a step with no id can't be marked
    marked = [done[s["fertilizer_id"]] for s in markable if s["fertilizer_id"] in done]
    if markable and len(marked) == len(markable):
        header = {"state": "completed", "completed_on": max(marked).isoformat()}
    elif today > end:
        header = {"state": "period_passed", "period_end": end.isoformat()}
    elif marked:
        header = {"state": "started", "started_on": min(marked).isoformat()}
    elif status["status"] == "ongoing":
        header = {"state": "ongoing", "day_of_round": status["day_of_round"], "total_days": status["total_days"]}
    else:
        header = {"state": "upcoming", "days_until_start": status["days_until_start"]}
    plain_waits = header["state"] in ("upcoming", "period_passed", "completed")

    carded, previous_day, previous_wait = [], start, None
    for step in steps:
        spread_on = done.get(step["fertilizer_id"])
        available = deadline = None
        if previous_day is not None:
            available = previous_day + timedelta(days=previous_wait["min"] if previous_wait else 0)
            deadline = (min(previous_day + timedelta(days=previous_wait["max"]), end)
                        if previous_wait and previous_wait["max"] is not None else end)
        if spread_on:
            card = {"state": "done", "done_on": spread_on.isoformat()}
        elif available is None or today < start:
            card = {"state": "locked"}
        elif today > deadline:
            card = {"state": "overdue", "overdue_since": deadline.isoformat()}
        elif today < available:
            card = {"state": "waiting", "available_from": available.isoformat()}
        else:
            card = {"state": "ready", "available_from": available.isoformat(), "deadline": deadline.isoformat()}
        wait = step["wait_after_days"]
        if wait is not None:
            next_from = spread_on + timedelta(days=wait["min"]) if spread_on else None
            if plain_waits or next_from is None:
                wait = {**wait, "state": "idle", "until": None}
            elif today < next_from:
                wait = {**wait, "state": "counting", "until": next_from.isoformat()}
            else:
                wait = {**wait, "state": "done", "until": None}
        carded.append({**step, "card": card, "wait_after_days": wait})
        previous_day, previous_wait = spread_on, step["wait_after_days"]
    return header, carded


def wait_reason(l8, p30, today):
    """Why not to spread now, or None. L8 first (the month and the forecast), then L5 (a dry spell).

    From integration/krops_vari/spreading.py, unchanged."""
    month = MONTHS_EN[today.month - 1]
    state = l8["l8_state"]
    if state == "tunda":
        return f"wait: {month} is a tunda month (avg {l8['month_gate']['avg_mm']:.0f} mm)"
    if state == "cek":
        return f"check soil and weather first: {month} averages {l8['month_gate']['avg_mm']:.0f} mm"
    if state == "cek_hujan":
        return "wait: no safe day in the 10-day forecast"
    if p30 and p30.get("water_stress"):
        return f"wait for rain: P30 {p30['p30_mm']} mm, under 60 mm"
    return None


def daily_plan(start, end, steps, forecast, p30, today, stage_a, done):
    """(the round's daily block, its steps with a schedule each) from the forecast (L8 Stage B) and P30 (L5).

    The rules are the team's: rules.l8_decision picks the safe days, wait_reason says why to wait.
    A step goes on the first safe day at least the previous step's minimum wait after that step's day (the day it
    was spread, from `done`, or its planned day). Past the forecast it gets "not before" that day instead.
    """
    import krops_rules as rules  # the copy next to this file (README.md "Copied from")

    day = max(today, start)
    l8 = rules.l8_decision(forecast, stage_a, day)
    safe_days = [d for d in l8["dates"] if d <= end]
    wait = wait_reason(l8, p30, day)
    scheduled, anchor, gap = [], day, 0
    for step in steps:
        spread_on = done.get(step["fertilizer_id"])
        earliest = max(anchor + timedelta(days=gap), day)
        if spread_on:
            schedule = {"status": "done", "date": spread_on.isoformat(), "not_before": None, "wait": None}
            anchor = spread_on
        elif earliest > end:
            schedule = {"status": "wait", "date": None, "not_before": earliest.isoformat(),
                        "wait": f"the wait runs past this round's window (ends {end.isoformat()})"}
            anchor = earliest
        elif wait:
            schedule = {"status": "wait", "date": None,
                        "not_before": earliest.isoformat() if earliest > day else None, "wait": wait}
            anchor = earliest
        else:
            pick = next((d for d in safe_days if d >= earliest), None)
            if pick:
                schedule = {"status": "planned", "date": pick.isoformat(), "not_before": None, "wait": None}
                anchor = pick
            else:
                schedule = {"status": "not_before", "date": None, "not_before": earliest.isoformat(),
                            "wait": "no safe day that far ahead in the forecast: run again then"}
                anchor = earliest
        gap = step["wait_after_days"]["min"] if step["wait_after_days"] else 0
        scheduled.append({**step, "schedule": schedule})
    block = {
        "checked_on": today.isoformat(),
        "reference_day": day.isoformat(),
        "l5": p30,
        "l8": {"state": l8["l8_state"], "month_gate": l8["month_gate"],
               "safe_days": [d.isoformat() for d in safe_days],
               "removed": {k: [d.isoformat() for d in v] for k, v in (l8["removed"] or {}).items()}},
        "wait": wait,
    }
    return block, scheduled


def build_schedule(request, stage_a, today, year=None, period=None, daily_weather=None):
    """The JSON for the Jadwal Pemupukan screen. stage_a: {month 1–12: {"avg_mm", "in_band_share", "gate"}}.

    daily_weather: a function returning (forecast {date: mm}, P30 record, flags), called once and only when a round
    is active (under way, or starting within FORECAST_DAYS). Without it, no round gets daily dates."""
    rounds = rounds_per_year(request)
    steps, flags = fertilizer_steps(request.get("fertilizers") or [], rounds)
    if not steps:
        flags.append("no fertilizers listed: every round is empty")
    chosen, open_sets = round_months(stage_a, rounds)
    if chosen is None:
        flags.append(f"no {rounds} open months {12 // rounds} months apart: the calendar needs a decision")
    year = year or default_plan_year(chosen, today)
    windows = {number: round_window(year, month) for number, month in enumerate(chosen or (), start=1)}
    done, application_flags = applications_by_round(request.get("applications"), windows, year)
    flags += application_flags
    step_ids = {s["fertilizer_id"] for s in steps if s["fertilizer_id"] is not None}  # no id: can't be marked

    def completed(number):  # a round with every markable step marked needs no weather, as its card says
        return not steps or (step_ids and step_ids <= set(done.get(number, {})))

    active = [n for n, (start, end) in windows.items()
              if is_active(round_status(start, end, today)) and not completed(n)]
    weather = None
    if active and daily_weather is not None:
        forecast, p30, weather_flags = daily_weather()
        flags += weather_flags
        if not forecast:
            flags.append("no 10-day forecast: L8 has no safe day, so every step waits")
        weather = (forecast, p30)

    longest_wait = sum(w["max"] if w["max"] is not None else w["min"]
                       for w in (s["wait_after_days"] for s in steps) if w)
    plan = []
    for number, month in enumerate(chosen or (), start=1):
        start, end = windows[number]
        status = round_status(start, end, today)
        if longest_wait > status["total_days"] - 1:
            flags.append(f"round {number}: the waits add up to {longest_wait} days, longer than its "
                         f"{status['total_days']}-day window")
        later = [(month - 1 + k) % 12 + 1 for k in range(1, ROUND_WINDOW_MONTHS)]
        closed = [MONTHS_ID[m - 1] for m in later if stage_a.get(m, {}).get("gate") != "open"]
        if closed:
            flags.append(f"round {number}: {', '.join(closed)} is not an open month (spread early in the window)")
        card, round_steps = card_states(start, end, steps, done.get(number, {}), today)
        daily = None
        if number in active and weather is not None:
            daily, round_steps = daily_plan(start, end, round_steps, *weather, today, stage_a, done.get(number, {}))
        plan.append({
            "round": number,
            "start_month": start.month,
            "start_month_name": MONTHS_ID[start.month - 1],
            "end_month": end.month,
            "end_month_name": MONTHS_ID[end.month - 1],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            **status,
            "card": card,
            "daily": daily,
            "steps": round_steps,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "plan_year": year,
        "today": today.isoformat(),
        "rounds_per_year": rounds,
        "rounds": plan,
        "weather_basis": {
            "rule": "L8 Stage A: a month is open when its average rain is 100–300 mm (weather/krops_rules.py); "
                    f"rounds go in open months {12 // rounds} months apart, the set with the highest in-band "
                    "(100–200 mm) share",
            "period": list(period) if period else None,
            "months": {str(m): {"name": MONTHS_ID[m - 1], **stage_a[m]} for m in sorted(stage_a)},
            "candidate_sets": [{"months": [MONTHS_ID[m - 1] for m in s], "in_band_share": share}
                               for s, share in open_sets],
            "note": "climate averages (1991–2020) set the months; the days inside an active round's window come "
                    "from the 10-day forecast (L8 Stage B) and P30 (L5), in that round's `daily`",
        },
        "flags": flags,
    }


def load_climate(lat, lon, today, cache_dir=DEFAULT_CACHE):
    """climate.json for the plot's NASA POWER cell: cached, or fetched (30 years) and cached."""
    from weather_sources import WeatherClient  # needs requests: only when the climate is read

    return WeatherClient(cache_dir, use_weatherkit=False).climate(lat, lon, today)


def load_daily_weather(lat, lon, today, normals_mm, cache_dir=DEFAULT_CACHE):
    """(forecast {date: mm}, P30 record, flags): WeatherKit's 10-day forecast and the L5 P30 as of today.

    A failure never stops the run: it becomes a flag, by error type only (a request URL carries the coordinates)."""
    from weather_sources import WeatherClient

    client = WeatherClient(cache_dir)
    forecast, p30, flags = {}, None, []
    if client.weatherkit is None:
        flags.append("WeatherKit not configured (.env, pyjwt, cryptography): no forecast, P30 from NASA POWER")
    try:
        forecast = client.forecast(lat, lon, today)
    except Exception as error:  # noqa: BLE001 – reported, never fatal
        flags.append(f"forecast (WeatherKit) unavailable ({type(error).__name__})")
    try:
        p30 = client.p30(lat, lon, today, normals_mm)
    except Exception as error:  # noqa: BLE001
        flags.append(f"P30 (L5) unavailable ({type(error).__name__})")
    return forecast, p30, flags


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Fertilizer rounds for the Jadwal Pemupukan screen")
    parser.add_argument("--request", type=Path, required=True, help="request JSON (latitude, longitude, fertilizerFreq, fertilizers)")
    parser.add_argument("--today", type=date.fromisoformat, help="YYYY-MM-DD; default today in WIB")
    parser.add_argument("--year", type=int,
                        help="plan year; default this year while one of its rounds hasn't ended, else next year")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="climate cache per NASA POWER cell")
    parser.add_argument("--no-daily", action="store_true", help="skip the daily weather (no forecast or P30 calls)")
    parser.add_argument("--out", type=Path, help="write the schedule here instead of printing it")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("latitude") is None or request.get("longitude") is None:
        raise SystemExit("latitude and longitude are required: the months come from the plot's climate")
    today = args.today or datetime.now(TZ).date()
    lat, lon = float(request["latitude"]), float(request["longitude"])
    climate = load_climate(lat, lon, today, args.cache_dir)
    stage_a = {int(m): row for m, row in climate["stage_a"].items()}
    daily_weather = None if args.no_daily else (
        lambda: load_daily_weather(lat, lon, today, climate["normals_mm"], args.cache_dir))
    schedule = build_schedule(request, stage_a, today, year=args.year, period=climate.get("period"),
                              daily_weather=daily_weather)
    text = json.dumps(schedule, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"{args.out}: {len(schedule['rounds'])} rounds in {schedule['plan_year']}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
