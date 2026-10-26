"""The spreading plan (weather_fertilizer.ipynb): which product, on which day, from L5 and L8.

- This round: the annual dose ÷ rounds_per_year. Mg on the first safe day of L8's window; N on the first safe day at
  least min_days_mg_to_n[Mg product] later (14 after dolomite, 1 after kieserite), else "not before" that date.
- Wait instead of dates when L8 says tunda, cek or cek_hujan, or L5 says water_stress.
- The year: two open months round_gap_months apart, the pair with the highest in-band (100–200 mm) share.

Rounds, the round months and the kieserite gap are GUESSes; the 14 days after dolomite is practice, not a PPKS/MPOB
number. Pure functions: no network.
"""
from datetime import timedelta

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def assign_days(window, min_gap):
    """(Mg day, N day) from the safe days; N is None when no safe day is min_gap days after Mg."""
    if not window:
        return None, None
    mg_day = window[0]
    later = [d for d in window if (d - mg_day).days >= min_gap]
    return mg_day, (later[0] if later else None)


def wait_reason(l8, p30, today):
    """Why not to spread now, or None. L8 first (the month and the forecast), then L5 (a dry spell)."""
    month = MONTHS[today.month - 1]
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


def schedule(nutrient, mg_product, safe_days, wait, settings):
    """{"date", "not_before", "wait"} for spreading `nutrient` in a round where Mg is `mg_product`."""
    if wait:
        return {"date": None, "not_before": None, "wait": wait}
    gap = settings.min_days_mg_to_n[mg_product]
    mg_day, n_day = assign_days(safe_days, gap)
    day = mg_day if nutrient == "Mg" else n_day
    if day:
        return {"date": day.isoformat(), "not_before": None, "wait": None}
    return {"date": None, "not_before": (mg_day + timedelta(days=gap)).isoformat(),
            "wait": "no safe day that far ahead in the forecast: run again then"}


def this_round(products, safe_days, wait, settings):
    """One row per zone and product for the coming round. Mg rows carry their own product's schedule; N rows one
    schedule per Mg product, since the N day depends on which Mg product was spread."""
    rows = []
    for p in products:
        if p["product_kg_per_zone_per_year"] is None:
            continue
        mg_products = [p["product"]] if p["nutrient"] == "Mg" else list(settings.min_days_mg_to_n)
        rows.append({
            "zone_id": p["zone_id"], "nutrient": p["nutrient"], "product": p["product"],
            "kg_per_zone": p["product_kg_per_zone_per_year"] / settings.rounds_per_year,
            "kg_per_palm": (p["product_kg_per_palm_per_year"] / settings.rounds_per_year
                            if p["product_kg_per_palm_per_year"] is not None else None),
            "schedule_if_mg_is": {m: schedule(p["nutrient"], m, safe_days, wait, settings) for m in mg_products},
        })
    return rows


def round_months(stage_a, settings):
    """(chosen pair of months, every open pair with its combined in-band share), months 1–12; (None, []) if none."""
    gap = settings.round_gap_months
    pairs = sorted({tuple(sorted((m, (m + gap - 1) % 12 + 1))) for m in range(1, 13)})
    open_pairs = [(p, sum(stage_a[m]["in_band_share"] for m in p)) for p in pairs
                  if all(stage_a.get(m, {}).get("gate") == "open" for m in p)]
    if not open_pairs:
        return None, []
    return max(open_pairs, key=lambda item: item[1])[0], open_pairs


def round_totals(products, settings):
    """kg of each product per round for the whole photo (pick one product per nutrient)."""
    totals = {}
    for p in products:
        if p["product_kg_per_zone_per_year"] is not None:
            totals[p["product"]] = totals.get(p["product"], 0.0) + p["product_kg_per_zone_per_year"] / settings.rounds_per_year
    return totals


def spreading_plan(products, stage_a, l8, p30, today, settings):
    """The plan: L5 and L8 as read, this round per zone, and the yearly calendar."""
    wait = wait_reason(l8, p30, today)
    safe_days = l8["dates"]
    chosen, open_pairs = round_months(stage_a, settings)
    flags = [] if chosen else [f"no two open months {settings.round_gap_months} months apart: the calendar needs a decision"]
    return {
        "today": today.isoformat(),
        "l5": p30,
        "l8": {"state": l8["l8_state"], "month_gate": l8["month_gate"],
               "safe_days": [d.isoformat() for d in safe_days],
               "removed": {k: [d.isoformat() for d in v] for k, v in (l8["removed"] or {}).items()}},
        "wait": wait,
        "rounds_per_year": settings.rounds_per_year,
        "min_days_mg_to_n": settings.min_days_mg_to_n,
        "this_round": this_round(products, safe_days, wait, settings),
        "per_round_totals_kg": round_totals(products, settings),
        "calendar": {
            "round_months": [MONTHS[m - 1] for m in chosen] if chosen else None,
            "open_pairs": [{"months": [MONTHS[m - 1] for m in p], "in_band_share": round(s, 3)} for p, s in open_pairs],
            "note": "climate averages (1991–2020); the dates come from running the plan in that month",
        },
        "flags": flags,
    }
