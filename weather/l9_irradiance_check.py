"""L9 / data QA (iteration 1, log-only): is NASA POWER's daily sunlight physically possible?

Surface sunlight under real skies can't beat the same day under a clear sky, and neither can beat the sunlight at
the top of the atmosphere: ALLSKY_SFC_SW_DWN ≤ CLRSKY_SFC_SW_DWN ≤ Ra. CLRSKY runs about 73 days behind (11 Oct
2026), so on recent days the FAO-56 clear-sky estimate (0.75·Ra at sea level) stands in for it.

No rule reads the result. It is written to latest.json so L9 can see when the irradiance log went wrong, before any
future rule leans on it. Passing says the values are possible, not that they are accurate.
"""
from __future__ import annotations

import solar

IRRADIANCE_TOLERANCE = 0.02  # GUESS – no source yet: relative slack for rounding and for two models' differences


def check_irradiance(allsky, clrsky, lat, tolerance=IRRADIANCE_TOLERANCE, elevation_m=0.0):
    """Days that break ALLSKY ≤ CLRSKY ≤ Ra, and how many days were checked against what.

    `allsky`/`clrsky` map a date to MJ m-2 day-1 (missing = absent or None); `lat` is where NASA POWER was queried.
    A bound is broken when a value is over it by more than `tolerance` (relative). Codes per flagged day:
    allsky_gt_ra, clrsky_gt_ra, allsky_gt_clrsky, and allsky_gt_rso_fao56 (only on days without CLRSKY).
    Output is ready for JSON: flagged days are ISO strings, values rounded to 0.01.
    """
    allsky = {d: float(v) for d, v in (allsky or {}).items() if v is not None}
    clrsky = {d: float(v) for d, v in (clrsky or {}).items() if v is not None}
    days = sorted(set(allsky) | set(clrsky))
    flagged = {}
    for d in days:
        ra = solar.extraterrestrial_daily(lat, solar.day_of_year(d))
        a, c = allsky.get(d), clrsky.get(d)
        codes = []
        if a is not None and a > ra * (1 + tolerance):
            codes.append("allsky_gt_ra")
        if c is not None and c > ra * (1 + tolerance):
            codes.append("clrsky_gt_ra")
        if a is not None and c is not None and a > c * (1 + tolerance):
            codes.append("allsky_gt_clrsky")
        if a is not None and c is None and a > solar.clear_sky_fao56(ra, elevation_m) * (1 + tolerance):
            codes.append("allsky_gt_rso_fao56")
        if codes:
            flagged[d.isoformat()] = {"allsky": None if a is None else round(a, 2),
                                      "clrsky": None if c is None else round(c, 2),
                                      "ra": round(ra, 2), "codes": codes}
    return {
        "rule": "ALLSKY <= CLRSKY <= Ra (FAO-56 Rso where CLRSKY is missing)",
        "status": "log-only",
        "tolerance": tolerance,
        "first_day": days[0].isoformat() if days else None,
        "last_day": days[-1].isoformat() if days else None,
        "days_checked": len(days),
        "days_with_clrsky": sum(1 for d in days if d in clrsky and d in allsky),
        "days_flagged": len(flagged),
        "flagged": flagged,
    }
