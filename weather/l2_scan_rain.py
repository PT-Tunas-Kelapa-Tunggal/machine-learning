"""L2 trend (iteration 2+): the rain between two scans, and P30 as it stood at each scan.

A crown index that falls between two scans may be weather rather than nutrients; these numbers let L2 tell them
apart. Helpers only for now: the backend doesn't see scan dates yet, so the job doesn't call them (notebook 03
shows them on the trial plot's history).

Same conventions as krops_rules.py: daily rain is {date: mm}, and a missing day is never 0 mm. No thresholds: P30
at a scan goes through rules.choose_p30 unchanged, with the scan date as "today".
"""
from __future__ import annotations

from datetime import timedelta

import krops_rules as rules


def rain_between(daily, start, end):
    """Rain on the days start ≤ d < end: half-open, so back-to-back scan intervals add up without overlap.

    rain_mm is None when any day is missing (never a sum of fewer days); an empty interval (start == end) is 0 mm.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    rain = {d: float(v) for d, v in (daily or {}).items() if v is not None}
    days = [start + timedelta(days=i) for i in range((end - start).days)]
    missing = sum(1 for d in days if d not in rain)
    return {
        "start": start,
        "end": end,
        "days": len(days),
        "missing_days": missing,
        "rain_mm": None if missing else round(sum(rain[d] for d in days), 1),
    }


def p30_at(sources, scan_date, order=rules.P30_SOURCE_ORDER):
    """The L5 P30 record as it stood on scan_date: each source cut to the days before it, then rules.choose_p30.

    The scan day itself is left out, as the job leaves out today: its rain isn't complete when the drone flies.
    """
    before = {name: {d: v for d, v in (daily or {}).items() if d < scan_date} for name, daily in sources.items()}
    return rules.choose_p30(before, scan_date, order=order)


def _rain_from_first_full_source(sources, start, end, order):
    for name in order:
        record = rain_between(sources.get(name), start, end)
        if record["rain_mm"] is not None:
            return record | {"source": name}
    return {"start": start, "end": end, "days": (end - start).days, "missing_days": None, "rain_mm": None,
            "source": None}


def scan_rain_log(sources, scan_dates, order=rules.P30_SOURCE_ORDER):
    """One entry per scan date, oldest first: P30 at the scan, and the rain since the previous scan.

    The rain since the previous scan comes from the first source in `order` that has every day of the interval,
    so one interval never mixes sources; None (unknown) when none has.
    """
    log, previous = [], None
    for scan in sorted(set(scan_dates)):
        log.append({
            "scan_date": scan,
            "p30": p30_at(sources, scan, order),
            "since_previous": None if previous is None else _rain_from_first_full_source(sources, previous, scan, order),
        })
        previous = scan
    return log
