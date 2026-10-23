"""Sentinel-2 plot-level baseline, feasibility tests (Uji 1–3): pure functions and the thresholds they use.

No network, no files. Pixel arrays are numpy; everything else is plain values, as in weather/krops_rules.py.

Conventions
- A month is a (year, month) tuple; in outputs it is written "YYYY-MM".
- Only 20 m bands are read (B5, B7, B8A, SCL); nothing is resampled to 10 m.
- A plot's interior is the set of 20 m pixels whose whole footprint lies inside the plot buffered inward by
  INNER_BUFFER_M (s2_geom.full_footprint_mask). Never all_touched.
- Nothing here measures nutrient status. These tests only ask whether a plot-level red-edge signal could be
  measured at all.
"""
from __future__ import annotations

import re
from collections import Counter

try:
    import numpy as np
except ImportError:  # the constants, month helpers and the coordinate scan work without numpy
    np = None

# ============================================================
# THRESHOLDS
# ============================================================

CLEAR_SHARE_MIN = 0.80     # GUESS – PENDING TEAM DECISION: share of a plot's interior pixels that must be clear for a scene to count
INNER_BUFFER_M = 20        # GUESS – PENDING TEAM DECISION: inward buffer from the plot boundary
MIN_CLEAN_PIXELS = 20      # GUESS – PENDING TEAM DECISION: Uji 2 pass mark
MIN_PIXELS_PER_STAT = 10   # GUESS – PENDING TEAM DECISION: clear interior pixels needed for a plot median in a month
AGE_CLASSES = (            # GUESS – PENDING TEAM DECISION: (first age, last age, name), whole years since planting
    (0, 3, "0-3 immature"),
    (4, 8, "4-8 young"),
    (9, 15, "9-15 prime"),
    (16, 99, "16+ old"),
)
OFFSET_DARK_SHARE_MAX = 0.05     # GUESS – PENDING TEAM DECISION: metadata says the +1000 DN offset is in the pixels;
                                 # believe it only when at most this share of the tile's valid pixels is under 1000 DN
DARK_EVIDENCE_MIN_PIXELS = 500   # GUESS – PENDING TEAM DECISION: fewer valid pixels in the tile overview → no pixel evidence
DEDUP = "highest processing baseline per acquisition"  # GUESS – PENDING TEAM DECISION: ESA reprocessing left
                                                       # two copies of many 2019–2022 acquisitions in the catalogue

# Fixed by the brief or by the product, not guesses
FIRST_MONTH = (2019, 1)
PROXY_BUFFER_M = 100             # no plots.geojson → Uji 1 on this buffer around the .env point
CLEAR_CLASSES = (4, 5)           # SCL vegetation, not vegetated
LOGGED_CLASSES = (2, 3, 7, 8, 9, 10)  # the brief's classes to watch; every class is counted anyway
SCL_NAMES = {
    0: "no_data", 1: "saturated_or_defective", 2: "dark_area", 3: "cloud_shadow", 4: "vegetation",
    5: "not_vegetated", 6: "water", 7: "unclassified", 8: "cloud_medium", 9: "cloud_high", 10: "thin_cirrus",
    11: "snow",
}
OFFSET_FROM_BASELINE = "04.00"   # ESA processing baseline that added BOA_ADD_OFFSET (late Jan 2022)
OFFSET_DN = 1000                 # BOA_ADD_OFFSET = -1000: reflectance = (DN - 1000) / 10000
REFLECTANCE_SCALE = 1e-4


# ============================================================
# MONTHS
# ============================================================

def month_of(day):
    return (day.year, day.month)


def month_label(month):
    return f"{month[0]:04d}-{month[1]:02d}"


def latest_complete_month(today):
    return (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)


def months(first, last):
    """Every month from first to last, inclusive."""
    out, (y, m) = [], first
    while (y, m) <= last:
        out.append((y, m))
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return out


# ============================================================
# CATALOGUE
# ============================================================

def dedup_items(items):
    """One item per acquisition (date and tile): the highest processing baseline, then the latest update.

    Returns (kept, dropped), both lists of the item records given.
    """
    best = {}
    for item in items:
        key = (item["date"], item["tile"])
        rank = (item["baseline"] or "", item["updated"] or "")
        if key not in best or rank > (best[key]["baseline"] or "", best[key]["updated"] or ""):
            best[key] = item
    kept = sorted(best.values(), key=lambda i: (i["date"], i["tile"]))
    kept_ids = {i["id"] for i in kept}
    return kept, [i for i in items if i["id"] not in kept_ids]


# ============================================================
# SCL MASKING
# ============================================================

def scl_counts(scl, interior):
    """Pixels per SCL class inside the interior mask: {class: count}, every class 0–11 present."""
    values, counts = np.unique(scl[interior], return_counts=True)
    out = {c: 0 for c in SCL_NAMES}
    out.update({int(v): int(n) for v, n in zip(values, counts)})
    return out


def clear_share(counts):
    """Share of interior pixels that are clear (SCL 4 or 5); None for an empty interior."""
    total = sum(counts.values())
    return None if total == 0 else sum(counts[c] for c in CLEAR_CLASSES) / total


def is_clear_scene(share, minimum=CLEAR_SHARE_MIN):
    return share is not None and share >= minimum


# ============================================================
# REFLECTANCE OFFSET (processing baseline 04.00 and later)
# ============================================================

def offset_in_metadata(baseline, boa_offset_applied):
    """What the item metadata says: True when the pixels still carry the +1000 DN offset.

    Earth Search marks items it already harmonised with earthsearch:boa_offset_applied = True; a baseline before
    04.00 never had the offset.
    """
    return baseline is not None and baseline >= OFFSET_FROM_BASELINE and boa_offset_applied is not True


def dark_share(b5, scl, min_pixels=DARK_EVIDENCE_MIN_PIXELS):
    """Share of the valid pixels (SCL 4, 5, 6 with data) whose B5 DN is under the offset.

    With the offset still in the pixels, a DN under 1000 means a negative reflectance, so the share is near 0;
    without it, dark water and vegetation put most pixels under 1000. None when too few pixels are valid.
    """
    valid = b5[np.isin(scl, (4, 5, 6)) & (b5 > 0)]
    return None if valid.size < min_pixels else float((valid < OFFSET_DN).mean())


def offset_decision(in_metadata, share, max_share=OFFSET_DARK_SHARE_MAX):
    """"offset" (subtract 1000 DN), "none", or "unresolved" (metadata says offset, no pixel evidence either way).

    The pixels can only overrule the metadata towards "none": a scene whose metadata says it was harmonised is
    never given the offset back. Returns (decision, conflict), conflict = the pixels contradicted the metadata.
    """
    if not in_metadata:
        return "none", False
    if share is None:
        return "unresolved", False
    if share <= max_share:
        return "offset", False
    return "none", True


def harmonize(dn, offset):
    """Reflectance on one scale for every date: DN 0 (no data) → NaN, the offset removed where it is present."""
    dn = dn.astype("float64")
    out = (dn - (OFFSET_DN if offset else 0)) * REFLECTANCE_SCALE
    out[dn == 0] = np.nan
    return out


# ============================================================
# UJI 1: CLEAR-IMAGE AVAILABILITY
# ============================================================

def uji1_table(scenes, first, last, minimum=CLEAR_SHARE_MIN):
    """One row per month: scenes_total, scenes_clear (clear share ≥ minimum), has_clear, best_clear_share.

    `scenes` is a list of {"date": date, "clear_share": float or None}.
    """
    by_month = {}
    for s in scenes:
        by_month.setdefault(month_of(s["date"]), []).append(s["clear_share"])
    rows = []
    for month in months(first, last):
        shares = by_month.get(month, [])
        known = [s for s in shares if s is not None]
        clear = sum(1 for s in shares if is_clear_scene(s, minimum))
        rows.append({
            "month": month_label(month),
            "has_clear": clear > 0,
            "scenes_total": len(shares),
            "scenes_clear": clear,
            "best_clear_share": round(max(known), 3) if known else None,
        })
    return rows


def month_of_year_summary(rows):
    """For each calendar month: the years covered, the years with a clear scene, and their share."""
    out = []
    for m in range(1, 13):
        these = [r for r in rows if int(r["month"][5:]) == m]
        clear = sum(r["has_clear"] for r in these)
        out.append({"month_of_year": m, "years": len(these), "years_with_clear": clear,
                    "share": round(clear / len(these), 3) if these else None})
    return out


def longest_gap(rows):
    """The longest run of consecutive months without a clear scene: {"months", "first", "last"} (first run wins)."""
    best, run = {"months": 0, "first": None, "last": None}, []
    for r in rows + [{"has_clear": True}]:
        if not r["has_clear"]:
            run.append(r["month"])
            continue
        if len(run) > best["months"]:
            best = {"months": len(run), "first": run[0], "last": run[-1]}
        run = []
    return best


# ============================================================
# NO COORDINATES IN OUTPUTS
# ============================================================

_NUMBER_PAIR = re.compile(r"(-?\d{1,3}\.\d{2,})[\s,;/]+(-?\d{1,3}\.\d{2,})")
_PROJECTED_PAIR = re.compile(r"\d{6,7}\.\d+[\s,;/]+\d{6,7}\.\d+")
_FORBIDDEN = re.compile(
    r'"(coordinates|bbox|geometry|lat|lon|latitude|longitude|transform|crs)"\s*:'  # GeoJSON / raster keys
    r"|\b(MULTIPOLYGON|POLYGON|POINT)\s*\(+\s*-?\d"                               # WKT
    r"|EPSG:\d+|MGRS-\d|\bS2[ABC]_\d{2}[A-Z]{3}_"                                   # CRS, tile and item ids
    r"|https?://|s3://|/vsi"                                                       # tile paths (GDAL messages)
)


def find_coordinates(text, known=()):
    """Everything in `text` that looks like a location, as a list of short reasons (empty = clean).

    Flags a pair of decimals that reads as latitude, longitude (one |v| ≤ 90, the other 20 ≤ |v| ≤ 180, either
    order), a pair of projected metres, GeoJSON/WKT/CRS/tile markers, and any of the `known` strings (the
    .env longitude, say). A bare latitude near the equator can't be told from an index value, so it is only
    caught next to its longitude.
    """
    found = []
    for a, b in _NUMBER_PAIR.findall(text):
        x, y = abs(float(a)), abs(float(b))
        if (x <= 90 and 20 <= y <= 180) or (y <= 90 and 20 <= x <= 180):
            found.append("degree pair")
    if _PROJECTED_PAIR.search(text):
        found.append("projected pair")
    found += [f"marker {m.group(0)!r}" for m in _FORBIDDEN.finditer(text)]
    found += [f"known value {k!r}" for k in known if k and re.search(rf"(?<![\d.]){re.escape(k)}", text)]
    return found


def class_totals(counts_list):
    """Sum per SCL class over scenes, keyed by class name."""
    total = Counter()
    for counts in counts_list:
        total.update(counts)
    return {SCL_NAMES[c]: int(total[c]) for c in SCL_NAMES}
