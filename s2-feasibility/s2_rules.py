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

import math
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
MONTH_POOLING = "every clear interior pixel of every scene in the month"  # GUESS – PENDING TEAM DECISION
MONTH_WATER_STRESS = "any scene"  # GUESS – PENDING TEAM DECISION: a plot-month is water_stress when any of its scenes was
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
# UJI 2: CLEAN PIXEL COUNT
# ============================================================

PIXEL_M = 20
UJI2_METHOD = (f"20 m pixels counted only when the whole {PIXEL_M} m square lies inside the polygon (shapely "
               f"contains on each pixel box; not all_touched, not pixel centres). pixels_clean uses the plot "
               f"buffered inward by INNER_BUFFER_M; pixels_raw the plot as drawn.")


def compact_clean_pixels(area_m2, buffer_m=INNER_BUFFER_M):
    """Pixels a round plot of the same area would keep after the inward buffer: the shape-free expectation."""
    r = math.sqrt(area_m2 / math.pi) - buffer_m
    return 0 if r <= 0 else int(math.pi * r * r // (PIXEL_M * PIXEL_M))


def uji2_row(area_m2, pixels_raw, pixels_clean, minimum=MIN_CLEAN_PIXELS, buffer_m=INNER_BUFFER_M):
    """area_ha, pixels_raw, pixels_clean, pass; fails_for_shape when a round plot of the same area would pass."""
    pixels_raw, pixels_clean = int(pixels_raw), int(pixels_clean)
    passed = pixels_clean >= minimum
    return {
        "area_ha": round(float(area_m2) / 10_000, 2),
        "pixels_raw": pixels_raw,
        "pixels_clean": pixels_clean,
        "pass": passed,
        "fails_for_shape": not passed and compact_clean_pixels(area_m2, buffer_m) >= minimum,
    }


# ============================================================
# UJI 3: SIGNAL VS NOISE
# ============================================================

def ndre(b8a, b5):
    """(B8A − B5) / (B8A + B5) on reflectance; NaN where the sum is 0 or a band has no data."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (b8a - b5) / (b8a + b5)
    return np.where(np.isfinite(out), out, np.nan)


def cire(b7, b5):
    """B7 / B5 − 1 on reflectance; NaN where B5 is 0 or a band has no data."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = b7 / b5 - 1
    return np.where(np.isfinite(out), out, np.nan)


def median_iqr(values):
    """{"n", "median", "q1", "q3", "iqr"} over the finite values; None when there are none."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    return {"n": int(v.size), "median": round(med, 4), "q1": round(q1, 4), "q3": round(q3, 4),
            "iqr": round(q3 - q1, 4)}


def month_stats(pixels_by_month, min_pixels=MIN_PIXELS_PER_STAT):
    """{month: median_iqr} for each month with at least min_pixels finite values (pooled over its scenes)."""
    out = {}
    for month, chunks in sorted(pixels_by_month.items()):
        v = np.concatenate([np.asarray(c, dtype="float64").ravel() for c in chunks]) if chunks else np.array([])
        v = v[np.isfinite(v)]
        if v.size >= min_pixels:
            out[month] = median_iqr(v)
    return out


def month_water_stress(flags):
    """MONTH_WATER_STRESS over the scenes' L5 water_stress: True if any is, None if any is unknown, else False."""
    if any(f is True for f in flags):
        return True
    if not flags or any(f is None for f in flags):
        return None
    return False


def plot_age(plot, year, ref_year):
    """Whole years since planting during `year`; age_years is taken as of ref_year. None when neither is given."""
    if plot.get("year_planted") is not None:
        return year - int(plot["year_planted"])
    if plot.get("age_years") is not None:
        return int(plot["age_years"]) - (ref_year - year)
    return None


def age_class(age, classes=AGE_CLASSES):
    if age is None:
        return None
    return next((name for lo, hi, name in classes if lo <= age <= hi), None)


def _ranks(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):  # ties share their mean rank
        tie = x == value
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x, y):
    """Spearman's rho with tied ranks averaged; None for fewer than 3 pairs or a constant side."""
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return round(float(np.corrcoef(_ranks(x), _ranks(y))[0, 1]), 4)


def signal_vs_noise(groups, min_plots=3):
    """Between-plot vs within-plot variation of monthly plot medians, within one age class.

    `groups` is {plot_id: [monthly medians]}; plots with fewer than 2 months are left out. between = variance of
    the plots' medians (of their monthly medians); within = mean over plots of the variance of their monthly
    medians (both ddof=1); ratio = between / within. icc = one-way random-effects ICC(1) on the monthly
    medians, unbalanced ANOVA form. Fewer than min_plots plots → nothing reported, and the record says why.
    """
    groups = {k: np.asarray(v, dtype="float64") for k, v in groups.items() if len(v) >= 2}
    base = {"plots": len(groups), "months": int(sum(v.size for v in groups.values()))}
    if len(groups) < min_plots:
        return base | {"not_reported": f"fewer than {min_plots} plots with 2+ months"}
    values = list(groups.values())
    between = float(np.var([np.median(v) for v in values], ddof=1))
    within = float(np.mean([np.var(v, ddof=1) for v in values]))
    k, sizes = len(values), np.array([v.size for v in values])
    n_total, grand = sizes.sum(), np.concatenate(values).mean()
    msb = sum(v.size * (v.mean() - grand) ** 2 for v in values) / (k - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in values) / (n_total - k)
    n0 = (n_total - (sizes ** 2).sum() / n_total) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if msb + (n0 - 1) * msw > 0 else None
    return base | {
        "between_plot_variance": round(between, 6),
        "within_plot_variance": round(within, 6),
        "ratio": round(between / within, 3) if within > 0 else None,
        "icc": round(float(icc), 3) if icc is not None else None,
    }


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
