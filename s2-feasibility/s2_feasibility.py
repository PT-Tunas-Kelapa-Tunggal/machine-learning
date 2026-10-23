"""Sentinel-2 plot-level baseline, feasibility tests Uji 1–3, written to outputs/s2_feasibility/ (repo root).

    cd s2-feasibility && python s2_feasibility.py                         # fetches only what the cache lacks
    cd s2-feasibility && AS_OF_DATE=2026-10-23 python s2_feasibility.py   # cache only; never fetches

Plots come from data/private/plots.geojson (repo root, gitignored): plot_id, year_planted or age_years, source.
Without it, Uji 1 runs on one proxy plot (a 100 m buffer around the point in weather/.env), marked as a proxy,
and Uji 2–3 are left to the synthetic tests.

Outputs are keyed by plot_id. They carry no coordinates, bounding boxes, geometries, tile ids or paths, and the
run fails if the final scan finds anything that looks like one.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import shapely
from dotenv import load_dotenv

import s2_fetch as fetch
import s2_geom as geom
import s2_rules as rules

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PLOTS_PATH = ROOT / "data" / "private" / "plots.geojson"
OUT_DIR = ROOT / "outputs" / "s2_feasibility"
for env in (HERE / ".env", ROOT / "weather" / ".env"):
    load_dotenv(env)
LAT, LON = float(os.environ["LAT"]), float(os.environ["LON"])
THREADS = 12

WEATHER_DIR = ROOT / "weather"
RAIN_AS_OF = date(2026, 10, 11)  # weather notebook 1's run: Uji 3 reads that day's saved rain, as notebook 03 does
RAIN_END = date(2026, 10, 8)     # NASA POWER's last day of rain in that run
WK_FROM = date(2021, 8, 1)       # WeatherKit history starts here


# ============================================================
# CATALOGUE AND TILES
# ============================================================

def catalogue(plots, first, last):
    """Items over all plots, COGs only, one per acquisition. Returns (kept, log)."""
    after_last = rules.months(last, (last[0] + 1, 1))[1]
    end = date(*after_last, 1) - timedelta(days=1)
    features = fetch.search_items(shapely.union_all([p["geometry"] for p in plots]), date(*first, 1).isoformat(),
                                  end.isoformat())
    records = [fetch.item_record(f) for f in features]
    cog = [r for r in records if r["cog"]]
    kept, dropped = rules.dedup_items(cog)
    return kept, {
        "items_found": len(records),
        "items_not_cog_skipped": len(records) - len(cog),
        "duplicates_dropped": len(dropped),
        "duplicates_dropped_by_baseline": dict(Counter(d["baseline"] for d in dropped)),
        "items_kept": len(kept),
        "tiles": len({r["tile"] for r in kept}),
    }


def place_plot(plot, items):
    """The tile that holds the whole plot with the most acquisitions: (tile, geometry in its CRS, window) or None."""
    for tile, _ in Counter(i["tile"] for i in items).most_common():
        ref = next(i for i in items if i["tile"] == tile and i["transform"] is not None)
        g = geom.project(plot["geometry"], geom.WGS84, ref["crs"])
        window = geom.plot_window(g, ref["transform"])
        if geom.window_inside(window, *ref["shape"]):
            return tile, g, window, ref["transform"]
    return None


def read_all(items, band, window):
    """{item id: array or the ReadFailed reason}, read in parallel; NotCached (as-of run) propagates."""
    def one(item):
        try:
            return item["id"], fetch.read_window(item, band, window)
        except fetch.ReadFailed as e:
            return item["id"], f"read failed ({e})"

    with ThreadPoolExecutor(THREADS) as ex:
        return dict(ex.map(one, items))


# ============================================================
# OFFSET
# ============================================================

def offset_report(items):
    """Metadata says, pixels check, per item. Returns ({item id: decision}, report)."""
    flagged = [i for i in items if rules.offset_in_metadata(i["baseline"], i["boa_offset_applied"])]
    with ThreadPoolExecutor(THREADS) as ex:
        shares = dict(ex.map(lambda i: (i["id"], _dark(i)), flagged))
    decisions, rows = {}, []
    for i in items:
        if i["id"] not in shares:
            decisions[i["id"]] = "none"
            continue
        share = shares[i["id"]]
        decision, conflict = rules.offset_decision(True, share if isinstance(share, float) else None)
        decisions[i["id"]] = decision
        rows.append({"date": i["date"].isoformat(), "baseline": i["baseline"],
                     "boa_offset_applied": i["boa_offset_applied"],
                     "dark_share": round(share, 3) if isinstance(share, float) else share,
                     "decision": decision, "metadata_contradicted": conflict})
    return decisions, {
        "rule": f"metadata (baseline ≥ {rules.OFFSET_FROM_BASELINE} and not earthsearch:boa_offset_applied) "
                f"says the +{rules.OFFSET_DN} DN offset is in the pixels; kept only when ≤ "
                f"{rules.OFFSET_DARK_SHARE_MAX} of the tile's valid pixels are under {rules.OFFSET_DN} DN (GUESS)",
        "items_metadata_says_offset": len(flagged),
        "dates_needing_offset": sorted(r["date"] for r in rows if r["decision"] == "offset"),
        "dates_metadata_contradicted": sorted(r["date"] for r in rows if r["metadata_contradicted"]),
        "dates_unresolved": sorted(r["date"] for r in rows if r["decision"] == "unresolved"),
        "items": sorted(rows, key=lambda r: r["date"]),
    }


def _dark(item):
    try:
        return fetch.dark_evidence(item)
    except fetch.ReadFailed as e:
        return f"read failed ({e})"


# ============================================================
# UJI 1
# ============================================================

def uji1(plot, items, first, last):
    placed = place_plot(plot, items)
    if placed is None:
        return {"proxy": plot["proxy"], "error": "no tile holds the whole plot"}, None
    tile, g, window, transform = placed
    raw, interior = geom.plot_masks(g, transform, window)
    on_tile = [i for i in items if i["tile"] == tile]
    scl = read_all(on_tile, "SCL", window)
    scenes, failed = [], Counter()
    for i in on_tile:
        if isinstance(scl[i["id"]], str):
            failed[scl[i["id"]]] += 1
            continue
        counts = rules.scl_counts(scl[i["id"]], interior)
        scenes.append({"id": i["id"], "date": i["date"], "counts": counts, "clear_share": rules.clear_share(counts)})
    rows = rules.uji1_table(scenes, first, last)
    result = {
        "proxy": plot["proxy"],
        "source": plot["source"],
        "interior_pixels": int(interior.sum()),
        "scenes_read": len(scenes),
        "scenes_failed": dict(failed),
        "months": rows,
        "months_with_clear": sum(r["has_clear"] for r in rows),
        "months_total": len(rows),
        "month_of_year": rules.month_of_year_summary(rows),
        "longest_gap": rules.longest_gap(rows),
        "scl_interior_pixels_by_class": rules.class_totals(s["counts"] for s in scenes),
    }
    return result, {"tile": tile, "geometry": g, "window": window, "transform": transform, "raw": raw,
                    "interior": interior, "items": on_tile, "scenes": scenes, "scl": scl}


# ============================================================
# UJI 2
# ============================================================

NOT_RUN_PROXY = ("data/private/plots.geojson is missing: the brief runs Uji 2–3 on real plots only, so this "
                 "run checked them on synthetic tests alone")


def uji2(plots, placed):
    out = {}
    for plot in plots:
        ctx = placed[plot["plot_id"]]
        if ctx is None:
            out[plot["plot_id"]] = {"error": "no tile holds the whole plot"}
            continue
        out[plot["plot_id"]] = rules.uji2_row(ctx["geometry"].area, ctx["raw"].sum(), ctx["interior"].sum())
    return out


# ============================================================
# UJI 3
# ============================================================

def rain_sources():
    """Daily rain per source, built as weather notebook 03 builds it: the same calls on the same saved responses.

    P30 and water_stress then come from weather/l2_scan_rain.p30_at (krops_rules.choose_p30), imported, not copied.
    """
    sys.path.insert(0, str(WEATHER_DIR))
    import pandas as pd
    import krops_weather as kw

    history, _, _ = kw.fetch_nasa_power(date(1991, 1, 1), date(2025, 12, 31), params=("PRECTOTCORR",))
    this_year, _, _ = kw.fetch_nasa_power(date(2026, 1, 1), RAIN_AS_OF, forever=True)
    nasa_all = pd.concat([history["PRECTOTCORR"], this_year["PRECTOTCORR"]]).dropna()
    nasa_all = nasa_all[~nasa_all.index.duplicated(keep="last")].sort_index()
    wk_rain = kw.fetch_weatherkit_history(WK_FROM, RAIN_END, forever=True)["rain_mm"].dropna()
    check = kw.suspect_months(nasa_all[nasa_all.index >= WK_FROM], kw.monthly_normals(nasa_all), wk_rain)
    nasa_ok = kw.drop_months(nasa_all, check.index[check["suspect"]])
    return {"weatherkit-history": wk_rain.to_dict(), "nasa-power": nasa_ok.dropna().to_dict()}


def uji3_plot(ctx, decisions, sources):
    """Monthly NDRE and CIre (median, IQR) over the clear interior pixels, each month annotated with L5 P30."""
    sys.path.insert(0, str(WEATHER_DIR))
    import l2_scan_rain as l2

    items = {i["id"]: i for i in ctx["items"]}
    candidates = [s for s in ctx["scenes"] if s["counts"][4] + s["counts"][5] > 0]
    usable = [s for s in candidates if decisions[s["id"]] != "unresolved"]
    bands = {b: read_all([items[s["id"]] for s in usable], b, ctx["window"]) for b in ("B05", "B07", "B8A")}
    pixels = {"ndre": defaultdict(list), "cire": defaultdict(list)}
    dates, failed = defaultdict(set), Counter()
    for s in usable:
        arrays = [bands[b][s["id"]] for b in ("B05", "B07", "B8A")]
        if any(isinstance(a, str) for a in arrays):
            failed[next(a for a in arrays if isinstance(a, str))] += 1
            continue
        clear = ctx["interior"] & np.isin(ctx["scl"][s["id"]], rules.CLEAR_CLASSES)
        b5, b7, b8a = (rules.harmonize(a, decisions[s["id"]] == "offset")[clear] for a in arrays)
        month = rules.month_of(s["date"])
        pixels["ndre"][month].append(rules.ndre(b8a, b5))
        pixels["cire"][month].append(rules.cire(b7, b5))
        dates[month].add(s["date"])
    stats = {name: rules.month_stats(by_month) for name, by_month in pixels.items()}
    rows = []
    for month in sorted(stats["ndre"]):
        p30 = [l2.p30_at(sources, d) for d in sorted(dates[month])]
        nd, ci = stats["ndre"][month], stats["cire"].get(month) or {}
        known = [r["p30_mm"] for r in p30 if r["p30_mm"] is not None]
        rows.append({
            "month": rules.month_label(month), "scenes": len(dates[month]), "pixels": nd["n"],
            "ndre_median": nd["median"], "ndre_q1": nd["q1"], "ndre_q3": nd["q3"], "ndre_iqr": nd["iqr"],
            "cire_median": ci.get("median"), "cire_q1": ci.get("q1"), "cire_q3": ci.get("q3"),
            "cire_iqr": ci.get("iqr"),
            "p30_mm_min": min(known) if known else None, "p30_mm_max": max(known) if known else None,
            "p30_sources": "+".join(sorted({r["p30_source"] or "none" for r in p30})),
            "water_stress": rules.month_water_stress([r["water_stress"] for r in p30]),
        })
    return {
        "scenes_with_clear_interior": len(candidates),
        "scenes_skipped_offset_unresolved": len(candidates) - len(usable),
        "scenes_failed": dict(failed),
        "months_with_stats": len(rows),
        "months": rows,
    }


def uji3_summary(plots, per_plot, ref_year):
    """a) Spearman of age vs annual median NDRE, per year; b) signal vs noise per age class, ± water_stress months."""
    by_id = {p["plot_id"]: p for p in plots}
    years = sorted({int(r["month"][:4]) for v in per_plot.values() for r in v["months"]})
    age_signal = []
    for year in years:
        pairs = []
        for pid, v in per_plot.items():
            medians = [r["ndre_median"] for r in v["months"] if r["month"].startswith(f"{year}-")]
            age = rules.plot_age(by_id[pid], year, ref_year)
            if medians and age is not None:
                pairs.append((age, float(np.median(medians))))
        rho = rules.spearman([a for a, _ in pairs], [m for _, m in pairs])
        age_signal.append({"year": year, "plots": len(pairs), "spearman_rho": rho,
                           "note": "" if rho is not None else "not computed: fewer than 3 plots or no spread"})
    classes = [name for _, _, name in rules.AGE_CLASSES]
    noise = []
    for name in classes:
        for index in ("ndre", "cire"):
            for drop_stress in (False, True):
                groups = defaultdict(list)
                for pid, v in per_plot.items():
                    for r in v["months"]:
                        if rules.age_class(rules.plot_age(by_id[pid], int(r["month"][:4]), ref_year)) != name:
                            continue
                        if (drop_stress and r["water_stress"] is True) or r[f"{index}_median"] is None:
                            continue
                        groups[pid].append(r[f"{index}_median"])
                noise.append({"age_class": name, "index": index.upper() if index == "ndre" else "CIre",
                              "water_stress_months": "excluded" if drop_stress else "included"}
                             | rules.signal_vs_noise(groups))
    return {"age_signal": age_signal, "signal_vs_noise": noise,
            "plots_without_age": sorted(pid for pid in per_plot
                                        if rules.plot_age(by_id[pid], ref_year, ref_year) is None)}


# ============================================================
# OUTPUT
# ============================================================

def write_json(name, data):
    (OUT_DIR / name).write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n")


def write_csv(name, rows):
    with open(OUT_DIR / name, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def scan_outputs():
    """Fail the run if any output looks like it holds a location."""
    known = [os.environ["LON"].strip(), os.environ["LAT"].strip() if len(os.environ["LAT"].split(".")[-1]) > 2 else ""]
    bad = {p.name: hits for p in sorted(OUT_DIR.iterdir()) if p.is_file()
           if (hits := rules.find_coordinates(p.read_text(), known))}
    if bad:
        raise SystemExit(f"outputs look like they hold a location, not finishing: {bad}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = geom.load_plots(PLOTS_PATH, LAT, LON)
    first, last = rules.FIRST_MONTH, rules.latest_complete_month(fetch.TODAY)
    items, catalogue_log = catalogue(plots, first, last)
    decisions, offsets = offset_report(items)

    results, months_csv, moy_csv, placed = {}, [], [], {}
    for plot in plots:
        result, ctx = uji1(plot, items, first, last)
        results[plot["plot_id"]] = result
        placed[plot["plot_id"]] = ctx
        months_csv += [{"plot_id": plot["plot_id"]} | r for r in result.get("months", [])]
        moy_csv += [{"plot_id": plot["plot_id"]} | r for r in result.get("month_of_year", [])]
    write_json("uji1.json", results)
    write_csv("uji1_months.csv", months_csv)
    write_csv("uji1_month_of_year.csv", moy_csv)

    proxy = plots[0]["proxy"]
    if proxy:
        write_json("uji2.json", {"not_run": NOT_RUN_PROXY, "method": rules.UJI2_METHOD})
    else:
        u2 = uji2(plots, placed)
        write_json("uji2.json", {"method": rules.UJI2_METHOD, "plots": u2})
        write_csv("uji2.csv", [{"plot_id": k} | v for k, v in u2.items()])

    if proxy:
        write_json("uji3.json", {"not_run": NOT_RUN_PROXY})
    else:
        sources = rain_sources()
        per_plot = {p["plot_id"]: uji3_plot(placed[p["plot_id"]], decisions, sources)
                    for p in plots if placed[p["plot_id"]] is not None}
        summary = uji3_summary(plots, per_plot, fetch.TODAY.year)
        write_json("uji3.json", {"month_pooling": rules.MONTH_POOLING, "month_water_stress": rules.MONTH_WATER_STRESS,
                                 "plots": per_plot} | summary)
        write_csv("uji3_plot_months.csv", [{"plot_id": pid} | r for pid, v in per_plot.items() for r in v["months"]])
        write_csv("uji3_age_signal.csv", summary["age_signal"])
        write_csv("uji3_signal_vs_noise.csv", summary["signal_vs_noise"])
    write_json("run.json", {
        "as_of": fetch.TODAY.isoformat(),
        "as_of_run": bool(fetch.AS_OF_DATE),
        "period": [rules.month_label(first), rules.month_label(last)],
        "plots": len(plots),
        "plots_source": "proxy" if plots[0]["proxy"] else "data/private/plots.geojson",
        "thresholds": {"CLEAR_SHARE_MIN": rules.CLEAR_SHARE_MIN, "INNER_BUFFER_M": rules.INNER_BUFFER_M,
                       "MIN_CLEAN_PIXELS": rules.MIN_CLEAN_PIXELS, "MIN_PIXELS_PER_STAT": rules.MIN_PIXELS_PER_STAT,
                       "OFFSET_DARK_SHARE_MAX": rules.OFFSET_DARK_SHARE_MAX, "DEDUP": rules.DEDUP},
        "catalogue": catalogue_log,
        "offset": offsets,
    })
    scan_outputs()
    print(f"Uji 1{'' if proxy else '–3'} written to {OUT_DIR.relative_to(ROOT)}/ for {len(plots)} plot(s)")


if __name__ == "__main__":
    main()
