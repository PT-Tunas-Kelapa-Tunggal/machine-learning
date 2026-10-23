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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

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
                    "interior": interior, "items": on_tile, "scenes": scenes}


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
    _, offsets = offset_report(items)

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
    print(f"Uji 1{'' if proxy else '–2'} written to {OUT_DIR.relative_to(ROOT)}/ for {len(plots)} plot(s)")


if __name__ == "__main__":
    main()
