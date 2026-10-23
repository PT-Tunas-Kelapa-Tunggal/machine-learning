"""REPORT.md for the Sentinel-2 feasibility run, built only from the JSON outputs next to it.

    cd s2-feasibility && python s2_report.py    # rebuild the report from s2-feasibility/outputs/*.json

The report repeats numbers from the JSON outputs; it adds no coordinates and decides nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import s2_rules as rules

OUT_DIR = Path(__file__).resolve().parent / "outputs"
MONTH_NAMES = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _load(out_dir, name):
    path = out_dir / name
    return json.loads(path.read_text()) if path.is_file() else None


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _dates(dates):
    return ", ".join(dates) if dates else "none"


def _uji1(uji1):
    out = ["## Uji 1 — clear-image availability",
           f"A scene counts as clear for a plot when at least CLEAR_SHARE_MIN = {rules.CLEAR_SHARE_MIN} (GUESS) of the "
           f"plot's interior pixels are SCL 4 or 5. A month passes when it has at least one clear scene."]
    for pid, r in uji1.items():
        out.append(f"### {pid}" + (" (proxy)" if r.get("proxy") else ""))
        if "error" in r:
            out.append(f"Not run: {r['error']}.")
            continue
        classes = r["scl_interior_pixels_by_class"]
        clear = classes["vegetation"] + classes["not_vegetated"]
        removed = sorted(((n, c) for c, n in classes.items() if c not in ("vegetation", "not_vegetated") and n),
                         reverse=True)
        gap = r["longest_gap"]
        out += [
            f"- Interior pixels: {r['interior_pixels']}; scenes read: {r['scenes_read']}"
            + (f"; failed reads: {r['scenes_failed']}" if r["scenes_failed"] else "") + ".",
            f"- Months with a clear scene: **{r['months_with_clear']} of {r['months_total']}** "
            f"({r['months_with_clear'] / r['months_total']:.0%}).",
            f"- Longest run without one: {gap['months']} month(s)"
            + (f", {gap['first']} to {gap['last']}" if gap["months"] else "") + ".",
            f"- Months without a clear scene: "
            f"{_dates([m['month'] for m in r['months'] if not m['has_clear']])}.",
            f"- Clear interior pixels, all scenes: {clear} — vegetation {classes['vegetation']}, not vegetated "
            f"{classes['not_vegetated']}"
            + (f" ({classes['vegetation'] / clear:.0%} vegetation)" if clear else "") + ".",
            "- What removed pixels (interior pixel-scenes, most first): "
            + ", ".join(f"{c} {n}" for n, c in removed) + ".",
            "",
            _table(["Month of year", "Years", "Years with a clear scene", "Share"],
                   [(MONTH_NAMES[m["month_of_year"] - 1], m["years"], m["years_with_clear"], m["share"])
                    for m in r["month_of_year"]]),
        ]
    return out


def _uji2(uji2):
    out = ["## Uji 2 — clean pixel count", f"Method: {uji2['method']}"]
    if "not_run" in uji2:
        return out + [f"Not run: {uji2['not_run']}."]
    rows = [(pid, r.get("area_ha"), r.get("pixels_raw"), r.get("pixels_clean"),
             "pass" if r.get("pass") else "fail", "yes" if r.get("fails_for_shape") else "", r.get("error"))
            for pid, r in uji2["plots"].items()]
    return out + [f"Pass mark: pixels_clean ≥ MIN_CLEAN_PIXELS = {rules.MIN_CLEAN_PIXELS} (GUESS), inward buffer "
                  f"INNER_BUFFER_M = {rules.INNER_BUFFER_M} m (GUESS).", "",
                  _table(["plot_id", "area_ha", "pixels_raw", "pixels_clean", "result", "fails for shape", "error"],
                         rows)]


def _uji3(uji3):
    out = ["## Uji 3 — signal vs noise"]
    if "not_run" in uji3:
        return out + [f"Not run: {uji3['not_run']}."]
    out += [f"Plot-month statistics need ≥ MIN_PIXELS_PER_STAT = {rules.MIN_PIXELS_PER_STAT} (GUESS) clear interior "
            f"pixels; pooling: {uji3['month_pooling']} (GUESS); a month is water_stress when "
            f"{uji3['month_water_stress']} was (GUESS), from weather/ P30.",
            f"Rain for P30: {uji3['rain_location']} — {uji3['rain_places']['nasa_cells']} NASA POWER cell(s), "
            f"{uji3['rain_places']['weatherkit_areas']} WeatherKit area(s)"
            + ("" if uji3["rain_places"]["weatherkit"] else "; no WeatherKit keys, so NASA POWER alone") + ".", "",
            _table(["plot_id", "months with stats", "scenes with clear interior", "skipped: offset unresolved",
                    "water_stress months"],
                   [(pid, v["months_with_stats"], v["scenes_with_clear_interior"],
                     v["scenes_skipped_offset_unresolved"], sum(1 for m in v["months"] if m["water_stress"] is True))
                    for pid, v in uji3["plots"].items()]),
            "", "### a) Age signal: Spearman of plot age vs annual median NDRE",
            _table(["Year", "Plots", "rho", "Note"],
                   [(a["year"], a["plots"], a["spearman_rho"], a["note"]) for a in uji3["age_signal"]]),
            "", "### b) Signal vs noise per age class (AGE_CLASSES is a GUESS)",
            _table(["Age class", "Index", "water_stress months", "Plots", "Months", "Between", "Within", "Ratio",
                    "ICC", "Note"],
                   [(n["age_class"], n["index"], n["water_stress_months"], n["plots"], n["months"],
                     n.get("between_plot_variance"), n.get("within_plot_variance"), n.get("ratio"), n.get("icc"),
                     n.get("not_reported")) for n in uji3["signal_vs_noise"]])]
    if uji3["plots_without_age"]:
        out.append(f"\nPlots without year_planted or age_years (left out of a and b): "
                   f"{', '.join(uji3['plots_without_age'])}.")
    return out


def build(out_dir=OUT_DIR):
    run, uji1, uji2, uji3 = (_load(out_dir, n) for n in ("run.json", "uji1.json", "uji2.json", "uji3.json"))
    cat, off = run["catalogue"], run["offset"]
    proxy = run["plots_source"] == "proxy"
    lines = [
        "# Sentinel-2 plot-level baseline — feasibility tests (Uji 1–3)",
        "",
        "These tests ask only whether a plot-level red-edge signal could be measured from Sentinel-2 L2A. "
        "**Nothing here validates nutrient status.** The decisive test (red-edge vs harvest records) waits for "
        "the team's decision on which KUD/PKS archive we can access.",
        "",
        f"- Data as of {run['as_of']}" + (" (as-of rerun from the cache; nothing fetched)" if run["as_of_run"] else "")
        + f"; months {run['period'][0]} to {run['period'][1]}.",
        f"- Plots: {run['plots']} from {run['plots_source']}.",
    ]
    if proxy:
        lines.append("- **Proxy only.** s2-feasibility/data/private/plots.geojson is missing (plot_id, year_planted or age_years, "
                     "source per feature), so Uji 1 ran on a 100 m buffer around the .env point and Uji 2–3 were "
                     "checked on synthetic tests only.")
    lines += [
        "",
        "## Catalogue and offset",
        f"- Earth Search items found: {cat['items_found']}; not COG (skipped): {cat['items_not_cog_skipped']}; "
        f"duplicate acquisitions dropped ({rules.DEDUP}, GUESS): {cat['duplicates_dropped']}; kept: "
        f"{cat['items_kept']} on {cat['tiles']} tile(s).",
        f"- Offset rule: {off['rule']}.",
        f"- Metadata says offset: {off['items_metadata_says_offset']} items.",
        f"- Dates that needed the offset removed: {_dates(off['dates_needing_offset'])}.",
        f"- Dates where the pixels contradicted the metadata (no offset in the pixels): "
        f"{_dates(off['dates_metadata_contradicted'])}.",
        f"- Dates left unresolved (too cloudy for pixel evidence; excluded from Uji 3): "
        f"{_dates(off['dates_unresolved'])}.",
        "",
    ]
    lines += _uji1(uji1) + [""] + _uji2(uji2) + [""] + _uji3(uji3) + [""]
    lines += [
        "## Thresholds used (all GUESS – PENDING TEAM DECISION)",
        _table(["Name", "Value"], [(k, v) for k, v in run["thresholds"].items()]),
        "",
        "## Proposals that need a team decision (not decided here)",
        f"1. CLEAR_SHARE_MIN, INNER_BUFFER_M, MIN_CLEAN_PIXELS, MIN_PIXELS_PER_STAT: confirm or change the starting "
        f"values ({rules.CLEAR_SHARE_MIN}, {rules.INNER_BUFFER_M} m, {rules.MIN_CLEAN_PIXELS}, "
        f"{rules.MIN_PIXELS_PER_STAT}).",
        "2. Age classes for oil palm (currently 0–3, 4–8, 9–15, 16+).",
        f"3. Offset: trust the metadata only when the tile's dark pixels agree (share under 1000 DN ≤ "
        f"{rules.OFFSET_DARK_SHARE_MAX}); or exclude every contradicted date instead.",
        "4. Offset-unresolved scenes: keep excluding them from Uji 3, or follow the metadata.",
        f"5. Duplicate acquisitions: keep {rules.DEDUP}, or keep the original processing only.",
        f"6. Month statistic: pool {rules.MONTH_POOLING}, or use the single clearest scene.",
        f"7. Month water_stress: {rules.MONTH_WATER_STRESS}, or the P30 at the clearest scene, or at month end.",
        "8. Should a clear scene require vegetation (SCL 4) rather than SCL 4 or 5? A plot interior that SCL calls "
        "'not vegetated' counts as clear today.",
    ]
    if proxy:
        lines.append("9. Plot boundaries: who supplies s2-feasibility/data/private/plots.geojson, and from which source.")
    text = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(text)
    return text


if __name__ == "__main__":
    build()
    print("REPORT.md rebuilt in s2-feasibility/outputs/")
