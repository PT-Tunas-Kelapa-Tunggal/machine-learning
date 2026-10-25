"""Google Earth GeoJSON export → plots.geojson for the Sentinel-2 feasibility tests.

    python s2-feasibility/tools/ge_to_plots.py [export.geojson] [plots.geojson]

Defaults: s2-feasibility/data/private/ge_export.geojson → s2-feasibility/data/private/plots.geojson (both gitignored).

Each placemark is one plot: its name is the plot_id, and its description holds "year_planted=YYYY", or
"age_uncertain" when the planting year isn't known. Output properties: plot_id, source = "google_earth_trace", and
year_planted or age_uncertain = true.

Only Polygons are read; z is dropped and holes are ignored (the outer ring is the plot). Anything else is skipped
with its name and the reason. Prints only plot_id, area in ha and year_planted: never a coordinate.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

PRIVATE = Path(__file__).resolve().parents[1] / "data" / "private"
SOURCE = "google_earth_trace"
SMALL_HA = 2.0  # print hint only ("terlalu kecil"); Uji 2 decides with MIN_CLEAN_PIXELS


class ExportError(ValueError):
    """A problem with the export, described by placemark name only."""


def _description(props):
    desc = props.get("description", "")
    return desc.get("value", "") if isinstance(desc, dict) else (desc or "")


def area_m2(ring):
    """Shoelace area of a closed [lon, lat] ring, scaled to metres at the ring's mean latitude."""
    lat0 = sum(y for _, y in ring) / len(ring)
    mx, my = 111320 * math.cos(math.radians(lat0)), 110574
    return abs(sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1] for i in range(len(ring) - 1))) / 2 * mx * my


def convert(src):
    """(plots FeatureCollection, summary rows {plot_id, area_ha, year_planted}, skipped [(name, reason)])."""
    out, rows, skipped, seen = {"type": "FeatureCollection", "features": []}, [], [], set()
    for f in src.get("features", []):
        p = f.get("properties") or {}
        name = (p.get("name") or "").strip()
        geometry = f.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            skipped.append((name or "(no name)", f"geometry is {geometry.get('type')}, not Polygon"))
            continue
        if not name:
            raise ExportError("a Polygon placemark has no name, so it has no plot_id")
        if name in seen:
            raise ExportError(f"plot_id {name!r} appears more than once")
        seen.add(name)

        ring = [pt[:2] for pt in geometry["coordinates"][0]]  # drop z; holes ignored
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            raise ExportError(f"plot_id {name!r} has fewer than 3 corners")

        desc = _description(p)
        year = re.search(r"year_planted\s*=\s*(\d{4})", desc)
        props = {"plot_id": name, "source": SOURCE}
        if year and "age_uncertain" not in desc:
            props["year_planted"] = int(year.group(1))
        else:
            props["age_uncertain"] = True
        out["features"].append({"type": "Feature", "properties": props,
                                "geometry": {"type": "Polygon", "coordinates": [ring]}})
        rows.append({"plot_id": name, "area_ha": area_m2(ring) / 10_000, "year_planted": props.get("year_planted")})
    return out, rows, skipped


def summary_line(row):
    year = row["year_planted"] if row["year_planted"] is not None else "?"
    return (f"{row['plot_id']}: {row['area_ha']:.2f} ha, year_planted={year}"
            + ("  <-- terlalu kecil" if row["area_ha"] < SMALL_HA else ""))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src_path = Path(argv[0]) if argv else PRIVATE / "ge_export.geojson"
    out_path = Path(argv[1]) if len(argv) > 1 else PRIVATE / "plots.geojson"
    try:
        plots, rows, skipped = convert(json.loads(src_path.read_text()))
    except ExportError as e:
        print(f"not converted: {e}", file=sys.stderr)
        return 1
    for row in rows:
        print(summary_line(row))
    for name, reason in skipped:
        print(f"skipped {name}: {reason}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plots, indent=2) + "\n")
    print(f"{len(rows)} plot(s) written to {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
