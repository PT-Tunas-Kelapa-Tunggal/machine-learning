"""Tests for tools/ge_to_plots.py on a synthetic Google Earth export: no GIS libraries needed.

    cd s2-feasibility && python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
import ge_to_plots as ge  # noqa: E402
import s2_rules as rules  # noqa: E402

LON0, LAT0 = 104.123457, 0.987654  # made-up corner, never a real plot
D_LAT = 100 / 110574                # 100 m north
D_LON = 100 / (111320 * 0.99985)    # 100 m east at this latitude (cos 0.9877° ≈ 0.99985)


def square(lon, lat, closed=True, z=True):
    ring = [[lon, lat], [lon + D_LON, lat], [lon + D_LON, lat + D_LAT], [lon, lat + D_LAT]]
    ring = [pt + [0.0] if z else pt for pt in ring]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]] if closed else ring]}


def placemark(name, description, geometry):
    return {"type": "Feature", "properties": {"name": name, "description": description, "styleUrl": "#x"},
            "geometry": geometry}


EXPORT = {"type": "FeatureCollection", "features": [
    placemark("P01", {"@type": "html", "value": "<div>year_planted=2016</div>"}, square(LON0, LAT0)),
    placemark("P02", "year_planted = 2009; age_uncertain", square(LON0 + 0.01, LAT0, closed=False, z=False)),
    placemark("P03", "no year here", square(LON0 + 0.02, LAT0)),
    placemark("pin", "", {"type": "Point", "coordinates": [LON0, LAT0, 0.0]}),
]}


class Convert(unittest.TestCase):
    def setUp(self):
        self.plots, self.rows, self.skipped = ge.convert(EXPORT)
        self.props = {f["properties"]["plot_id"]: f["properties"] for f in self.plots["features"]}

    def test_properties(self):
        self.assertEqual(self.props["P01"], {"plot_id": "P01", "source": "google_earth_trace", "year_planted": 2016})
        self.assertEqual(self.props["P02"], {"plot_id": "P02", "source": "google_earth_trace", "age_uncertain": True})
        self.assertEqual(self.props["P03"], {"plot_id": "P03", "source": "google_earth_trace", "age_uncertain": True})

    def test_area_of_a_100_m_square(self):
        self.assertAlmostEqual(self.rows[0]["area_ha"], 1.0, places=2)

    def test_z_dropped_and_ring_closed(self):
        rings = {f["properties"]["plot_id"]: f["geometry"]["coordinates"][0] for f in self.plots["features"]}
        self.assertTrue(all(len(pt) == 2 for pt in rings["P01"]))
        self.assertEqual(rings["P02"][0], rings["P02"][-1])
        self.assertEqual(len(rings["P02"]), 5)

    def test_points_skipped_by_name(self):
        self.assertEqual(self.skipped, [("pin", "geometry is Point, not Polygon")])

    def test_duplicate_and_missing_names_rejected(self):
        dup = {"features": [placemark("P01", "", square(LON0, LAT0)), placemark("P01", "", square(LON0, LAT0))]}
        with self.assertRaisesRegex(ge.ExportError, "more than once"):
            ge.convert(dup)
        with self.assertRaisesRegex(ge.ExportError, "no name"):
            ge.convert({"features": [placemark(" ", "", square(LON0, LAT0))]})

    def test_loaded_by_the_runner(self):
        """plots.geojson as s2_geom.load_plots reads it: year_planted, or age_uncertain and no age."""
        try:
            import s2_geom as geom
        except ImportError:
            self.skipTest("GIS libraries not installed")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plots.geojson"
            path.write_text(json.dumps(self.plots))
            plots = {p["plot_id"]: p for p in geom.load_plots(path, 0.0, 0.0)}
        self.assertEqual((plots["P01"]["year_planted"], plots["P01"]["age_uncertain"]), (2016, False))
        self.assertEqual((plots["P02"]["year_planted"], plots["P02"]["age_uncertain"]), (None, True))
        self.assertIsNone(rules.plot_age(plots["P02"], 2024, 2026))


class Main(unittest.TestCase):
    def run_main(self, export):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "ge_export.geojson", Path(d) / "private" / "plots.geojson"
            src.write_text(json.dumps(export))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = ge.main([str(src), str(dst)])
            written = json.loads(dst.read_text()) if dst.is_file() else None
        return code, out.getvalue() + err.getvalue(), written

    def test_prints_only_id_area_and_year(self):
        code, printed, written = self.run_main(EXPORT)
        self.assertEqual(code, 0)
        self.assertEqual(printed.splitlines(), [
            "P01: 1.00 ha, year_planted=2016  <-- terlalu kecil",
            "P02: 1.00 ha, year_planted=?  <-- terlalu kecil",
            "P03: 1.00 ha, year_planted=?  <-- terlalu kecil",
            "skipped pin: geometry is Point, not Polygon",
            "3 plot(s) written to plots.geojson",
        ])
        self.assertEqual(len(written["features"]), 3)

    def test_nothing_printed_looks_like_a_coordinate(self):
        _, printed, _ = self.run_main(EXPORT)
        self.assertEqual(rules.find_coordinates(printed, known=[f"{LON0:.2f}", f"{LAT0:.3f}"]), [])
        for digits in ("104.12", "0.987", "0.98"):
            self.assertNotIn(digits, printed)

    def test_error_names_the_plot_only(self):
        dup = {"features": [placemark("P07", "", square(LON0, LAT0)), placemark("P07", "", square(LON0, LAT0))]}
        code, printed, written = self.run_main(dup)
        self.assertEqual((code, written), (1, None))
        self.assertIn("P07", printed)
        self.assertNotIn("104.", printed)


if __name__ == "__main__":
    unittest.main()
