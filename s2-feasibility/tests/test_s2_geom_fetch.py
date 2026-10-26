"""Tests for s2_geom.py and s2_fetch.py on synthetic rasters and features: no network.

    cd s2-feasibility && python -m unittest discover -s tests -v

Skipped without the GIS libraries (numpy, shapely, pyproj, rasterio).
"""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from rasterio.windows import Window
    from shapely.geometry import Polygon, box, mapping

    import s2_fetch as fetch
    import s2_geom as geom
    import s2_rules as rules
except ImportError:
    geom = None

needs_gis = unittest.skipUnless(geom, "GIS libraries not installed (conda env create -f environment.yml)")
GRID = Affine(20, 0, 0, 0, -20, 1000) if geom else None  # 20 m pixels, top-left at (0, 1000)


@needs_gis
class FullFootprint(unittest.TestCase):
    def test_grid_aligned_rectangle(self):
        raw, clean = geom.plot_masks(box(0, 800, 200, 900), GRID, Window(0, 0, 15, 15), buffer_m=20)
        self.assertEqual((int(raw.sum()), int(clean.sum())), (50, 24))  # 10 × 5, then 8 × 3

    def test_half_pixels_never_count(self):
        """Edges in the middle of pixels: only whole pixels count, where all_touched would take the partial ones."""
        plot = box(10, 790, 210, 910)  # 200 × 120 m, shifted half a pixel
        mask = geom.full_footprint_mask(plot, GRID, (15, 15))
        touched = rasterize([plot], out_shape=(15, 15), transform=GRID, all_touched=True).astype(bool)
        self.assertEqual(int(mask.sum()), 9 * 5)
        self.assertEqual(int(touched.sum()), 11 * 7)
        self.assertTrue(np.all(touched[mask]))

    def test_inward_buffer_on_a_triangle(self):
        tri = Polygon([(0, 1000), (300, 1000), (0, 700)])
        raw, clean = geom.plot_masks(tri, GRID, Window(0, 0, 16, 16), buffer_m=20)
        self.assertGreater(int(raw.sum()), int(clean.sum()))
        # every clean pixel lies wholly inside the buffered triangle
        inner = tri.buffer(-20)
        rows, cols = np.nonzero(clean)
        for r, c in zip(rows, cols):
            self.assertTrue(inner.contains(box(c * 20, 1000 - (r + 1) * 20, (c + 1) * 20, 1000 - r * 20)))

    def test_narrow_strip_keeps_nothing(self):
        raw, clean = geom.plot_masks(box(0, 940, 400, 980), GRID, Window(0, 0, 25, 5), buffer_m=20)
        self.assertEqual((int(raw.sum()), int(clean.sum())), (40, 0))

    def test_window_snaps_and_pads(self):
        w = geom.plot_window(box(105, 795, 195, 905), GRID, pad_px=2)
        self.assertEqual((w.col_off, w.row_off, w.width, w.height), (3, 2, 9, 11))
        self.assertTrue(geom.window_inside(w, 100, 100))
        self.assertFalse(geom.window_inside(Window(-1, 0, 5, 5), 100, 100))


@needs_gis
class Plots(unittest.TestCase):
    def test_proxy_is_a_100_m_circle(self):
        plots = geom.load_plots(Path("/nonexistent/plots.geojson"), 0.9, 104.5)
        self.assertEqual(len(plots), 1)
        p = plots[0]
        self.assertTrue(p["proxy"])
        self.assertEqual(p["plot_id"], geom.PROXY_ID)
        utm = geom.project(p["geometry"], geom.WGS84, geom.utm_crs(104.5, 0.9))
        self.assertAlmostEqual(utm.area / 10_000, 3.14, places=2)

    def test_geojson(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plots.geojson"
            path.write_text(json.dumps({"type": "FeatureCollection", "features": [{
                "type": "Feature", "geometry": mapping(box(104.5, 0.9, 104.501, 0.901)),
                "properties": {"plot_id": 7, "year_planted": 2012, "source": "survey"}}]}))
            plots = geom.load_plots(path, 0.0, 0.0)
        self.assertEqual((plots[0]["plot_id"], plots[0]["year_planted"], plots[0]["proxy"]), ("7", 2012, False))

    def test_utm_zone(self):
        self.assertEqual(geom.utm_crs(104.5, 0.9), "EPSG:32648")
        self.assertEqual(geom.utm_crs(106.8, -6.2), "EPSG:32748")


@needs_gis
class Cache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.saved = (fetch.CACHE_DIR, fetch.AS_OF_DATE, fetch.TODAY)
        fetch.CACHE_DIR = Path(self.dir.name)

    def tearDown(self):
        fetch.CACHE_DIR, fetch.AS_OF_DATE, fetch.TODAY = self.saved
        self.dir.cleanup()

    def test_second_call_reuses(self):
        calls = []
        fetch.cached_json("stac", "k", lambda: calls.append(1) or {"n": 1})
        self.assertEqual(fetch.cached_json("stac", "k", lambda: calls.append(1) or {"n": 2}), {"n": 1})
        self.assertEqual(len(calls), 1)

    def test_arrays(self):
        fetch.cached_array("win", "k", lambda: np.arange(4, dtype="uint16"))
        out = fetch.cached_array("win", "k", lambda: self.fail("fetched again"))
        self.assertEqual((out.dtype, out.tolist()), (np.dtype("uint16"), [0, 1, 2, 3]))

    def test_as_of_refuses_to_fetch(self):
        fetch.AS_OF_DATE, fetch.TODAY = "2026-10-01", date(2026, 10, 1)
        with self.assertRaises(fetch.NotCached):
            fetch.cached_json("stac", "k", lambda: self.fail("fetched in an as-of run"))

    def test_as_of_ignores_later_saves(self):
        (fetch.CACHE_DIR / "stac_k_20261005.json").write_text('{"n": "later"}')
        (fetch.CACHE_DIR / "stac_k_20260930.json").write_text('{"n": "earlier"}')
        fetch.AS_OF_DATE, fetch.TODAY = "2026-10-01", date(2026, 10, 1)
        self.assertEqual(fetch.cached_json("stac", "k", lambda: self.fail("fetched")), {"n": "earlier"})


@needs_gis
class Items(unittest.TestCase):
    def feature(self, href_ext=".tif", scheme="https://"):
        assets = {name: {"href": f"{scheme}example/{name}{href_ext}"} for name in fetch.ASSETS.values()}
        assets["scl"] |= {"proj:transform": [20, 0, 0, 0, -20, 1000, 0, 0, 1], "proj:shape": [5490, 5490]}
        return {"id": "x", "assets": assets, "properties": {
            "datetime": "2022-02-01T03:25:00Z", "grid:code": "MGRS-00AAA", "s2:processing_baseline": "04.00",
            "earthsearch:boa_offset_applied": False, "updated": "2022-02-02", "proj:code": "EPSG:32648"}}

    def test_record(self):
        r = fetch.item_record(self.feature())
        self.assertEqual((r["date"], r["baseline"], r["boa_offset_applied"], r["cog"]),
                         (date(2022, 2, 1), "04.00", False, True))
        self.assertEqual((r["transform"].a, r["shape"], r["crs"]), (20, (5490, 5490), "EPSG:32648"))
        self.assertTrue(rules.offset_in_metadata(r["baseline"], r["boa_offset_applied"]))

    def test_jp2_requester_pays_is_not_cog(self):
        self.assertFalse(fetch.item_record(self.feature(".jp2", "s3://"))["cog"])

    def test_read_failure_hides_the_path(self):
        with self.assertRaises(fetch.ReadFailed) as ctx:
            saved, fetch.READ_ATTEMPTS = fetch.READ_ATTEMPTS, 1
            try:
                fetch._read("/nonexistent/secret-tile-path.tif")
            finally:
                fetch.READ_ATTEMPTS = saved
        self.assertNotIn("secret", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
