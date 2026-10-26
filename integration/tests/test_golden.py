"""The package against the notebook: dji_10.jpg through run_scan gives the same zones, confidence, boost and product
mass as integration.ipynb (tests/fixtures/dji_10_notebook.json). Needs macOS (Core ML) and the photo and model."""
import json
import sys
import unittest
from pathlib import Path

from helpers import assert_close
from krops_vari.config import REPO, Settings

FIXTURE = Path(__file__).parent / "fixtures" / "dji_10_notebook.json"


@unittest.skipUnless(sys.platform == "darwin", "Core ML runs on macOS only")
class GoldenDji10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from krops_vari.pipeline import run_scan

        cls.want = json.loads(FIXTURE.read_text())
        photo = REPO / cls.want["photo"]
        if not photo.is_file() or not Path(Settings().model_path).exists():
            raise unittest.SkipTest("dji_10.jpg or the Core ML model is missing")
        cls.got = json.loads(json.dumps(run_scan(photo, Settings())))  # through JSON, as the API returns it

    def test_zones_match_the_notebook(self):
        assert_close(self, self.got["zones"], self.want["zones"], "zones")

    def test_product_mass_matches_the_notebook(self):
        assert_close(self, self.got["product_mass"], self.want["product_mass"], "product_mass")

    def test_summary(self):
        s = self.got["summary"]
        self.assertEqual(s["reference_zone_ids"], [1, 2, 3, 4])
        self.assertAlmostEqual(s["p25_zone_mean_vari"], 0.17927271500229836, places=12)
        self.assertEqual(s["boost_mode"], "zone_specific")
        self.assertEqual(self.got["image"]["trees"], 34)
        self.assertEqual(self.got["image"]["trees_dropped_partial"], 3)

    def test_no_coordinates_in_the_result(self):
        text = json.dumps(self.got)
        for key in ('"lat"', '"lon"', '"latitude"', '"longitude"'):
            self.assertNotIn(key, text)


if __name__ == "__main__":
    unittest.main()
