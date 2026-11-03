"""Tests for Iteration 2 features: user submission handling, NPK comparison, and ui_data."""
import json
import unittest
from pathlib import Path

from krops_vari.config import REPO, Settings
from krops_vari.main import parse_args
from krops_vari.pipeline import run_scan

SUBMISSION_FIXTURE = REPO / "segmentation" / "iteration2" / "user_submission_request.json"
PHOTO_FIXTURE = REPO / "segmentation" / "final_flow_ut1" / "dji_10.jpg"


class TestSubmissionIntegration(unittest.TestCase):
    def test_parse_args_submission_and_model(self):
        args = parse_args(["--photo", str(PHOTO_FIXTURE), "--submission", str(SUBMISSION_FIXTURE), "--model", "dummy.mlpackage"])
        self.assertEqual(args.photo, PHOTO_FIXTURE)
        self.assertEqual(args.submission, SUBMISSION_FIXTURE)
        self.assertEqual(args.model, Path("dummy.mlpackage"))

    def test_run_scan_with_submission(self):
        if not SUBMISSION_FIXTURE.is_file() or not PHOTO_FIXTURE.is_file():
            raise unittest.SkipTest("Fixtures missing")
        submission_data = json.loads(SUBMISSION_FIXTURE.read_text(encoding="utf-8"))
        result = run_scan(PHOTO_FIXTURE, Settings(), submission=submission_data)

        # Check top-level additions
        self.assertIn("combined_fertilizer_calculations", result)
        self.assertIn("ui_data", result)

        # Check combined_fertilizer_calculations structure
        comb = result["combined_fertilizer_calculations"]
        self.assertIn("field_wide_npk_baseline_comparison", comb)
        self.assertIn("zone_vari_boost_calculations", comb)
        self.assertIn("scope_note", comb)

        field_npk = comb["field_wide_npk_baseline_comparison"]
        self.assertIn("user_blend", field_npk)
        self.assertIn("nutrient_comparison", field_npk)
        self.assertIn("standalone_top_up_options", field_npk)
        self.assertIn("routing", field_npk)

        # Check ui_data structure
        ui = result["ui_data"]
        self.assertEqual(ui["schema_version"], "1.0")
        self.assertEqual(ui["submission_month"], 10)
        self.assertEqual(ui["farm_overview"]["name"], "Kebun Pak Sutrisno")
        self.assertEqual(ui["farm_overview"]["area_ha"], 1.0)
        self.assertIn("nutrient_cards", ui["configured_annual_nutrient_comparison"])
        self.assertIn("zone_recommendations", ui)

        # Check lowest min VARI flag present in zones
        zone0_vari = result["zones"][0]["vari"]
        self.assertIn("lowest_min_vari_zone", zone0_vari)
        self.assertIn("lowest_zone_min_vari_value", zone0_vari)
        self.assertIn("lowest_mean_vari_zone", zone0_vari)

        # Check base product columns in product mass
        pm0 = result["product_mass"][0]
        self.assertIn("base_product_kg_per_zone_per_year", pm0)
        self.assertIn("base_plus_supplement_product_kg_per_zone_per_year", pm0)

    def test_run_scan_without_submission_backward_compat(self):
        if not PHOTO_FIXTURE.is_file():
            raise unittest.SkipTest("Photo fixture missing")
        result = run_scan(PHOTO_FIXTURE, Settings())
        self.assertNotIn("combined_fertilizer_calculations", result)
        self.assertNotIn("ui_data", result)
        self.assertNotIn("lowest_min_vari_zone", result["zones"][0]["vari"])
        self.assertNotIn("base_product_kg_per_zone_per_year", result["product_mass"][0])


if __name__ == "__main__":
    unittest.main()

