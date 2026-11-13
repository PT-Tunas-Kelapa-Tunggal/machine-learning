"""Tests for Krops VARI Engine (Iteration 2).

Validates:
- Strict concrete model locking to krops_model_v0.2_yolo26l.mlpackage with zero fallback
- Pruning of legacy CLI flags (--photo, --out, --settings, etc.)
- Engine self-check (--check)
- Dynamic --output-dir and dual JSON generation (vari_result.json + oil_palm_zones_vari.json)
- Swift compatibility contract (image.trees, summary.reference_zone_mean_vari, zones)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from krops_vari.config import CONCRETE_MODEL_NAME, resolve_model_path
from krops_vari.main import check_engine, main, parse_args
from krops_vari.v2_engine import run_v2_scan

REPO = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = REPO / "segmentation" / "iteration2" / "webodm_new_2nd_flight.jpeg"
SAMPLE_SUBMISSION_FILE = REPO / "integration" / "v2" / "fertlizer_calculations" / "user_submission_request.json"


@unittest.skipUnless(sys.platform == "darwin", "Core ML runs on macOS only")
class V2EngineTests(unittest.TestCase):
    def test_concrete_model_name(self):
        self.assertEqual(CONCRETE_MODEL_NAME, "krops_model_v0.2_yolo26l.mlpackage")

    def test_self_check_succeeds(self):
        status = check_engine()
        self.assertEqual(status, 0)

    def test_missing_model_fail_fast(self):
        old_env = os.environ.get("KROPS_MODEL_PATH")
        try:
            # 1. Non-existent path
            os.environ["KROPS_MODEL_PATH"] = "/tmp/nonexistent/krops_model_v0.2_yolo26l.mlpackage"
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_model_path()
            self.assertIn("does not exist", str(ctx.exception))

            # 2. Legacy model attempt
            os.environ["KROPS_MODEL_PATH"] = "/tmp/yolo26m-krops-v0.1.mlpackage"
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_model_path()
            self.assertIn("Fallback to legacy or other models is strictly disabled", str(ctx.exception))
        finally:
            if old_env is not None:
                os.environ["KROPS_MODEL_PATH"] = old_env
            else:
                os.environ.pop("KROPS_MODEL_PATH", None)

    def test_legacy_flags_pruned(self):
        # --photo is rejected
        with self.assertRaises(SystemExit) as cm:
            parse_args(["--photo", "test.jpg"])
        self.assertEqual(cm.exception.code, 2)

        # --out is rejected
        with self.assertRaises(SystemExit) as cm:
            parse_args(["--out", "result.json"])
        self.assertEqual(cm.exception.code, 2)

        # --settings is rejected
        with self.assertRaises(SystemExit) as cm:
            parse_args(["--settings", "settings.json"])
        self.assertEqual(cm.exception.code, 2)

    def test_invalid_submission_json_string(self):
        status = main([
            "--image", str(SAMPLE_IMAGE),
            "--user-request-submission-json", "{broken_json",
        ])
        self.assertEqual(status, 1)

    def test_end_to_end_v2_scan_outputs_and_swift_contract(self):
        if not SAMPLE_IMAGE.exists() or not SAMPLE_SUBMISSION_FILE.exists():
            raise unittest.SkipTest("Sample test image or submission file missing")

        submission_text = SAMPLE_SUBMISSION_FILE.read_text()
        temp_dir = Path(tempfile.mkdtemp(prefix="krops_v2_test_"))
        try:
            result = run_v2_scan(
                image_path=SAMPLE_IMAGE,
                user_submission_data=submission_text,
                output_dir=temp_dir,
            )

            # Check return structure
            self.assertEqual(result["schema"], 2)
            self.assertEqual(result["package_version"], "0.2.0")
            self.assertEqual(result["image"]["trees"], 47)
            self.assertEqual(len(result["zones"]), 6)

            # Check that files were written directly to temp_dir
            vari_result_path = temp_dir / "vari_result.json"
            zones_vari_path = temp_dir / "oil_palm_zones_vari.json"
            user_summary_path = temp_dir / "user_fertilizer_summary.json"
            annotated_jpg = temp_dir / "oil_palm_zones_annotated.jpg"
            white_sep_jpg = temp_dir / "oil_palm_zones_white_separators.jpg"
            vari_plot_png = temp_dir / "oil_palm_vari_distribution.png"
            heatmaps_dir = temp_dir / "oil_palm_tree_vari_heatmaps"

            self.assertTrue(vari_result_path.is_file(), "vari_result.json must exist")
            self.assertTrue(zones_vari_path.is_file(), "oil_palm_zones_vari.json must exist")
            self.assertTrue(user_summary_path.is_file(), "user_fertilizer_summary.json must exist")
            self.assertTrue(annotated_jpg.is_file(), "oil_palm_zones_annotated.jpg must exist")
            self.assertTrue(white_sep_jpg.is_file(), "oil_palm_zones_white_separators.jpg must exist")
            self.assertTrue(vari_plot_png.is_file(), "oil_palm_vari_distribution.png must exist")
            self.assertTrue(heatmaps_dir.is_dir(), "heatmaps directory must exist")
            self.assertEqual(len(list(heatmaps_dir.glob("*.png"))), 47)

            # Swift compatibility validation
            with vari_result_path.open("r", encoding="utf-8") as f:
                parsed_json = json.load(f)

            image_info = parsed_json.get("image", {})
            tree_count = image_info.get("trees", 0)
            zones = parsed_json.get("zones", [])
            summary = parsed_json.get("summary", {})
            mean_vari = summary.get("reference_zone_mean_vari") or summary.get("p25_zone_mean_vari")

            self.assertEqual(tree_count, 47)
            self.assertEqual(len(zones), 6)
            self.assertIsNotNone(mean_vari)
            self.assertGreater(mean_vari, 0.0)

            # Verify user_fertilizer_summary.json
            with user_summary_path.open("r", encoding="utf-8") as f:
                summary_json = json.load(f)
            self.assertIn("fertilizers", summary_json)
            self.assertIn("fertilizerFreq", summary_json)
            self.assertIsInstance(summary_json["fertilizers"], list)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
