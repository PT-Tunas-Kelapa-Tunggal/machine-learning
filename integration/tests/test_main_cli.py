"""Regression tests for the VARI command line and its import behavior."""
import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from krops_vari import main as cli
from krops_vari.v2_engine import run_v2_scan


class MainCliTests(unittest.TestCase):
    def test_module_help_does_not_run_a_scan(self):
        result = subprocess.run(
            [sys.executable, "-m", "krops_vari.main", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check", result.stdout)
        self.assertIn("--image", result.stdout)

    def test_check_does_not_require_scan_arguments(self):
        with patch.object(cli, "check_engine", return_value=0) as check:
            self.assertEqual(cli.main(["--check"]), 0)
        check.assert_called_once_with()

    def test_scan_validates_input_before_loading_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_image = Path(directory) / "missing.jpg"
            with self.assertRaisesRegex(FileNotFoundError, "Image not found"):
                run_v2_scan(missing_image, {})

    def test_scan_arguments_are_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            cli.main([])
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
