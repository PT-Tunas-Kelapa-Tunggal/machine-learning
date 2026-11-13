"""CLI and packaging tests for krops_vari.timing (fertilization_timing integration)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import krops_vari
from krops_vari import timing

REPO = Path(__file__).resolve().parents[2]
TIMING_DIR = REPO / "integration" / "v2" / "fertilization_timing"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if "VariEngine" not in krops_vari.__file__ and "PYTHONPATH" not in env:
        env["PYTHONPATH"] = str(REPO / "integration")
    return env


class TimingCliTests(unittest.TestCase):
    def test_help_surfaces_all_timing_flags(self):
        proc = subprocess.run(
            [sys.executable, "-m", "krops_vari.timing", "--help"],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ("--request", "--today", "--year", "--cache-dir", "--no-daily", "--out"):
            self.assertIn(flag, proc.stdout)

    def test_plain_request_via_cli_matches_example_output(self):
        with tempfile.TemporaryDirectory(prefix="krops_timing_plain_") as tmp:
            out_path = Path(tmp) / "schedule.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "krops_vari.timing",
                    "--request",
                    str(TIMING_DIR / "example_request.json"),
                    "--today",
                    "2026-11-08",
                    "--no-daily",
                    "--out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                env=_subprocess_env(),
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got = json.loads(out_path.read_text(encoding="utf-8"))
            want = json.loads((TIMING_DIR / "example_output.json").read_text(encoding="utf-8"))
            self.assertEqual(got, want)

    def test_in_progress_request_via_cli_matches_example_output(self):
        with tempfile.TemporaryDirectory(prefix="krops_timing_prog_") as tmp:
            out_path = Path(tmp) / "schedule_in_progress.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "krops_vari.timing",
                    "--request",
                    str(TIMING_DIR / "example_request_in_progress.json"),
                    "--today",
                    "2027-01-19",
                    "--no-daily",
                    "--out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                env=_subprocess_env(),
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got = json.loads(out_path.read_text(encoding="utf-8"))
            want = json.loads((TIMING_DIR / "example_output_in_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(got, want)

    def test_stdout_output_year_and_custom_cache_dir(self):
        with tempfile.TemporaryDirectory(prefix="krops_timing_cache_") as tmp:
            cache_dir = Path(tmp) / "weather_cache"
            cell_src = TIMING_DIR / "data" / "weather_cache" / "cells" / "n1.000_e104.3750"
            cell_dst = cache_dir / "cells" / "n1.000_e104.3750"
            cell_dst.mkdir(parents=True)
            shutil.copy2(cell_src / "climate.json", cell_dst / "climate.json")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "krops_vari.timing",
                    "--request",
                    str(TIMING_DIR / "example_request.json"),
                    "--today",
                    "2026-11-08",
                    "--year",
                    "2028",
                    "--cache-dir",
                    str(cache_dir),
                    "--no-daily",
                ],
                capture_output=True,
                text=True,
                env=_subprocess_env(),
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got = json.loads(proc.stdout)
            self.assertEqual(got["plan_year"], 2028)
            self.assertEqual(got["today"], "2026-11-08")

    def test_argv_forwarded_untouched_without_injected_defaults(self):
        resolved = timing.resolve_timing_dir()
        self.assertTrue((resolved / "fertilization_timing.py").is_file())

        timing_str = str(resolved)
        if timing_str in sys.path:
            sys.path.remove(timing_str)
        sys.path.insert(0, timing_str)
        import fertilization_timing

        recorded: list[list[str] | None] = []
        original_main = fertilization_timing.main
        try:
            fertilization_timing.main = lambda argv=None: recorded.append(argv)
            sample_args = ["--request", "req.json", "--no-daily"]
            self.assertEqual(timing.main(sample_args), 0)
            self.assertEqual(recorded, [sample_args])
        finally:
            fertilization_timing.main = original_main

    def test_imports_verbatim_timing_dependencies_before_weather_dir(self):
        env = _subprocess_env()
        if "VariEngine" not in krops_vari.__file__:
            weather_dir = str(REPO / "weather")
            env["PYTHONPATH"] = f"{weather_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        code = (
            "import sys; "
            "from krops_vari import timing; "
            f"timing.main(['--request', {str(TIMING_DIR / 'example_request.json')!r}, '--today', '2026-11-08', '--no-daily', '--out', sys.argv[1]]); "
            "import krops_rules, weather_sources; "
            "print(krops_rules.__file__); print(weather_sources.__file__)"
        )
        with tempfile.TemporaryDirectory(prefix="krops_timing_imp_") as tmp:
            out_path = Path(tmp) / "out.json"
            proc = subprocess.run(
                [sys.executable, "-c", code, str(out_path)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(len(lines), 2)
        resolved = timing.resolve_timing_dir().resolve()
        self.assertEqual(Path(lines[0]).resolve().parent, resolved)
        self.assertEqual(Path(lines[1]).resolve().parent, resolved)

    def test_missing_coordinates_exits_nonzero(self):
        with tempfile.TemporaryDirectory(prefix="krops_timing_null_") as tmp:
            req_path = Path(tmp) / "null_coords.json"
            req_path.write_text(
                json.dumps({"latitude": None, "longitude": None, "fertilizerFreq": 2, "fertilizers": []}),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "krops_vari.timing",
                    "--request",
                    str(req_path),
                    "--no-daily",
                ],
                capture_output=True,
                text=True,
                env=_subprocess_env(),
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("latitude and longitude are required", proc.stderr)


if __name__ == "__main__":
    unittest.main()
