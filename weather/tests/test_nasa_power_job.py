"""Network-free tests for nasa_power_job.py: grid cells, the files it builds, and a full run with fake sources.

    cd weather && python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nasa_power_job as job  # noqa: E402


def days(start, values):
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


def climate_rain(first=1991, last=2020, mm_per_day=6.0):
    """Every day of the period at a constant rate: each month's normal = its length × mm_per_day."""
    start, end = date(first, 1, 1), date(last, 12, 31)
    return {start + timedelta(days=i): mm_per_day for i in range((end - start).days + 1)}


class GridCells(unittest.TestCase):
    def test_trial_plot_cell(self):
        self.assertEqual(job.grid_cell(0.92, 104.45), {"id": "n1.000_e104.3750", "lat": 1.0, "lon": 104.375})

    def test_southern_hemisphere_id(self):
        self.assertEqual(job.grid_cell(-0.4, 101.4)["id"], "s0.500_e101.2500")

    def test_nearby_plots_share_one_cell(self):
        cells = job.group_plots([{"plot_id": "a", "lat": 0.92, "lon": 104.45},
                                 {"plot_id": "b", "lat": 0.95, "lon": 104.40},
                                 {"plot_id": "c", "lat": 0.40, "lon": 104.45}])
        self.assertEqual(sorted(len(c["plots"]) for c in cells.values()), [1, 2])


class BuildFiles(unittest.TestCase):
    def test_climate_has_twelve_months_with_gates(self):
        climate = job.build_climate(climate_rain(mm_per_day=6.0), date(2026, 10, 11))
        self.assertEqual(climate["normals_mm"]["1"], 186.0)          # 31 × 6
        self.assertEqual(climate["stage_a"]["2"]["gate"], "open")     # 28 or 29 × 6 = 168–174
        self.assertEqual(climate["stage_a"]["1"]["in_band_share"], 1.0)
        self.assertEqual(len(climate["stage_a"]), 12)

    def test_climate_refuses_a_period_with_missing_months(self):
        with self.assertRaises(ValueError):
            job.build_climate(days(date(1991, 1, 1), [5.0] * 31), date(2026, 10, 11))

    def test_recent_drops_a_suspect_month_and_computes_p30(self):
        normals = {str(m): 200.0 for m in range(1, 13)}
        rain = days(date(2026, 8, 1), [40.0] * 31 + [2.5] * 30 + [None] * 3)  # Aug 1,240 mm: broken; Sep fine
        recent = job.build_recent(rain, normals)
        self.assertEqual(recent["suspect_months"], ["2026-08"])
        self.assertEqual(recent["checked_against"], "normals")
        self.assertNotIn("2026-08-15", recent["rain_mm"])
        self.assertEqual(recent["last_day"], "2026-09-30")
        self.assertEqual(recent["p30"]["p30_mm"], 75.0)
        self.assertEqual(recent["p30"]["p30_start_date"], "2026-09-01")

    def test_recent_keeps_a_wet_month_weatherkit_also_saw(self):
        normals = {str(m): 200.0 for m in range(1, 13)}
        rain = days(date(2026, 8, 1), [25.0] * 31)
        recent = job.build_recent(rain, normals, weatherkit_rain=days(date(2026, 8, 1), [10.0] * 31))
        self.assertEqual(recent["suspect_months"], [])
        self.assertEqual(recent["checked_against"], "normals and weatherkit")


class FullRun(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def fake_fetch(self, lat, lon, start, end, params, session):
        self.calls.append((lat, lon, start, end, params))
        if start.year == 1991:
            return {job.RAIN: climate_rain()}, {job.RAIN: "mm/day"}, ["MERRA2"]
        rain = {start + timedelta(days=i): 3.0 for i in range((end - start).days - 2)}  # 3 days behind
        sun = {d: 20.0 for d in rain}
        return {job.RAIN: rain, job.SUN: sun}, {job.RAIN: "mm/day", job.SUN: "MJ/m^2/day"}, ["MERRA2"]

    def test_run_writes_cell_files_and_index(self):
        plots = [{"plot_id": "a", "lat": 0.92, "lon": 104.45}, {"plot_id": "b", "lat": 0.95, "lon": 104.40}]
        with tempfile.TemporaryDirectory() as out:
            failed = job.run(plots, out, date(2026, 10, 11), session=None, fetch=self.fake_fetch)
            self.assertEqual(failed, 0)
            self.assertEqual(len(self.calls), 2)  # one climate + one recent request for the shared cell
            snapshot = json.loads((Path(out) / "cells/n1.000_e104.3750/latest.json").read_text())
            self.assertEqual(snapshot["cell"]["plots"], ["a", "b"])
            self.assertEqual(snapshot["recent_rain"]["p30"]["p30_mm"], 90.0)
            self.assertEqual(snapshot["recent_rain"]["rain_mm"]["2026-08-01"], 3.0)  # whole of last month
            self.assertEqual(snapshot["irradiance"]["unit"], "MJ/m^2/day")
            index = json.loads((Path(out) / "plots.json").read_text())
            self.assertEqual(index["plots"]["b"]["path"], "cells/n1.000_e104.3750/latest.json")

            # Second run the same day reuses the climate table: only the recent request is made
            job.run(plots, out, date(2026, 10, 11), session=None, fetch=self.fake_fetch)
            self.assertEqual(len(self.calls), 3)

    def test_a_failing_cell_is_counted_and_others_still_written(self):
        def flaky(lat, lon, start, end, params, session):
            if lat < 0.75:
                raise RuntimeError("NASA POWER down")
            return self.fake_fetch(lat, lon, start, end, params, session)

        plots = [{"plot_id": "ok", "lat": 0.92, "lon": 104.45}, {"plot_id": "bad", "lat": 0.40, "lon": 104.45}]
        with tempfile.TemporaryDirectory() as out:
            with self.assertLogs("nasa_power_job", level="ERROR"):
                failed = job.run(plots, out, date(2026, 10, 11), session=None, fetch=flaky)
            self.assertEqual(failed, 1)
            self.assertTrue((Path(out) / "cells/n1.000_e104.3750/latest.json").is_file())


if __name__ == "__main__":
    unittest.main()
