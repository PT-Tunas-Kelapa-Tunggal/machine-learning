"""Network-free tests for nasa_power_job.py: grid cells, the files it builds, and a full run with fake sources.

    cd weather && python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

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

    def test_weatherkit_area_is_about_11_km(self):
        # 104.45 sits on a box edge; it falls in the box centred on 104.5 (the same rounding in Swift)
        self.assertEqual(job.weatherkit_area(0.92, 104.45)["id"], "n0.900_e104.5000")
        self.assertEqual(job.weatherkit_area(0.93, 104.47)["id"], "n0.900_e104.5000")   # ~2 km away: same area
        self.assertNotEqual(job.weatherkit_area(1.05, 104.45)["id"], "n0.900_e104.5000")  # ~15 km away: not

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
        self.assertEqual(recent["nasa_p30"]["p30_mm"], 75.0)
        self.assertEqual(recent["nasa_p30"]["p30_start_date"], "2026-09-01")

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
        clear = {d: 24.0 for d in rain if d <= end - timedelta(days=73)}  # CLRSKY runs ~73 days behind
        temps = {p: {d: t for d in rain} for p, t in zip(job.TEMPS, (27.0, 31.0, 24.0))}
        return ({job.RAIN: rain, job.SUN: sun, job.SUN_CLEAR: clear} | temps,
                {job.RAIN: "mm/day", job.SUN: "MJ/m^2/day", job.SUN_CLEAR: "MJ/m^2/day"} | {p: "C" for p in job.TEMPS},
                ["MERRA2"])

    def test_run_writes_cell_files_and_index(self):
        plots = [{"plot_id": "a", "lat": 0.92, "lon": 104.45}, {"plot_id": "b", "lat": 0.95, "lon": 104.40}]
        with tempfile.TemporaryDirectory() as out:
            failed = job.run(plots, out, date(2026, 10, 11), session=None, fetch=self.fake_fetch)
            self.assertEqual(failed, 0)
            self.assertEqual(len(self.calls), 2)  # one climate + one recent request for the shared cell
            snapshot = json.loads((Path(out) / "cells/n1.000_e104.3750/latest.json").read_text())
            self.assertEqual(snapshot["cell"]["plots"], ["a", "b"])
            self.assertEqual(snapshot["schema"], 2)
            self.assertEqual(snapshot["recent_rain"]["nasa_p30"]["p30_mm"], 90.0)
            # No WeatherKit configured: every plot's P30 falls back to checked NASA POWER
            self.assertEqual(snapshot["l5"]["a"]["p30_source"], "nasa-power")
            self.assertEqual(snapshot["l5"]["a"]["p30_quality"], "fallback")
            self.assertEqual(snapshot["l5"]["b"]["p30_mm"], 90.0)
            self.assertEqual(snapshot["recent_rain"]["rain_mm"]["2026-08-01"], 3.0)  # whole of last month
            self.assertEqual(snapshot["irradiance"]["unit"], "MJ/m^2/day")
            index = json.loads((Path(out) / "plots.json").read_text())
            self.assertEqual(index["plots"]["b"]["path"], "cells/n1.000_e104.3750/latest.json")
            self.assertEqual(index["plots"]["a"]["weatherkit_area"], "n0.900_e104.5000")

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


class FakeWeatherKit:
    """Returns the same rate every day; counts history requests per area."""

    def __init__(self, mm_per_day, last_day_offset=1):
        self.mm, self.offset, self.requests = mm_per_day, last_day_offset, []

    def history(self, lat, lon, start, end):
        self.requests.append((lat, lon))
        stop = end - timedelta(days=self.offset - 1)  # offset > 1 = data stops early (stale)
        return {start + timedelta(days=i): self.mm for i in range((stop - start).days + 1)}

    def daily(self, lat, lon, start=None, end=None):
        return {"forecastDaily": {"days": []}}


class WeatherKitFirst(unittest.TestCase):
    TODAY = date(2026, 10, 11)

    def nasa(self, lat, lon, start, end, params, session):
        return FullRun.fake_fetch(self, lat, lon, start, end, params, session)

    def setUp(self):
        self.calls = []

    def run_job(self, plots, weatherkit):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        failed = job.run(plots, out.name, self.TODAY, session=None, weatherkit=weatherkit, fetch=self.nasa)
        self.assertEqual(failed, 0)
        cell = job.grid_cell(plots[0]["lat"], plots[0]["lon"])["id"]
        return json.loads((Path(out.name) / "cells" / cell / "latest.json").read_text())

    def test_p30_comes_from_weatherkit_when_it_is_recent(self):
        wk = FakeWeatherKit(mm_per_day=1.5)
        snap = self.run_job([{"plot_id": "a", "lat": 0.92, "lon": 104.45}], wk)
        l5 = snap["l5"]["a"]
        self.assertEqual((l5["p30_source"], l5["p30_quality"], l5["p30_mm"]), ("weatherkit-history", "ok", 45.0))
        self.assertTrue(l5["water_stress"])
        self.assertEqual(l5["p30_end_date"], "2026-10-10")  # yesterday: today's value is still partly forecast
        self.assertEqual(l5["p30_missing_days"], {"weatherkit-history": 0, "nasa-power": 0})
        self.assertIn("n0.900_e104.5000", snap["weatherkit_rain"])

    def test_a_weatherkit_gap_falls_back_with_the_count(self):
        class GappyWeatherKit(FakeWeatherKit):
            def history(self, lat, lon, start, end):
                rain = super().history(lat, lon, start, end)
                rain.pop(date(2026, 9, 25), None)  # one day WeatherKit never returned, like 1 Jun 2023
                return rain

        l5 = self.run_job([{"plot_id": "a", "lat": 0.92, "lon": 104.45}], GappyWeatherKit(mm_per_day=2.05))["l5"]["a"]
        # 29 × 2.05 = 59.45 mm: summing the gap as 0 would have raised a false water_stress
        self.assertEqual((l5["p30_source"], l5["p30_quality"], l5["p30_mm"], l5["water_stress"]),
                         ("nasa-power", "fallback", 90.0, False))
        self.assertEqual(l5["p30_missing_days"], {"weatherkit-history": 1, "nasa-power": 0})

    def test_plots_in_one_area_share_one_weatherkit_fetch(self):
        wk = FakeWeatherKit(mm_per_day=3.0)
        snap = self.run_job([{"plot_id": "a", "lat": 0.92, "lon": 104.45},
                             {"plot_id": "b", "lat": 0.93, "lon": 104.47},
                             {"plot_id": "c", "lat": 1.05, "lon": 104.45}], wk)
        p30_requests = [r for r in wk.requests if r != (1.0, 104.375)]  # the cell-centre call is the monthly check
        self.assertEqual(len(p30_requests), 2)
        self.assertEqual(snap["l5"]["a"]["weatherkit_area"], snap["l5"]["b"]["weatherkit_area"])

    def test_stale_weatherkit_falls_back_to_nasa_power(self):
        wk = FakeWeatherKit(mm_per_day=3.0, last_day_offset=10)  # WeatherKit data stops 10 days ago
        l5 = self.run_job([{"plot_id": "a", "lat": 0.92, "lon": 104.45}], wk)["l5"]["a"]
        self.assertEqual((l5["p30_source"], l5["p30_quality"]), ("nasa-power", "fallback"))

    def test_weatherkit_outage_degrades_instead_of_failing_the_cell(self):
        class DownWeatherKit(FakeWeatherKit):
            def history(self, lat, lon, start, end):
                raise ConnectionError("weatherkit.apple.com unreachable")

        with self.assertLogs("nasa_power_job", level="WARNING"):
            snap = self.run_job([{"plot_id": "a", "lat": 0.92, "lon": 104.45}], DownWeatherKit(1.0))
        self.assertEqual(snap["recent_rain"]["checked_against"], "normals")
        self.assertEqual((snap["l5"]["a"]["p30_source"], snap["l5"]["a"]["p30_quality"]), ("nasa-power", "fallback"))

    def test_no_source_gives_an_unknown_flag_not_false(self):
        normals = {str(m): 10.0 for m in range(1, 13)}
        recent = job.build_recent(days(date(2026, 9, 1), [40.0] * 40), normals)  # NASA months all suspect
        l5 = job.build_l5([{"plot_id": "a", "area": "x"}], {"x": None}, recent, self.TODAY)["a"]
        self.assertEqual((l5["p30_quality"], l5["p30_mm"], l5["water_stress"]), ("none", None, None))


class ScriptedWeatherKit(job.WeatherKit):
    """The real history() over scripted daily() replies: no key, no network."""

    def __init__(self, rain, drop=(), extra=()):
        self.tz, self.calls = ZoneInfo("Asia/Jakarta"), 0
        self.rain, self.drop, self.extra = rain, set(drop), list(extra)

    def _day(self, d, mm):
        start = datetime.combine(d, time(0), self.tz).astimezone(timezone.utc)  # local midnight, as Apple sends it
        return {"forecastStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "precipitationAmount": mm}

    def daily(self, lat, lon, start=None, end=None):
        asked = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        days = [self._day(d, self.rain[d]) for d in asked if d not in self.drop]
        days += [self._day(d, mm) for d, mm in self.extra if start <= d <= end]
        return {"forecastDaily": {"days": days}}


class WeatherKitHistory(unittest.TestCase):
    START, END = date(2023, 5, 23), date(2023, 6, 11)  # two 10-day requests
    RAIN = {date(2023, 5, 23) + timedelta(days=i): 5.0 for i in range(20)}

    def test_a_short_reply_is_logged_and_the_day_stays_missing(self):
        wk = ScriptedWeatherKit(self.RAIN, drop={date(2023, 6, 1)})  # the first request's last day, as Apple did
        with self.assertLogs("nasa_power_job", level="WARNING") as logs:
            got = wk.history(0.9, 104.5, self.START, self.END)
        self.assertNotIn(date(2023, 6, 1), got)
        self.assertEqual(len(got), 19)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("9 of 10 days", logs.output[0])
        self.assertIn("missing 2023-06-01", logs.output[0])

    def test_a_date_returned_twice_counts_once_or_as_missing(self):
        wk = ScriptedWeatherKit(self.RAIN, extra=[(date(2023, 5, 25), 5.0), (date(2023, 5, 26), 9.0)])
        got = wk.history(0.9, 104.5, self.START, self.END)
        self.assertEqual(got[date(2023, 5, 25)], 5.0)  # the same value twice: kept once
        self.assertIsNone(got[date(2023, 5, 26)])      # two different values: missing, not either one


class TrialCellRun(unittest.TestCase):
    """Helpers for one full run of the trial plot's cell with fake NASA POWER (no tests of its own).

    run_job returns latest.json and keeps the backend-only qa_log.json in self.qa_log."""

    TODAY = date(2026, 10, 11)
    PLOTS = [{"plot_id": "a", "lat": 0.92, "lon": 104.45}]

    def setUp(self):
        self.calls = []

    def run_job(self, fetch):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        failed = job.run(self.PLOTS, out.name, self.TODAY, session=None, fetch=fetch)
        self.assertEqual(failed, 0)
        cell = Path(out.name) / "cells/n1.000_e104.3750"
        self.qa_log = json.loads((cell / "qa_log.json").read_text())
        return json.loads((cell / "latest.json").read_text())

    def fetch(self, lat, lon, start, end, params, session):
        return FullRun.fake_fetch(self, lat, lon, start, end, params, session)


class LeanLatestJson(TrialCellRun):
    """The phone downloads latest.json on a weak signal: it keeps exactly the schema-2 keys."""

    SCHEMA_2_KEYS = ["schema", "generated_at", "cell", "climate", "l5", "weatherkit_rain", "recent_rain", "irradiance",
                     "nasa_power_sources"]

    def test_latest_json_keeps_the_schema_2_keys_and_the_qa_log_holds_the_rest(self):
        snap = self.run_job(self.fetch)
        self.assertEqual((snap["schema"], list(snap)), (2, self.SCHEMA_2_KEYS))
        self.assertEqual(list(self.qa_log), ["schema", "generated_at", "cell", "irradiance_check", "temperature"])
        self.assertEqual(self.qa_log["generated_at"], snap["generated_at"])
        self.assertEqual(self.qa_log["cell"]["id"], snap["cell"]["id"])
        self.assertNotIn("plots", self.qa_log["cell"])

    def test_the_index_points_only_at_latest_json(self):
        with tempfile.TemporaryDirectory() as out:
            job.run(self.PLOTS, out, self.TODAY, session=None, fetch=self.fetch)
            entry = json.loads((Path(out) / "plots.json").read_text())["plots"]["a"]
        self.assertEqual(sorted(entry), ["cell", "path", "weatherkit_area"])
        self.assertTrue(entry["path"].endswith("latest.json"))


class IrradianceCheckInTheJob(TrialCellRun):
    """The log-only irradiance check reaches further back in the same request, without touching L5."""

    def test_one_request_reaches_back_but_rain_and_irradiance_start_where_they_did(self):
        snap = self.run_job(self.fetch)
        recent_calls = [c for c in self.calls if c[2].year != 1991]
        self.assertEqual(len(recent_calls), 1)
        self.assertEqual(recent_calls[0][2], self.TODAY - timedelta(days=job.QA_LOOKBACK_DAYS))
        self.assertIn(job.SUN_CLEAR, recent_calls[0][4])
        self.assertEqual(min(snap["recent_rain"]["rain_mm"]), "2026-08-01")  # 1st of the month holding today − 45
        self.assertEqual(min(snap["irradiance"]["days"]), "2026-08-01")
        check = self.qa_log["irradiance_check"]
        self.assertEqual((check["first_day"], check["status"], check["days_flagged"]), ("2026-06-13", "log-only", 0))
        self.assertGreater(check["days_with_clrsky"], 0)

    def test_rain_before_the_recent_window_cannot_reach_l5(self):
        def wet_past(lat, lon, start, end, params, session):
            values, units, sources = self.fetch(lat, lon, start, end, params, session)
            if start.year != 1991:
                values[job.RAIN] = {d: (999.0 if d < date(2026, 8, 1) else v) for d, v in values[job.RAIN].items()}
            return values, units, sources

        snap = self.run_job(wet_past)
        self.assertEqual(snap["recent_rain"]["suspect_months"], [])
        self.assertEqual(snap["recent_rain"]["nasa_p30"]["p30_mm"], 90.0)
        self.assertEqual(snap["l5"]["a"]["p30_mm"], 90.0)

    def test_a_broken_irradiance_day_is_flagged_not_fatal(self):
        def broken(lat, lon, start, end, params, session):
            values, units, sources = self.fetch(lat, lon, start, end, params, session)
            if start.year != 1991:
                values[job.SUN][date(2026, 7, 1)] = 30.0  # over that day's CLRSKY of 24
            return values, units, sources

        self.run_job(broken)
        flagged = self.qa_log["irradiance_check"]["flagged"]
        self.assertEqual(list(flagged), ["2026-07-01"])
        self.assertEqual(flagged["2026-07-01"]["codes"], ["allsky_gt_clrsky"])

    def test_a_failing_check_is_written_as_null(self):
        with mock.patch.object(job.irradiance, "check_irradiance", side_effect=RuntimeError("bad maths")):
            with self.assertLogs("nasa_power_job", level="WARNING"):
                snap = self.run_job(self.fetch)
        self.assertIsNone(self.qa_log["irradiance_check"])
        self.assertEqual(snap["l5"]["a"]["p30_mm"], 90.0)



class TemperatureStoredOnly(TrialCellRun):
    """T2M, T2M_MAX and T2M_MIN ride in the same request and are stored over the irradiance window; no rule reads them."""

    def test_temperature_is_stored_over_the_irradiance_window(self):
        snap = self.run_job(self.fetch)
        recent_calls = [c for c in self.calls if c[2].year != 1991]
        self.assertEqual(len(recent_calls), 1)
        self.assertTrue(set(job.TEMPS) <= set(recent_calls[0][4]))
        temperature = self.qa_log["temperature"]
        self.assertEqual(temperature["units"], {"T2M": "C", "T2M_MAX": "C", "T2M_MIN": "C"})
        self.assertEqual(sorted(temperature["days"]), sorted(snap["irradiance"]["days"]))
        self.assertEqual(temperature["days"]["2026-08-01"], {"T2M": 27.0, "T2M_MAX": 31.0, "T2M_MIN": 24.0})

    def test_a_missing_value_is_null_and_a_missing_parameter_is_no_failure(self):
        def patchy(lat, lon, start, end, params, session):
            values, units, sources = self.fetch(lat, lon, start, end, params, session)
            if start.year != 1991:
                values["T2M_MAX"][date(2026, 8, 2)] = None
                del values["T2M_MIN"]  # NASA POWER left a parameter out
            return values, units, sources

        self.run_job(patchy)
        days = self.qa_log["temperature"]["days"]
        self.assertEqual(days["2026-08-02"], {"T2M": 27.0, "T2M_MAX": None, "T2M_MIN": None})

    def test_l5_and_l8_inputs_ignore_temperature(self):
        def hot(lat, lon, start, end, params, session):
            values, units, sources = self.fetch(lat, lon, start, end, params, session)
            for p in job.TEMPS:
                values[p] = {d: 45.0 for d in values.get(p, {})}
            return values, units, sources

        snap = self.run_job(hot)
        self.assertEqual(snap["l5"]["a"]["p30_mm"], 90.0)
        self.assertEqual(snap["climate"]["stage_a"]["10"]["gate"], "open")


if __name__ == "__main__":
    unittest.main()
