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

SECRET = "made-up-secret-for-tests"  # names schema-3 folders in these runs


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

    def test_a_broken_month_is_dropped_in_its_first_days(self):
        normals = {str(m): 186.0 for m in range(1, 13)}
        rain = days(date(2026, 8, 1), [2.0] * 31 + [80.0] * 5)  # August fine; 1-5 Sep broken, 400 mm < 3 × 186
        recent = job.build_recent(rain, normals)
        self.assertEqual((recent["suspect_months"], recent["unchecked_months"]), (["2026-09"], []))
        self.assertEqual(recent["nasa_p30"]["p30_end_date"], "2026-08-31")  # the fallback window stays in August

    def test_a_month_with_too_few_days_is_unchecked(self):
        normals = {str(m): 186.0 for m in range(1, 13)}
        recent = job.build_recent(days(date(2026, 8, 1), [2.0] * 34), normals)  # 1-3 Sep only
        self.assertEqual((recent["suspect_months"], recent["unchecked_months"]), ([], ["2026-09"]))
        self.assertNotIn("2026-09-01", recent["rain_mm"])

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
            failed = job.run(plots, out, date(2026, 10, 11), session=None, plot_key_secret=SECRET, fetch=self.fake_fetch)
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
            job.run(plots, out, date(2026, 10, 11), session=None, plot_key_secret=SECRET, fetch=self.fake_fetch)
            self.assertEqual(len(self.calls), 3)

    def test_a_failing_cell_is_counted_and_others_still_written(self):
        def flaky(lat, lon, start, end, params, session):
            if lat < 0.75:
                raise RuntimeError("NASA POWER down")
            return self.fake_fetch(lat, lon, start, end, params, session)

        plots = [{"plot_id": "ok", "lat": 0.92, "lon": 104.45}, {"plot_id": "bad", "lat": 0.40, "lon": 104.45}]
        with tempfile.TemporaryDirectory() as out:
            with self.assertLogs("nasa_power_job", level="ERROR"):
                failed = job.run(plots, out, date(2026, 10, 11), session=None, plot_key_secret=SECRET, fetch=flaky)
            self.assertEqual(failed, 1)
            self.assertTrue((Path(out) / "cells/n1.000_e104.3750/latest.json").is_file())


def power_session(units, day, value):
    """A fake requests session: one day of NASA POWER for each {param: unit}, all at the same value."""
    payload = {"header": {"fill_value": -999.0, "sources": ["MERRA2"]},
               "parameters": {p: {"units": u} for p, u in units.items()},
               "properties": {"parameter": {p: {f"{day:%Y%m%d}": value} for p in units}}}
    return mock.Mock(get=mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: payload)))


class FetchNasaPowerUnits(unittest.TestCase):
    """The rain unit is checked only when rain was asked for."""

    DAY = date(2023, 6, 1)

    def test_sunlight_alone_needs_no_rain_unit(self):  # the VARI confidence's cloud fallback asks for this
        session = power_session({job.SUN: "MJ/m^2/day"}, self.DAY, 18.5)
        values, units, _ = job.fetch_nasa_power(1.0, 104.375, self.DAY, self.DAY, (job.SUN,), session)
        self.assertEqual(values[job.SUN][self.DAY], 18.5)
        self.assertEqual(units, {job.SUN: "MJ/m^2/day"})

    def test_a_wrong_rain_unit_still_fails(self):
        session = power_session({job.RAIN: "mm/hr"}, self.DAY, 3.0)
        with self.assertRaisesRegex(ValueError, "unexpected rain unit"):
            job.fetch_nasa_power(1.0, 104.375, self.DAY, self.DAY, (job.RAIN,), session)


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
        failed = job.run(plots, out.name, self.TODAY, session=None, plot_key_secret=SECRET, weatherkit=weatherkit, fetch=self.nasa)
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


FIXTURE = json.loads((Path(__file__).with_name("fixtures") / "weatherkit_2023-06-01_gap.json").read_text())


class RecordedWeatherKit(job.WeatherKit):
    """The real history() over WeatherKit replies recorded for 23 May-11 Jun 2023; any other request fails."""

    def __init__(self):
        self.tz, self.asked = ZoneInfo("Asia/Jakarta"), []
        self.replies = {(r["dailyStart"], r["dailyEnd"]): r["days"] for r in FIXTURE["replies"]}

    def daily(self, lat, lon, start=None, end=None):
        key = (start.isoformat(), end.isoformat())
        self.asked.append(key)
        if key not in self.replies:
            raise AssertionError(f"no recorded reply for {key}")
        return {"forecastDaily": {"days": self.replies[key]}}


class WeatherKitRetry(unittest.TestCase):
    """A short reply is re-requested once per gap, with the gap in the middle of the range (recorded replies)."""

    def test_the_recorded_gap_is_recovered_by_one_centred_retry(self):
        wk = RecordedWeatherKit()
        with self.assertLogs("nasa_power_job", level="WARNING") as logs:
            got = wk.history(0.0, 0.0, date(2023, 5, 23), date(2023, 6, 1))
        self.assertEqual(wk.asked, [("2023-05-23", "2023-06-01"), ("2023-05-28", "2023-06-06")])
        self.assertEqual((wk.retries, len(got), got[date(2023, 6, 1)]), (1, 10, 30.98))
        self.assertIn("missing 2023-06-01", logs.output[0])
        self.assertIn("recovered 1 of 1 day(s) (2023-06-01)", logs.output[1])

    def test_days_the_retry_repeats_count_once(self):
        wk = RecordedWeatherKit()
        with self.assertLogs("nasa_power_job", level="WARNING"):
            got = wk.history(0.0, 0.0, date(2023, 5, 23), date(2023, 6, 11))
        self.assertEqual(wk.asked, [("2023-05-23", "2023-06-01"), ("2023-05-28", "2023-06-06"), ("2023-06-02", "2023-06-11")])
        self.assertEqual(len(got), 20)
        self.assertTrue(all(v is not None for v in got.values()))  # 28 May-6 Jun came twice and agreed

    def test_the_quota_estimate_counts_cells_and_retries(self):
        self.assertEqual(job.weatherkit_calls_per_month(1, 0, days=1), (4, 4))  # 37 days of P30 history, 10 a request
        self.assertEqual(job.weatherkit_calls_per_month(0, 1, days=1), (5, 8))  # the per-cell check: 43-73 days
        low, high = job.weatherkit_calls_per_month(1000, 1000)
        self.assertGreater(low, 1000 * 4 * 30 + 1000 * 5 * 30)                 # retries on top of areas and cells
        self.assertLess(high, 500_000)

    def test_spans_group_consecutive_days(self):
        d = date(2023, 6, 1)
        days = [d, d + timedelta(days=1), d + timedelta(days=2), d + timedelta(days=6)]
        self.assertEqual(job._spans(days), [(d, d + timedelta(days=2)), (d + timedelta(days=6), d + timedelta(days=6))])
        self.assertEqual(job._spans([]), [])


HOURLY = json.loads((Path(__file__).with_name("fixtures") / "weatherkit_hourly_2023-06-01.json").read_text())


class RecordedSession:
    """Answers every GET with one recorded WeatherKit reply and remembers the query."""

    def __init__(self, reply):
        self.reply, self.params = reply, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.params.append(params)
        reply = self.reply
        return type("Reply", (), {"raise_for_status": lambda self: None, "json": lambda self: reply})()


class WeatherKitHourly(unittest.TestCase):
    """Past hourly values, parsed from the recorded reply for 1 Jun 2023 (a wet morning)."""

    TZ = ZoneInfo("Asia/Jakarta")
    START = datetime(2023, 6, 1, tzinfo=ZoneInfo("Asia/Jakarta"))

    def client(self, hours):
        wk = job.WeatherKit.__new__(job.WeatherKit)
        wk.tz, wk.calls, wk._token = self.TZ, 0, lambda: "made-up-token"
        wk.session = RecordedSession({"forecastHourly": {"hours": hours}})
        return wk

    def test_the_recorded_day_parses_hour_by_hour(self):
        wk = self.client(HOURLY["hours"])
        hours = wk.hourly(0.0, 0.0, self.START, self.START + timedelta(days=1))
        self.assertEqual(len(hours), 24)
        nine = datetime(2023, 6, 1, 2, tzinfo=timezone.utc)  # 09:00 WIB
        self.assertEqual(hours[nine], {"cloud_cover": 0.92, "rain_mm": 5.19, "daylight": True, "condition": "Rain"})
        self.assertEqual(wk.session.params[0]["dataSets"], "forecastHourly")
        self.assertEqual((wk.session.params[0]["hourlyStart"], wk.calls), (HOURLY["hourlyStart"], 1))

    def test_only_hours_inside_the_range_are_kept(self):
        wk = self.client(HOURLY["hours"])
        start = datetime(2023, 6, 1, 9, tzinfo=self.TZ)
        hours = wk.hourly(0.0, 0.0, start, start + timedelta(hours=3))
        self.assertEqual([t.astimezone(self.TZ).hour for t in hours], [9, 10, 11])

    def test_an_hour_given_twice_keeps_only_values_that_agree(self):
        nine = next(h for h in HOURLY["hours"] if h["forecastStart"] == "2023-06-01T02:00:00Z")
        twice = HOURLY["hours"] + [nine | {"precipitationAmount": 0.0}]  # same cloud, different rain
        hours = self.client(twice).hourly(0.0, 0.0, self.START, self.START + timedelta(days=1))
        t = datetime(2023, 6, 1, 2, tzinfo=timezone.utc)
        self.assertEqual((hours[t]["cloud_cover"], hours[t]["rain_mm"]), (0.92, None))


class ScriptedWeatherKit(job.WeatherKit):
    """The real history() over scripted daily() replies: no key, no network.

    `drop`: days never returned. `extra`: (day, mm) returned again in any request covering it. `later`: values that
    every request after the first returns instead. `fail_on`: numbers of the requests (1 = first) that raise."""

    def __init__(self, rain, drop=(), extra=(), later=None, fail_on=()):
        self.tz, self.asked = ZoneInfo("Asia/Jakarta"), []
        self.rain, self.drop, self.extra, self.later, self.fail_on = rain, set(drop), list(extra), later or {}, set(fail_on)

    def _day(self, d, mm):
        start = datetime.combine(d, time(0), self.tz).astimezone(timezone.utc)  # local midnight, as Apple sends it
        return {"forecastStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "precipitationAmount": mm}

    def daily(self, lat, lon, start=None, end=None):
        self.asked.append((start, end))
        if len(self.asked) in self.fail_on:
            raise ConnectionError("weatherkit.apple.com unreachable")
        values = self.rain | (self.later if len(self.asked) > 1 else {})
        asked = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        days = [self._day(d, values[d]) for d in asked if d in values and d not in self.drop]
        days += [self._day(d, mm) for d, mm in self.extra if start <= d <= end]
        return {"forecastDaily": {"days": days}}


class WeatherKitHistory(unittest.TestCase):
    START, END = date(2023, 5, 23), date(2023, 6, 11)  # two 10-day requests
    RAIN = {date(2023, 5, 23) + timedelta(days=i): 5.0 for i in range(25)}

    def test_a_day_missing_from_every_reply_is_retried_once_then_stays_missing(self):
        wk = ScriptedWeatherKit(self.RAIN, drop={date(2023, 6, 1)})
        with self.assertLogs("nasa_power_job", level="WARNING") as logs:
            got = wk.history(0.0, 0.0, self.START, self.END)
        self.assertEqual((wk.retries, len(wk.asked)), (1, 3))
        self.assertNotIn(date(2023, 6, 1), got)
        self.assertEqual(len(got), 19)
        self.assertIn("9 of 10 days", logs.output[0])
        self.assertIn("recovered 0 of 1", logs.output[1])

    def test_a_retry_that_disagrees_makes_that_day_missing(self):
        wk = ScriptedWeatherKit(self.RAIN, drop={date(2023, 6, 1)}, later={date(2023, 5, 29): 9.0})
        with self.assertLogs("nasa_power_job", level="WARNING") as logs:
            got = wk.history(0.0, 0.0, self.START, self.END)
        self.assertIsNone(got[date(2023, 5, 29)])  # 5.0 first, 9.0 on the retry: no telling which is right
        self.assertIn("1 day(s) disagree", logs.output[1])

    def test_a_failing_retry_leaves_the_gap_and_keeps_the_rest(self):
        wk = ScriptedWeatherKit(self.RAIN, drop={date(2023, 6, 1)}, fail_on={2})  # request 2 is the retry
        with self.assertLogs("nasa_power_job", level="WARNING") as logs:
            got = wk.history(0.0, 0.0, self.START, self.END)
        self.assertEqual((len(wk.asked), len(got)), (3, 19))
        self.assertNotIn(date(2023, 6, 1), got)
        self.assertIn("retry 2023-05-28 to 2023-06-06 failed", "\n".join(logs.output))

    def test_a_date_returned_twice_counts_once_or_as_missing(self):
        wk = ScriptedWeatherKit(self.RAIN, extra=[(date(2023, 5, 25), 5.0), (date(2023, 5, 26), 9.0)])
        got = wk.history(0.0, 0.0, self.START, self.END)
        self.assertEqual(wk.retries, 0)
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
        failed = job.run(self.PLOTS, out.name, self.TODAY, session=None, plot_key_secret=SECRET, fetch=fetch)
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
            job.run(self.PLOTS, out, self.TODAY, session=None, plot_key_secret=SECRET, fetch=self.fetch)
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


SCHEMA3_DIR = Path(__file__).with_name("fixtures") / "schema3"
try:
    import jsonschema
except ImportError:  # the schema-3 tests that don't need it still run
    jsonschema = None


class Schema3(unittest.TestCase):
    """Per-plot files: only what L5 and L8 read on that plot's phone, in a folder nobody can guess."""

    TODAY = date(2026, 10, 11)
    PLOTS = [{"plot_id": "plot-alpha", "lat": 0.01, "lon": 0.01}, {"plot_id": "plot-bravo", "lat": 0.02, "lon": 0.03}]

    def setUp(self):
        self.calls = []

    def fetch(self, lat, lon, start, end, params, session):
        return FullRun.fake_fetch(self, lat, lon, start, end, params, session)

    def run_job(self, **kwargs):
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        self.assertEqual(job.run(self.PLOTS, out.name, self.TODAY, session=None, fetch=self.fetch, **kwargs), 0)
        return Path(out.name)

    def test_each_plot_gets_only_its_own_record(self):
        out = self.run_job(plot_key_secret=SECRET)
        cell = json.loads(next((out / "cells").glob("*/latest.json")).read_text())
        for plot in self.PLOTS:
            path = out / "plots" / job.plot_key(plot["plot_id"], SECRET) / "latest.json"
            text = path.read_text()
            mine = json.loads(text)
            self.assertEqual(list(mine), ["schema", "generated_at", "l5", "stage_a", "attribution"])
            self.assertEqual((mine["schema"], mine["generated_at"]), (3, cell["generated_at"]))
            self.assertEqual(mine["l5"], {k: v for k, v in cell["l5"][plot["plot_id"]].items() if k != "weatherkit_area"})
            self.assertEqual(mine["stage_a"]["10"], {"avg_mm": cell["climate"]["stage_a"]["10"]["avg_mm"]})
            for leak in ("plot-alpha", "plot-bravo", "weatherkit_area", '"lat"', '"lon"', "cell", "gate", "in_band"):
                self.assertNotIn(leak, text)

    def test_the_folder_is_an_hmac_of_the_plot_id(self):
        out = self.run_job(plot_key_secret=SECRET)
        keys = sorted(p.name for p in (out / "plots").iterdir())
        self.assertEqual(keys, sorted(job.plot_key(p["plot_id"], SECRET) for p in self.PLOTS))
        key = job.plot_key("plot-alpha", SECRET)
        self.assertRegex(key, r"^[0-9a-f]{32}$")
        self.assertEqual(key, job.plot_key("plot-alpha", SECRET))              # stable
        self.assertNotEqual(key, job.plot_key("plot-alpha", "another-secret"))  # nothing without the secret
        self.assertNotEqual(key, job.plot_key("plot-alphb", SECRET))

    def test_without_a_secret_only_schema_2_is_written(self):
        with self.assertLogs("nasa_power_job", level="WARNING") as logs:
            out = self.run_job()
        self.assertFalse((out / "plots").exists())
        self.assertTrue((out / "plots.json").is_file())
        self.assertIn("PLOT_KEY_SECRET not set", logs.output[0])

    def test_schema_2_can_be_switched_off(self):
        out = self.run_job(plot_key_secret=SECRET, write_schema2=False)
        self.assertFalse((out / "plots.json").exists())
        self.assertEqual(list((out / "cells").glob("*/latest.json")), [])
        self.assertEqual(len(list((out / "plots").glob("*/latest.json"))), 2)
        self.assertEqual(len(list((out / "cells").glob("*/qa_log.json"))), 1)  # backend-only, kept either way

    def test_attribution_follows_the_data(self):
        inputs = json.loads((SCHEMA3_DIR / "inputs.json").read_text())
        sources = {name: [a["source"] for a in job.build_schema3(r, inputs["climate"], inputs["generated_at"])["attribution"]]
                   for name, r in inputs["records"].items()}
        self.assertEqual(sources, {"normal": ["apple-weather", "nasa-power"], "weatherkit_fallback": ["nasa-power"],
                                   "null_p30": ["nasa-power"]})

    def test_the_fixtures_are_what_the_job_writes(self):
        inputs = json.loads((SCHEMA3_DIR / "inputs.json").read_text())
        for name, record in inputs["records"].items():
            with self.subTest(name):
                expected = json.loads((SCHEMA3_DIR / f"{name}.json").read_text())
                self.assertEqual(job.build_schema3(record, inputs["climate"], inputs["generated_at"]), expected)

    @unittest.skipUnless(jsonschema, "jsonschema not installed")
    def test_files_match_the_json_schema(self):
        validator = jsonschema.Draft202012Validator(json.loads((Path(job.HERE) / "schema3.schema.json").read_text()))
        out = self.run_job(plot_key_secret=SECRET)
        files = [json.loads(p.read_text()) for p in (out / "plots").glob("*/latest.json")]
        files += [json.loads((SCHEMA3_DIR / f"{n}.json").read_text()) for n in ("normal", "weatherkit_fallback", "null_p30")]
        for f in files:
            self.assertEqual([e.message for e in validator.iter_errors(f)], [])
        normal = json.loads((SCHEMA3_DIR / "normal.json").read_text())
        broken = [normal | {"cell": {"plots": ["x"]}},                                            # nothing extra
                  normal | {"l5": normal["l5"] | {"p30_quality": "none"}},                         # none means null values
                  normal | {"attribution": [a for a in normal["attribution"] if a["source"] != "apple-weather"]}]
        for f in broken:
            self.assertFalse(validator.is_valid(f))


if __name__ == "__main__":
    unittest.main()
