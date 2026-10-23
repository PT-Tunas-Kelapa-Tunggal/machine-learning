"""Tests for l6_vari_confidence.py: the factors, the flight weather and its fallbacks, P30 on the scan day, the CLI.

    cd weather && python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l6_vari_confidence as conf  # noqa: E402
import nasa_power_job as job  # noqa: E402

FIXTURES = Path(__file__).with_name("fixtures")
HOURLY = json.loads((FIXTURES / "weatherkit_hourly_2023-06-01.json").read_text())
SCAN = json.loads((FIXTURES / "scan_example.json").read_text())
PLOT = {"plot_id": "plot-alpha", "lat": 0.5, "lon": 101.5}  # made-up place, not a real plot
NORMALS = {str(m): 200.0 for m in range(1, 13)}
TZ = ZoneInfo("Asia/Jakarta")


def utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


RECORDED_HOURS = {utc(h["forecastStart"]): {"cloud_cover": h["cloudCover"], "rain_mm": h["precipitationAmount"],
                                            "daylight": h["daylight"], "condition": h["conditionCode"]}
                  for h in HOURLY["hours"]}


def day_reply(*days):
    """A WeatherKit daily reply: (local date, daytime cloud, daytime rain, overnight rain) per day."""
    out = []
    for d, cloud, day_rain, night_rain in days:
        start = datetime.combine(d, datetime.min.time(), TZ).astimezone(timezone.utc)
        out.append({"forecastStart": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "daytimeForecast": {"cloudCover": cloud, "precipitationAmount": day_rain},
                    "overnightForecast": {"precipitationAmount": night_rain}})
    return {"forecastDaily": {"days": out}}


class FakeWeatherKit:
    tz = TZ

    def __init__(self, hours=None, daily=None, history=None, hourly_fails=False, daily_fails=False):
        self.hours, self.daily_reply, self.rain = hours or {}, daily, history or {}
        self.hourly_fails, self.daily_fails = hourly_fails, daily_fails

    def hourly(self, lat, lon, start, end):
        if self.hourly_fails:
            raise ConnectionError("no hourly")
        return {t: h for t, h in self.hours.items() if start <= t < end}

    def daily(self, lat, lon, start=None, end=None):
        if self.daily_fails or self.daily_reply is None:
            raise ConnectionError("no daily")
        return self.daily_reply

    def history(self, lat, lon, start, end):
        return {d: v for d, v in self.rain.items() if start <= d <= end}


def nasa(rain_mm=3.0, allsky=None, fail=False):
    """A fake fetch_nasa_power: the same rain every day, and ALLSKY when given."""
    def fetch(lat, lon, start, end, params, session):
        if fail:
            raise ConnectionError("no NASA POWER")
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        values = {job.RAIN: {d: rain_mm for d in days}}
        if allsky is not None:
            values[job.SUN] = {d: allsky for d in days}
        return values, {}, ["MERRA2"]
    return fetch


class Factors(unittest.TestCase):
    def test_sun_bands(self):
        cases = {45.0: (1.0, "high"), 44.9: (0.85, "medium"), 30.0: (0.85, "medium"), 29.9: (0.6, "low"),
                 15.0: (0.6, "low"), 14.9: (0.3, "very_low"), None: (0.6, "unknown")}
        for elevation, expected in cases.items():
            self.assertEqual(conf.sun_factor(elevation), expected, elevation)

    def test_cloud_and_clearness_classes(self):
        for cover, cls in {0.2: "clear", 0.21: "broken", 0.79: "broken", 0.8: "overcast", None: None}.items():
            self.assertEqual(conf.cloud_class(cover), cls, cover)
        for k, cls in {0.7: "clear", 0.69: "broken", 0.36: "broken", 0.35: "overcast", None: None}.items():
            self.assertEqual(conf.clearness_class(k), cls, k)

    def test_wet_leaves(self):
        cases = {(0.5, 0.0): "rain_before", (0.4, 25.0): "dry", (0.4, 25.1): "heavy_day_before",
                 (None, 30.0): "heavy_day_before", (None, 3.0): "unknown", (0.0, None): "unknown"}
        for (r6, r24), cls in cases.items():
            self.assertEqual(conf.wet_class(r6, r24), cls, (r6, r24))

    def test_water_follows_the_l5_record(self):
        cases = [(None, "unknown"), ({"water_stress": None, "p30_quality": "none"}, "unknown"),
                 ({"water_stress": True, "p30_quality": "ok"}, "water_stress"),
                 ({"water_stress": False, "p30_quality": "ok"}, "ok"),
                 ({"water_stress": False, "p30_quality": "fallback"}, "fallback")]
        for record, cls in cases:
            self.assertEqual(conf.water_class(record), cls, record)

    def test_labels(self):
        for value, label in {0.8: "tinggi", 0.799: "sedang", 0.5: "sedang", 0.499: "rendah"}.items():
            self.assertEqual(conf.label_for(value), label, value)


class Combining(unittest.TestCase):
    GOOD_CLOUD = {"cover": 0.1, "class": "clear", "source": "weatherkit-hourly"}
    DRY = {"last_6h_mm": 0.0, "last_24h_mm": 2.0, "source": "weatherkit-hourly"}
    WATER_OK = {"p30_mm": 120.0, "water_stress": False, "p30_quality": "ok"}

    def test_a_good_flight_on_a_watered_plot_is_fully_trusted(self):
        r = conf.vari_confidence(55.0, self.GOOD_CLOUD, self.DRY, self.WATER_OK)
        self.assertEqual((r["confidence"], r["label"], r["final"], r["reasons"]), (1.0, "tinggi", True, []))

    def test_the_confidence_is_the_product_of_the_factors(self):
        r = conf.vari_confidence(24.0, {"cover": 0.95, "class": "overcast", "source": "weatherkit-hourly"},
                                 {"last_6h_mm": 12.4, "last_24h_mm": 31.0, "source": "weatherkit-hourly"},
                                 {"p30_mm": 41.0, "water_stress": True, "p30_quality": "fallback"})
        self.assertEqual(r["components"], {"pengukuran": round(0.6 * 0.85 * 0.7, 3), "air": 0.6})
        self.assertEqual((r["confidence"], r["label"]), (round(0.6 * 0.85 * 0.7 * 0.6, 3), "rendah"))
        self.assertEqual([x["code"] for x in r["reasons"]], ["sun_low", "cloud_overcast", "wet_rain_before",
                                                             "water_water_stress"])
        self.assertIn("12,4 mm", r["reasons"][2]["text"])  # Indonesian decimal comma
        self.assertIn("P30 41 mm, di bawah 60 mm", r["reasons"][3]["text"])

    def test_nothing_unknown_scores_one(self):
        r = conf.vari_confidence(None, {}, {}, None)
        self.assertFalse(r["final"])
        self.assertTrue(all(f["value"] < 1 for f in r["factors"].values()))
        self.assertEqual({x["code"] for x in r["reasons"]}, {"sun_unknown", "cloud_unknown", "wet_unknown",
                                                              "water_unknown"})
        self.assertLess(r["confidence"], 0.5)

    def test_one_unknown_input_is_enough_to_be_not_final(self):
        r = conf.vari_confidence(55.0, self.GOOD_CLOUD, {"last_6h_mm": 0.0, "last_24h_mm": None}, self.WATER_OK)
        self.assertEqual((r["factors"]["wet"]["class"], r["final"]), ("unknown", False))
        self.assertLess(r["confidence"], 1.0)


class FlightWeather(unittest.TestCase):
    """A flight at 09:40-10:05 WIB on 1 Jun 2023, a wet morning, against the recorded hourly reply."""

    START, END = utc(SCAN["flight_start"]), utc(SCAN["flight_end"])

    def test_cloud_is_the_mean_over_the_hours_the_flight_touches(self):
        nine, ten = (RECORDED_HOURS[utc(f"2023-06-01T0{h}:00:00Z")]["cloud_cover"] for h in (2, 3))
        self.assertEqual(conf.flight_cloud(RECORDED_HOURS, self.START, self.END), round((nine + ten) / 2, 3))

    def test_rain_before_take_off_needs_every_hour(self):
        hours = [utc("2023-05-31T20:00:00Z") + timedelta(hours=i) for i in range(7)]  # 03:00-09:59 WIB
        expected = round(sum(RECORDED_HOURS[t]["rain_mm"] for t in hours), 2)
        self.assertEqual(conf.rain_before(RECORDED_HOURS, self.START, 6), expected)
        self.assertIsNone(conf.rain_before(RECORDED_HOURS, self.START, 24))  # the day before isn't in the reply
        gap = {t: h for t, h in RECORDED_HOURS.items() if t != hours[3]}
        self.assertIsNone(conf.rain_before(gap, self.START, 6))  # a missing hour is unknown, never 0 mm

    def test_hourly_first(self):
        cloud, rain = conf.flight_conditions(PLOT["lat"], PLOT["lon"], self.START, self.END,
                                             weatherkit=FakeWeatherKit(hours=RECORDED_HOURS), nasa_fetch=None)
        self.assertEqual((cloud["source"], cloud["class"], rain["source"]), ("weatherkit-hourly", "overcast",
                                                                            "weatherkit-hourly"))
        self.assertEqual(conf.wet_class(rain["last_6h_mm"], rain["last_24h_mm"]), "rain_before")

    def test_the_hourly_request_starts_on_a_whole_hour(self):
        asked = []

        class Recording(FakeWeatherKit):
            def hourly(self, lat, lon, start, end):
                asked.append(start)
                return super().hourly(lat, lon, start, end)

        conf.flight_conditions(PLOT["lat"], PLOT["lon"], self.START, self.END, weatherkit=Recording(), nasa_fetch=None)
        self.assertEqual(asked, [utc("2023-05-31T02:00:00Z")])  # 24 h before 02:40, down to the hour

    def test_then_the_daily_day_parts(self):
        reply = day_reply((date(2023, 5, 31), 0.4, 6.0, 22.0), (date(2023, 6, 1), 0.9, 12.0, 1.0))
        with self.assertLogs("l6_vari_confidence", level="WARNING"):
            cloud, rain = conf.flight_conditions(PLOT["lat"], PLOT["lon"], self.START, self.END,
                                                 weatherkit=FakeWeatherKit(daily=reply, hourly_fails=True), nasa_fetch=None)
        self.assertEqual((cloud["source"], cloud["cover"], cloud["class"]), ("weatherkit-daytime", 0.9, "overcast"))
        self.assertEqual(rain, {"last_6h_mm": None, "last_24h_mm": 28.0, "source": "weatherkit-dayparts"})

    def test_then_nasa_power_for_cloud_only(self):
        day = date(2023, 6, 1)
        cell = job.grid_cell(PLOT["lat"], PLOT["lon"])
        clear = conf.solar.clear_sky_fao56(conf.solar.extraterrestrial_daily(cell["lat"], conf.solar.day_of_year(day)))
        with self.assertLogs("l6_vari_confidence", level="WARNING") as logs:
            cloud, rain = conf.flight_conditions(PLOT["lat"], PLOT["lon"], self.START, self.END,
                                                 weatherkit=FakeWeatherKit(hourly_fails=True, daily_fails=True),
                                                 nasa_fetch=nasa(allsky=0.5 * clear))
        self.assertEqual(len(logs.output), 2)  # hourly, then daily
        self.assertEqual((cloud["source"], cloud["class"], cloud["clearness"]), ("nasa-power", "broken", 0.5))
        self.assertEqual(rain["source"], None)

    def test_the_real_nasa_power_fetch_gives_cloud(self):
        # nasa() above skips fetch_nasa_power's own checks; a sunlight-only reply carries no rain unit
        day = date(2023, 6, 1)
        cell = job.grid_cell(PLOT["lat"], PLOT["lon"])
        clear = conf.solar.clear_sky_fao56(conf.solar.extraterrestrial_daily(cell["lat"], conf.solar.day_of_year(day)))
        payload = {"header": {"fill_value": -999.0, "sources": ["MERRA2"]},
                   "parameters": {job.SUN: {"units": "MJ/m^2/day"}},
                   "properties": {"parameter": {job.SUN: {f"{day:%Y%m%d}": 0.5 * clear}}}}
        session = mock.Mock(get=mock.Mock(return_value=mock.Mock(status_code=200, json=lambda: payload)))
        with self.assertNoLogs("l6_vari_confidence", level="WARNING"):
            cloud, _ = conf.flight_conditions(PLOT["lat"], PLOT["lon"], self.START, self.END, weatherkit=None,
                                              nasa_fetch=job.fetch_nasa_power, session=session)
        self.assertEqual((cloud["source"], cloud["class"], cloud["clearness"]), ("nasa-power", "broken", 0.5))

    def test_nothing_reachable_leaves_everything_unknown(self):
        with self.assertLogs("l6_vari_confidence", level="WARNING"):
            cloud, rain = conf.flight_conditions(PLOT["lat"], PLOT["lon"], self.START, self.END, weatherkit=None,
                                                 nasa_fetch=nasa(fail=True))
        self.assertEqual((cloud["class"], rain["last_6h_mm"], rain["last_24h_mm"]), (None, None, None))


class P30OnTheScanDay(unittest.TestCase):
    DAY = date(2023, 6, 1)

    def weatherkit(self, gap=None):
        rain = {self.DAY - timedelta(days=i): 2.5 for i in range(1, 80)}
        if gap:
            rain.pop(gap)
        return FakeWeatherKit(history=rain)

    def test_weatherkit_first(self):
        p30 = conf.scan_p30(PLOT["lat"], PLOT["lon"], self.DAY, NORMALS, weatherkit=self.weatherkit(), nasa_fetch=nasa())
        self.assertEqual((p30["p30_source"], p30["p30_quality"], p30["p30_mm"]), ("weatherkit-history", "ok", 75.0))
        self.assertEqual(p30["p30_end_date"], "2023-05-31")  # the scan day itself is left out

    def test_a_weatherkit_gap_falls_back_to_checked_nasa_power(self):
        p30 = conf.scan_p30(PLOT["lat"], PLOT["lon"], self.DAY, NORMALS, weatherkit=self.weatherkit(gap=date(2023, 5, 20)),
                            nasa_fetch=nasa(rain_mm=1.5))
        self.assertEqual((p30["p30_source"], p30["p30_quality"], p30["water_stress"]), ("nasa-power", "fallback", True))
        self.assertEqual(p30["p30_missing_days"], {"weatherkit-history": 1, "nasa-power": 0})

    def test_no_source_gives_an_unknown_record(self):
        with self.assertLogs("l6_vari_confidence", level="WARNING"):
            p30 = conf.scan_p30(PLOT["lat"], PLOT["lon"], self.DAY, NORMALS, weatherkit=None, nasa_fetch=nasa(fail=True))
        self.assertEqual((p30["p30_quality"], p30["water_stress"]), ("none", None))


class WholeScan(unittest.TestCase):
    def weatherkit(self):
        rain = {date(2023, 6, 1) - timedelta(days=i): 2.5 for i in range(1, 80)}
        return FakeWeatherKit(hours=RECORDED_HOURS, history=rain)

    def test_a_wet_overcast_morning_lowers_the_confidence(self):
        r = conf.confidence_for_scan(SCAN, PLOT, NORMALS, weatherkit=self.weatherkit(), nasa_fetch=nasa(),
                                     computed_at="2026-10-14T00:00:00Z")
        self.assertEqual([r["schema"], r["scan_id"], r["computed_at"]], [1, "scan-0001", "2026-10-14T00:00:00Z"])
        f = r["factors"]
        self.assertEqual((f["cloud"]["class"], f["wet"]["class"], f["water"]["class"]), ("overcast", "rain_before", "ok"))
        self.assertGreaterEqual(f["sun"]["elevation_deg"], 45.0)
        self.assertEqual((r["confidence"], r["label"]), (round(1.0 * 0.85 * 0.7 * 1.0, 3), "sedang"))
        self.assertEqual(r["vari"], SCAN["vari"])
        text = json.dumps(r)
        for leak in ('"lat"', '"lon"', "101.5", "plot-alpha"):
            self.assertNotIn(leak, text)

    def test_the_cli_writes_the_result(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "plots.json").write_text(json.dumps([PLOT]))
            cell = job.grid_cell(PLOT["lat"], PLOT["lon"])
            climate = d / "weather" / "cells" / cell["id"] / "climate.json"
            climate.parent.mkdir(parents=True)
            climate.write_text(json.dumps({"computed_on": date.today().isoformat(), "normals_mm": NORMALS}))
            with mock.patch.object(conf.job.WeatherKit, "from_env", return_value=self.weatherkit()), \
                    mock.patch.object(conf.job, "fetch_nasa_power", nasa()), mock.patch.object(conf.logging, "basicConfig"):
                code = conf.main(["--scan", str(FIXTURES / "scan_example.json"), "--plots", str(d / "plots.json"),
                                  "--weather-dir", str(d / "weather"), "--out", str(d / "result.json")])
            result = json.loads((d / "result.json").read_text())
        self.assertEqual((code, result["scan_id"], result["label"]), (0, "scan-0001", "sedang"))


if __name__ == "__main__":
    unittest.main()
