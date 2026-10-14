"""Runs every case in solar_cases.json against solar.py, plus a wide cross-check against pvlib when it is installed.

    cd weather && python -m unittest discover -s tests -v
"""
import json
import math
import random
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solar  # noqa: E402

CASES = json.loads((Path(__file__).with_name("solar_cases.json")).read_text())["cases"]

try:
    import pandas as pd
    import pvlib
except ImportError:  # the cases above still run; only the live cross-check is skipped
    pvlib = None


def to_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def run(case):
    f, x = case["function"], case["input"]
    if f == "julian_day":
        return solar.julian_day(to_utc(x["utc"]))
    if f == "day_of_year":
        return solar.day_of_year(date.fromisoformat(x["date"]))
    if f == "solar_position":
        return solar.solar_position(x["lat"], x["lon"], to_utc(x["utc"]))
    if f == "extraterrestrial_daily":
        return solar.extraterrestrial_daily(x["lat"], x["doy"])
    if f == "clear_sky_fao56":
        return solar.clear_sky_fao56(x["ra"], x["elevation_m"])
    if f == "haurwitz_ghi":
        return solar.haurwitz_ghi(x["zenith_deg"])
    raise ValueError(f"unknown function {f}")


def separation_deg(zen1, az1, zen2, az2):
    """Angle between two sun positions: unlike azimuth alone, it stays small when the sun is near the zenith."""
    z1, z2, da = math.radians(zen1), math.radians(zen2), math.radians(az1 - az2)
    c = math.cos(z1) * math.cos(z2) + math.sin(z1) * math.sin(z2) * math.cos(da)
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


class SolarCases(unittest.TestCase):
    def test_cases(self):
        for case in CASES:
            with self.subTest(case["name"]):
                actual, expected, tol = run(case), case["expected"], case["tolerance"]
                if case["function"] == "solar_position":
                    for key, t in tol.items():
                        diff = abs(actual[key] - expected[key])
                        if key == "azimuth_deg":
                            diff = min(diff, 360 - diff)
                        self.assertLessEqual(diff, t, f"{key}: {actual[key]} vs {expected[key]}")
                else:
                    self.assertLessEqual(abs(actual - expected), tol, f"{actual} vs {expected}")

    def test_every_function_is_covered(self):
        covered = {c["function"] for c in CASES}
        public = {"julian_day", "day_of_year", "solar_position", "extraterrestrial_daily", "clear_sky_fao56",
                  "haurwitz_ghi"}
        self.assertEqual(public - covered, set())

    def test_naive_datetime_is_refused(self):
        with self.assertRaises(ValueError):
            solar.julian_day(datetime(2026, 10, 11, 3, 0))

    def test_polar_day_and_night_do_not_crash(self):
        self.assertAlmostEqual(solar.extraterrestrial_daily(80.0, 355), 0.0, places=6)  # polar night
        self.assertGreater(solar.extraterrestrial_daily(80.0, 172), 40.0)               # polar day


@unittest.skipUnless(pvlib, "pvlib not installed (pip install -r requirements-dev.txt)")
class AgainstPvlib(unittest.TestCase):
    def test_sun_position_matches_spa_within_noaa_accuracy(self):
        rng = random.Random(7)
        worst_true, worst_apparent = 0.0, 0.0
        for _ in range(400):
            lat, lon = rng.uniform(-60, 60), rng.uniform(-180, 180)
            when = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=rng.uniform(0, 40 * 365.25 * 86400))
            ours = solar.solar_position(lat, lon, when)
            ref = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([when]), lat, lon, method="nrel_numpy").iloc[0]
            worst_true = max(worst_true, separation_deg(ours["true_zenith_deg"], ours["azimuth_deg"],
                                                        ref["zenith"], ref["azimuth"]))
            if ref["apparent_elevation"] > 5:
                worst_apparent = max(worst_apparent, abs(ours["elevation_deg"] - ref["apparent_elevation"]))
        self.assertLess(worst_true, 0.02)      # NOAA quotes about 1 arc-minute (0.017°)
        self.assertLess(worst_apparent, 0.02)

    def test_haurwitz_matches_pvlib(self):
        for z in (0.0, 15.0, 45.0, 70.0, 88.0, 95.0):
            self.assertAlmostEqual(solar.haurwitz_ghi(z), float(pvlib.clearsky.haurwitz(pd.Series([z])).iloc[0, 0]), places=6)


if __name__ == "__main__":
    unittest.main()
