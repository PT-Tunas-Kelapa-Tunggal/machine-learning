"""Tests for l5_water_balance.py, against FAO-56 worked examples and tables where they exist.

    cd weather && python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l5_water_balance as wb  # noqa: E402
import solar  # noqa: E402


class Atmosphere(unittest.TestCase):
    def test_saturation_and_slope_match_fao56_annex_2(self):
        self.assertAlmostEqual(wb.saturation_vapour_pressure(25.0), 3.168, places=3)  # Table 2.3
        self.assertAlmostEqual(wb.vapour_pressure_slope(25.0), 0.189, places=3)      # Table 2.4

    def test_psychrometric_constant_matches_fao56_example_2(self):
        self.assertAlmostEqual(wb.psychrometric_constant(1800.0), 0.054, places=3)   # 1,800 m: P 81.8 kPa
        self.assertAlmostEqual(wb.psychrometric_constant(0.0), 0.0674, places=4)


class Radiation(unittest.TestCase):
    def test_net_radiation_matches_fao56_example_11(self):
        # Rio de Janeiro (22°54'S) at sea level, 15 May: Rs 14.5, Tmax 25.1, Tmin 19.1, ea 2.1 → Rn 7.6 MJ m-2 day-1
        ra = solar.extraterrestrial_daily(-22.9, 135)
        self.assertAlmostEqual(ra, 25.1, delta=0.05)
        self.assertAlmostEqual(wb.net_radiation(14.5, 25.1, 19.1, ra, ea=2.1), 7.6, delta=0.05)

    def test_brighter_days_carry_more_net_radiation(self):
        ra = solar.extraterrestrial_daily(1.0, 180)
        self.assertLess(wb.net_radiation(10.0, 31.0, 24.0, ra), wb.net_radiation(20.0, 31.0, 24.0, ra))


class Et0(unittest.TestCase):
    RA = 36.0  # about the trial site's Ra in September

    def test_hargreaves_by_hand(self):
        # 0.0023 × (28 + 17.8) × √8 × 0.408 × 36
        expected = 0.0023 * 45.8 * math.sqrt(8.0) * 0.408 * 36.0
        self.assertAlmostEqual(wb.hargreaves_et0(32.0, 24.0, self.RA), expected, places=9)
        self.assertAlmostEqual(expected, 4.376, places=3)

    def test_hargreaves_grows_with_the_day_night_range_and_is_zero_without_one(self):
        self.assertEqual(wb.hargreaves_et0(27.0, 27.0, self.RA), 0.0)
        self.assertLess(wb.hargreaves_et0(30.0, 25.0, self.RA), wb.hargreaves_et0(32.0, 23.0, self.RA))

    def test_priestley_taylor_by_hand(self):
        rs, tmax, tmin = 18.0, 31.0, 24.0
        delta, gamma = wb.vapour_pressure_slope(27.5), wb.psychrometric_constant(0.0)
        rn = wb.net_radiation(rs, tmax, tmin, self.RA)
        self.assertAlmostEqual(wb.priestley_taylor_et0(rs, tmax, tmin, self.RA),
                               1.26 * delta / (delta + gamma) * rn / 2.45, places=9)

    def test_priestley_taylor_never_negative(self):
        self.assertLess(wb.net_radiation(0.1, 26.0, 25.0, self.RA), 0.0)  # an almost black day loses energy
        self.assertEqual(wb.priestley_taylor_et0(0.1, 26.0, 25.0, self.RA), 0.0)

    def test_dark_days_do_not_gain_longwave_energy(self):
        # Under Rs/Rso = 0.3 the cloud term is held at its 0.3 value instead of turning negative
        self.assertAlmostEqual(wb.net_radiation(2.0, 31.0, 24.0, self.RA) - 0.77 * 2.0,
                               wb.net_radiation(5.0, 31.0, 24.0, self.RA) - 0.77 * 5.0, places=9)


class Balance(unittest.TestCase):
    END = date(2026, 9, 30)

    def days(self, value, n=30):
        return {self.END - timedelta(days=i): value for i in range(n)}

    def test_rain_minus_et0_over_30_days(self):
        self.assertEqual(wb.water_balance(self.days(3.0), self.days(4.5), self.END), -45.0)

    def test_only_the_30_days_up_to_end_count(self):
        rain = self.days(3.0, n=40) | {self.END + timedelta(days=1): 500.0}
        self.assertEqual(wb.water_balance(rain, self.days(1.0), self.END), 60.0)

    def test_a_missing_day_gives_no_balance_and_an_unknown_flag(self):
        rain = self.days(3.0)
        rain[self.END - timedelta(days=4)] = None
        balance = wb.water_balance(rain, self.days(1.0), self.END)
        self.assertIsNone(balance)
        self.assertIsNone(wb.deficit(balance))

    def test_deficit_is_strictly_under_the_threshold(self):
        self.assertFalse(wb.deficit(0.0))
        self.assertTrue(wb.deficit(-0.1))
        self.assertTrue(wb.deficit(-50.0, threshold=-40.0))


if __name__ == "__main__":
    unittest.main()
