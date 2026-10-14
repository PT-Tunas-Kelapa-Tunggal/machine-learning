"""Tests for l9_irradiance_check.py: the ALLSKY ≤ CLRSKY ≤ Ra sanity check (log-only).

    cd weather && python -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import l9_irradiance_check as qa  # noqa: E402
import solar  # noqa: E402

LAT = 1.0  # the trial plot's NASA POWER cell centre
DAY = date(2026, 7, 1)
RA = solar.extraterrestrial_daily(LAT, solar.day_of_year(DAY))  # about 34.7 MJ m-2 day-1


class IrradianceCheck(unittest.TestCase):
    def test_typical_days_pass(self):
        # Values like the trial cell's 2026 data: ALLSKY 0.4–0.6 Ra, CLRSKY about 0.66 Ra
        allsky = {DAY + timedelta(days=i): 0.5 * RA for i in range(10)}
        clrsky = {DAY + timedelta(days=i): 0.66 * RA for i in range(5)}
        out = qa.check_irradiance(allsky, clrsky, LAT)
        self.assertEqual((out["days_checked"], out["days_with_clrsky"], out["days_flagged"]), (10, 5, 0))
        self.assertEqual((out["first_day"], out["last_day"]), ("2026-07-01", "2026-07-10"))
        self.assertEqual(out["status"], "log-only")

    def test_allsky_above_clrsky_is_flagged(self):
        out = qa.check_irradiance({DAY: 0.70 * RA}, {DAY: 0.60 * RA}, LAT)
        self.assertEqual(out["flagged"]["2026-07-01"]["codes"], ["allsky_gt_clrsky"])

    def test_clrsky_above_ra_is_flagged(self):
        out = qa.check_irradiance({DAY: 0.5 * RA}, {DAY: 1.1 * RA}, LAT)
        self.assertEqual(out["flagged"]["2026-07-01"]["codes"], ["clrsky_gt_ra"])

    def test_allsky_above_ra_breaks_every_bound_it_can(self):
        out = qa.check_irradiance({DAY: 1.2 * RA}, {DAY: 0.66 * RA}, LAT)
        self.assertEqual(out["flagged"]["2026-07-01"]["codes"], ["allsky_gt_ra", "allsky_gt_clrsky"])

    def test_without_clrsky_the_fao56_clear_sky_is_the_bound(self):
        out = qa.check_irradiance({DAY: 0.80 * RA}, {}, LAT)  # over 0.75 Ra, under Ra
        self.assertEqual(out["flagged"]["2026-07-01"]["codes"], ["allsky_gt_rso_fao56"])
        self.assertEqual(out["days_with_clrsky"], 0)

    def test_tolerance_is_relative_and_inclusive(self):
        at_edge = 0.60 * RA * (1 + qa.IRRADIANCE_TOLERANCE)
        self.assertEqual(qa.check_irradiance({DAY: at_edge}, {DAY: 0.60 * RA}, LAT)["days_flagged"], 0)
        self.assertEqual(qa.check_irradiance({DAY: at_edge * 1.001}, {DAY: 0.60 * RA}, LAT)["days_flagged"], 1)
        self.assertEqual(qa.check_irradiance({DAY: 0.61 * RA}, {DAY: 0.60 * RA}, LAT, tolerance=0.0)["days_flagged"], 1)

    def test_missing_values_are_skipped_not_zero(self):
        out = qa.check_irradiance({DAY: None, DAY + timedelta(days=1): 0.5 * RA}, {DAY: None}, LAT)
        self.assertEqual((out["days_checked"], out["days_flagged"]), (1, 0))

    def test_no_data_gives_an_empty_check(self):
        out = qa.check_irradiance({}, None, LAT)
        self.assertEqual((out["days_checked"], out["first_day"], out["flagged"]), (0, None, {}))


if __name__ == "__main__":
    unittest.main()
