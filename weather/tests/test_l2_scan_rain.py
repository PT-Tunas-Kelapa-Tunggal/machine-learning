"""Tests for l2_scan_rain.py: rain between scans and P30 at each scan date.

    cd weather && python -m unittest discover -s tests -v
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import krops_rules as rules  # noqa: E402
import l2_scan_rain as l2  # noqa: E402


def days(start, values):
    return {start + timedelta(days=i): v for i, v in enumerate(values)}


JAN1 = date(2026, 1, 1)


class RainBetween(unittest.TestCase):
    def test_start_counts_end_does_not(self):
        r = l2.rain_between(days(JAN1, [1.0, 2.0, 4.0, 8.0]), JAN1, date(2026, 1, 4))
        self.assertEqual((r["rain_mm"], r["days"], r["missing_days"]), (7.0, 3, 0))

    def test_back_to_back_intervals_add_up(self):
        rain = days(JAN1, [float(i) for i in range(60)])
        a, b, c = JAN1, date(2026, 1, 20), date(2026, 2, 15)
        self.assertEqual(l2.rain_between(rain, a, b)["rain_mm"] + l2.rain_between(rain, b, c)["rain_mm"],
                         l2.rain_between(rain, a, c)["rain_mm"])

    def test_a_missing_day_gives_no_total(self):
        r = l2.rain_between(days(JAN1, [1.0, None, 4.0]), JAN1, date(2026, 1, 5))  # 2 Jan None, 4 Jan absent
        self.assertEqual((r["rain_mm"], r["missing_days"]), (None, 2))

    def test_same_day_scans_have_no_rain_between(self):
        self.assertEqual(l2.rain_between({}, JAN1, JAN1)["rain_mm"], 0.0)

    def test_end_before_start_is_refused(self):
        with self.assertRaises(ValueError):
            l2.rain_between({}, date(2026, 1, 5), JAN1)


class P30AtScan(unittest.TestCase):
    SCAN = date(2026, 3, 1)

    def test_scan_day_and_later_rain_are_left_out(self):
        rain = days(date(2026, 1, 30), [1.0] * 30) | {self.SCAN: 500.0, date(2026, 3, 2): 500.0}
        p30 = l2.p30_at({"weatherkit-history": rain}, self.SCAN)
        self.assertEqual((p30["p30_mm"], p30["p30_end_date"], p30["water_stress"]), (30.0, date(2026, 2, 28), True))

    def test_same_as_the_job_on_the_day_it_runs(self):
        sources = {"weatherkit-history": days(date(2026, 1, 20), [2.5] * 40), "nasa-power": days(JAN1, [3.0] * 50)}
        today = date(2026, 3, 1)
        cut = {k: {d: v for d, v in s.items() if d < today} for k, s in sources.items()}
        self.assertEqual(l2.p30_at(sources, today), rules.choose_p30(cut, today))

    def test_follows_the_l5_order_and_age_limit(self):
        wk_stale = days(date(2026, 1, 1), [2.0] * 40)    # ends 9 Feb: 20 days before the scan
        nasa = days(date(2026, 1, 20), [1.0] * 40)         # ends 28 Feb
        p30 = l2.p30_at({"weatherkit-history": wk_stale, "nasa-power": nasa}, self.SCAN)
        self.assertEqual((p30["p30_source"], p30["p30_quality"], p30["p30_mm"]), ("nasa-power", "fallback", 30.0))
        none = l2.p30_at({"weatherkit-history": wk_stale}, self.SCAN)
        self.assertEqual((none["p30_quality"], none["water_stress"]), ("none", None))


class ScanRainLog(unittest.TestCase):
    def test_log_per_scan_oldest_first(self):
        wk = days(JAN1, [2.0] * 120)
        log = l2.scan_rain_log({"weatherkit-history": wk}, [date(2026, 4, 1), date(2026, 2, 1), date(2026, 2, 1)])
        self.assertEqual([e["scan_date"] for e in log], [date(2026, 2, 1), date(2026, 4, 1)])
        self.assertIsNone(log[0]["since_previous"])
        between = log[1]["since_previous"]
        self.assertEqual((between["rain_mm"], between["days"], between["source"]), (118.0, 59, "weatherkit-history"))
        self.assertEqual(log[1]["p30"]["p30_mm"], 60.0)

    def test_an_interval_never_mixes_sources(self):
        wk = days(JAN1, [2.0] * 120)
        wk[date(2026, 3, 10)] = None  # one gap in WeatherKit: the whole interval comes from NASA POWER
        nasa = days(JAN1, [3.0] * 120)
        between = l2.scan_rain_log({"weatherkit-history": wk, "nasa-power": nasa},
                                   [date(2026, 2, 1), date(2026, 4, 1)])[1]["since_previous"]
        self.assertEqual((between["source"], between["rain_mm"]), ("nasa-power", 177.0))

    def test_no_full_source_is_unknown_not_zero(self):
        between = l2.scan_rain_log({"weatherkit-history": days(JAN1, [2.0] * 10)},
                                   [date(2026, 2, 1), date(2026, 4, 1)])[1]["since_previous"]
        self.assertEqual((between["rain_mm"], between["source"]), (None, None))


if __name__ == "__main__":
    unittest.main()
