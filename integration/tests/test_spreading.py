"""The spreading plan on made-up weather (no network): day assignment, waits, the calendar and the round totals."""
import unittest
from datetime import date, timedelta

from krops_vari.config import Settings
from krops_vari.spreading import assign_days, round_months, spreading_plan, wait_reason

TODAY = date(2026, 10, 26)


def l8(state="jendela_tabur", days=10, avg=210.0):
    window = [TODAY + timedelta(days=i) for i in range(days)] if state == "jendela_tabur" else []
    return {"l8_state": state, "month_gate": {"month": 10, "avg_mm": avg, "gate": "open"}, "dates": window,
            "removed": {"heavy": [], "day_before_heavy": [], "last_cached": []}}


def stage_a(closed=(12,), shares=None):
    shares = shares or {}
    return {m: {"avg_mm": 150.0, "in_band_share": shares.get(m, 0.5), "gate": "tunda" if m in closed else "open"}
            for m in range(1, 13)}


PRODUCTS = [
    {"zone_id": 1, "product": "kieserite_Mg_expensive", "nutrient": "Mg", "product_kg_per_zone_per_year": 2.0,
     "product_kg_per_palm_per_year": 0.2},
    {"zone_id": 1, "product": "dolomite_Mg_cheaper", "nutrient": "Mg", "product_kg_per_zone_per_year": 3.0,
     "product_kg_per_palm_per_year": 0.3},
    {"zone_id": 1, "product": "urea_N_expensive", "nutrient": "N", "product_kg_per_zone_per_year": 4.0,
     "product_kg_per_palm_per_year": 0.4},
]


class Days(unittest.TestCase):
    def test_assign_days(self):
        d = [TODAY + timedelta(days=i) for i in range(10)]
        self.assertEqual(assign_days(d, 1), (d[0], d[1]))
        self.assertEqual(assign_days(d, 14), (d[0], None))
        self.assertEqual(assign_days([d[0], d[5]], 3), (d[0], d[5]))
        self.assertEqual(assign_days(d[:1], 1), (d[0], None))
        self.assertEqual(assign_days([], 1), (None, None))

    def test_wait_reasons(self):
        self.assertIn("tunda", wait_reason(l8("tunda"), None, TODAY))
        self.assertIn("no safe day", wait_reason(l8("cek_hujan"), None, TODAY))
        self.assertIn("P30 40.0 mm", wait_reason(l8(), {"water_stress": True, "p30_mm": 40.0}, TODAY))
        self.assertIsNone(wait_reason(l8(), {"water_stress": False, "p30_mm": 115.0}, TODAY))
        self.assertIsNone(wait_reason(l8(), {"water_stress": None, "p30_mm": None}, TODAY))  # unknown is not stress


class Plan(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def test_n_after_kieserite_next_day_after_dolomite_not_before_14_days(self):
        plan = spreading_plan(PRODUCTS, stage_a(), l8(), {"water_stress": False}, TODAY, self.s)
        urea = next(r for r in plan["this_round"] if r["product"] == "urea_N_expensive")
        self.assertEqual(urea["schedule_if_mg_is"]["kieserite_Mg_expensive"]["date"], "2026-10-27")
        dolo = urea["schedule_if_mg_is"]["dolomite_Mg_cheaper"]
        self.assertIsNone(dolo["date"])
        self.assertEqual(dolo["not_before"], "2026-11-09")
        kies = next(r for r in plan["this_round"] if r["product"] == "kieserite_Mg_expensive")
        self.assertEqual(list(kies["schedule_if_mg_is"]), ["kieserite_Mg_expensive"])
        self.assertEqual(kies["schedule_if_mg_is"]["kieserite_Mg_expensive"]["date"], "2026-10-26")
        self.assertAlmostEqual(kies["kg_per_zone"], 1.0)

    def test_wait_replaces_every_date(self):
        plan = spreading_plan(PRODUCTS, stage_a(), l8("cek_hujan"), None, TODAY, self.s)
        for row in plan["this_round"]:
            for s in row["schedule_if_mg_is"].values():
                self.assertIsNone(s["date"])
                self.assertTrue(s["wait"].startswith("wait"))

    def test_calendar_picks_the_open_pair_with_most_in_band_share(self):
        months, pairs = round_months(stage_a(closed=(12,), shares={3: 0.9, 9: 0.9}), self.s)
        self.assertEqual(months, (3, 9))
        self.assertNotIn((6, 12), [p for p, _ in pairs])  # December is closed

    def test_no_open_pair_is_flagged(self):
        plan = spreading_plan(PRODUCTS, stage_a(closed=tuple(range(1, 13))), l8(), None, TODAY, self.s)
        self.assertIsNone(plan["calendar"]["round_months"])
        self.assertTrue(plan["flags"])


class RoundsPerYear(unittest.TestCase):
    """The user picks the rounds a year; the months are 12 ÷ rounds apart and the dose per round is annual ÷ rounds."""

    def test_three_rounds_four_months_apart(self):
        s = Settings(rounds_per_year=3)
        self.assertEqual(s.round_gap_months, 4)
        months, sets = round_months(stage_a(closed=(12,), shares={2: 0.9, 6: 0.9, 10: 0.9}), s)
        self.assertEqual(months, (2, 6, 10))
        self.assertTrue(all(len(m) == 3 for m, _ in sets))
        self.assertNotIn((4, 8, 12), [m for m, _ in sets])  # December is closed

    def test_dose_per_round_and_the_calendar_follow_the_rounds(self):
        plan = spreading_plan(PRODUCTS, stage_a(closed=()), l8(), None, TODAY, Settings(rounds_per_year=4))
        kies = next(r for r in plan["this_round"] if r["product"] == "kieserite_Mg_expensive")
        self.assertAlmostEqual(kies["kg_per_zone"], 2.0 / 4)
        self.assertEqual(len(plan["calendar"]["round_months"]), 4)
        self.assertEqual((plan["rounds_per_year"], plan["round_gap_months"]), (4, 3))

    def test_one_and_six_rounds(self):
        self.assertEqual(round_months(stage_a(closed=(), shares={7: 0.9}), Settings(rounds_per_year=1))[0], (7,))
        self.assertEqual(round_months(stage_a(closed=()), Settings(rounds_per_year=6))[0], (1, 3, 5, 7, 9, 11))

    def test_too_few_open_months_is_flagged(self):
        plan = spreading_plan(PRODUCTS, stage_a(closed=(1, 5, 9, 2, 6, 10, 3, 7, 11)), l8(), None, TODAY,
                              Settings(rounds_per_year=3))  # only Apr, Aug and Dec open: exactly one set
        self.assertEqual(plan["calendar"]["round_months"], ["Apr", "Aug", "Dec"])
        plan = spreading_plan(PRODUCTS, stage_a(closed=(1, 5, 9, 2, 6, 10, 3, 7, 11, 12)), l8(), None, TODAY,
                              Settings(rounds_per_year=3))
        self.assertIsNone(plan["calendar"]["round_months"])
        self.assertIn("no 3 open month(s) 4 months apart", plan["flags"][0])

    def test_only_whole_month_gaps_are_allowed(self):
        for bad in (0, 5, 7, 12, True, "3"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Settings(rounds_per_year=bad)


if __name__ == "__main__":
    unittest.main()
