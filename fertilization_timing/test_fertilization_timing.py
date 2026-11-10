"""Tests for the Jadwal Pemupukan schedule. No network: the climate is a fixed Stage A table.

    python -m unittest discover -s fertilization_timing
"""
import json
import unittest
from datetime import date, timedelta
from pathlib import Path

import fertilization_timing as ft

EXAMPLE = json.loads((Path(__file__).parent / "example_request.json").read_text(encoding="utf-8"))

# The Tanjung Pinang cell's 1991–2020 Stage A table (integration/data/weather_cache, computed 2026-10-26).
STAGE_A = {m: {"avg_mm": avg, "in_band_share": share, "gate": gate} for m, avg, share, gate in [
    (1, 215.7, 0.4, "open"), (2, 107.9, 0.367, "open"), (3, 171.5, 0.367, "open"), (4, 205.4, 0.433, "open"),
    (5, 207.1, 0.4, "open"), (6, 174.5, 0.7, "open"), (7, 172.5, 0.433, "open"), (8, 167.9, 0.567, "open"),
    (9, 162.9, 0.6, "open"), (10, 209.7, 0.333, "open"), (11, 298.8, 0.1, "open"), (12, 318.7, 0.1, "tunda"),
]}


# Dolomit → NPK, the two products most tests plan with
TWO = [{"name": "Dolomite", "bagKg": 50, "id": "A01C2E47-5D6B-4B95-9D31-6A1C8217D045", "overallDoseWeightInKg": 22.0},
       {"name": "NPK", "composition": {"N": 15, "P": 15, "K": 15, "Mg": 0}, "bagKg": 50,
        "id": "D828E19B-7B72-4E11-A35D-B1B7D2F9C304", "overallDoseWeightInKg": 24.0}]


def request(freq=3, fertilizers=None):
    return {"latitude": 0.92, "longitude": 104.45, "fertilizerFreq": freq,
            "fertilizers": TWO if fertilizers is None else fertilizers}


BASIC = request()


class RoundMonthsTest(unittest.TestCase):
    def test_three_rounds_tanjung_pinang(self):
        # (1, 5, 9) and (2, 6, 10) both share 1.4: the tie goes to the earlier start, as on the screen
        chosen, open_sets = ft.round_months(STAGE_A, 3)
        self.assertEqual(chosen, (1, 5, 9))
        self.assertNotIn((4, 8, 12), [s for s, _ in open_sets])  # December is tunda

    def test_two_rounds_six_months_apart(self):
        chosen, _ = ft.round_months(STAGE_A, 2)
        self.assertEqual(chosen[1] - chosen[0], 6)

    def test_no_open_set(self):
        closed = {m: {**row, "gate": "tunda"} for m, row in STAGE_A.items()}
        schedule = ft.build_schedule(request(), closed, date(2026, 11, 8))
        self.assertEqual(schedule["rounds"], [])
        self.assertTrue(any("needs a decision" in f for f in schedule["flags"]))

    def test_freq_five_is_refused(self):
        with self.assertRaises(ValueError):
            ft.build_schedule(request(freq=5), STAGE_A, date(2026, 11, 8))


class StepsTest(unittest.TestCase):
    def test_dolomite_then_npk(self):
        steps, flags = ft.fertilizer_steps(TWO, 2)
        self.assertEqual([s["display_name"] for s in steps], ["Dolomit", "NPK 15-15-15"])
        self.assertEqual([s["annual_dose_kg"] for s in steps], [22.0, 24.0])
        self.assertEqual([s["dose_kg"] for s in steps], [11.0, 12.0])  # the year's kg ÷ 2 rounds
        self.assertEqual(steps[0]["wait_after_days"], {"min": 14, "max": 28})
        self.assertIsNone(steps[1]["wait_after_days"])
        self.assertEqual(flags, [])

    def test_order_follows_the_rule_not_the_input(self):
        steps, _ = ft.fertilizer_steps([
            {"name": "Urea", "overallDoseWeightInKg": 10},
            {"name": "NPK", "composition": {"N": 12, "P": 6, "K": 22, "Mg": 3}, "overallDoseWeightInKg": 20},
            {"name": "Dolomite", "overallDoseWeightInKg": 30},
        ], 1)
        self.assertEqual([s["display_name"] for s in steps], ["Dolomit", "NPK 12-6-22-3", "Urea"])
        self.assertEqual(steps[1]["wait_after_days"], {"min": 1, "max": None})  # urea: any later day

    def test_waits_before_and_after_the_straights(self):
        steps, _ = ft.fertilizer_steps([
            {"name": "NPK", "composition": {"N": 15, "P": 15, "K": 15, "Mg": 0}, "overallDoseWeightInKg": 20},
            {"name": "KCl", "overallDoseWeightInKg": 10},
        ], 1)
        self.assertEqual(steps[0]["wait_after_days"], {"min": 1, "max": 7})  # no N straight next: design GUESS
        steps, _ = ft.fertilizer_steps([{"name": "Urea", "overallDoseWeightInKg": 10},
                                        {"name": "Dolomit", "overallDoseWeightInKg": 30}], 1)
        self.assertEqual(steps[0]["wait_after_days"], {"min": 14, "max": 28})  # the lime's wait holds before N

    def test_example_request(self):
        steps, flags = ft.fertilizer_steps(EXAMPLE["fertilizers"], EXAMPLE["fertilizerFreq"])
        self.assertEqual([s["display_name"] for s in steps],
                         ["Dolomit", "NPK 15-15-15", "NPK 12-6-22-3", "NPK 13-6-27-4", "NPK 13-8-27-4",
                          "NPK 16-16-16", "Urea"])
        for step, entry in zip(steps, EXAMPLE["fertilizers"]):  # the request is already in spreading order
            self.assertAlmostEqual(step["dose_kg"], entry["overallDoseWeightInKg"] / 2)
        self.assertEqual(steps[0]["wait_after_days"], {"min": 14, "max": 28})
        self.assertEqual(steps[1]["wait_after_days"], {"min": 1, "max": 7})
        self.assertEqual(steps[5]["wait_after_days"], {"min": 1, "max": None})  # the last NPK, before urea
        self.assertEqual(len(flags), 5)  # the five products with no id
        self.assertTrue(all("no id" in f for f in flags))

    def test_unknown_product_is_flagged(self):
        steps, flags = ft.fertilizer_steps([{"name": "Pupuk Organik", "id": "org", "overallDoseWeightInKg": 5},
                                            {"name": "Urea", "id": "urea", "overallDoseWeightInKg": 10}], 1)
        self.assertEqual([s["display_name"] for s in steps], ["Pupuk Organik", "Urea"])
        self.assertEqual(len(flags), 1)


class ScheduleTest(unittest.TestCase):
    def test_year_plan(self):
        schedule = ft.build_schedule(BASIC, STAGE_A, date(2026, 11, 8))
        self.assertEqual(schedule["plan_year"], 2027)
        rounds = schedule["rounds"]
        self.assertEqual([(r["start_month_name"], r["end_month_name"]) for r in rounds],
                         [("Januari", "Februari"), ("Mei", "Juni"), ("September", "Oktober")])
        self.assertEqual(rounds[0]["total_days"], 59)
        self.assertEqual(rounds[0]["status"], "upcoming")
        self.assertEqual(rounds[0]["days_until_start"], 54)
        for r in rounds:  # dolomite in every round, each round a third of the year's kg
            dolomite, npk = (s["dose_kg"] for s in r["steps"])
            self.assertAlmostEqual(dolomite, 22.0 / 3)
            self.assertAlmostEqual(npk, 8.0)
        self.assertNotIn("lat", json.dumps(schedule))
        self.assertEqual(schedule["flags"], [])

    def test_ongoing_round_shows_day_of_window(self):
        schedule = ft.build_schedule(BASIC, STAGE_A, date(2027, 1, 5), year=2027)
        first = schedule["rounds"][0]
        self.assertEqual((first["status"], first["day_of_round"], first["total_days"]), ("ongoing", 5, 59))
        self.assertEqual(schedule["rounds"][1]["status"], "upcoming")

    def test_done_round(self):
        schedule = ft.build_schedule(BASIC, STAGE_A, date(2027, 3, 1), year=2027)
        self.assertEqual(schedule["rounds"][0]["status"], "done")

    def test_window_into_a_closed_month_is_flagged(self):
        # only November is open: its window runs on into December, which isn't
        only_nov = {m: {**row, "gate": "open" if m == 11 else "cek"} for m, row in STAGE_A.items()}
        schedule = ft.build_schedule(request(freq=1), only_nov, date(2026, 11, 8))
        self.assertEqual(schedule["rounds"][0]["end_date"], "2026-12-31")  # this year's round hasn't ended
        self.assertTrue(any("Desember is not an open month" in f for f in schedule["flags"]))


DOLOMITE_ID = TWO[0]["id"]
P30_OK = {"p30_mm": 110.8, "p30_source": "weatherkit-history", "p30_quality": "ok", "water_stress": False}


def forecast_from(first, rain_mm):
    """{date: mm} for consecutive days from `first`, like WeatherClient.forecast."""
    return {first + timedelta(days=i): mm for i, mm in enumerate(rain_mm)}


def weather(forecast, p30=P30_OK, flags=()):
    return lambda: (forecast, p30, list(flags))


class DailyTest(unittest.TestCase):
    def test_heavy_rain_and_the_day_before_are_skipped(self):
        today = date(2027, 1, 5)
        rain = [2, 2, 32, 2, 2, 2, 2, 2, 2, 2]  # 7 Jan heavy → 6 and 7 go, and 14 Jan (the last day)
        schedule = ft.build_schedule(BASIC, STAGE_A, today, daily_weather=weather(forecast_from(today, rain)))
        first = schedule["rounds"][0]
        self.assertEqual(first["daily"]["l8"]["state"], "jendela_tabur")
        self.assertEqual(first["daily"]["l8"]["safe_days"][:2], ["2027-01-05", "2027-01-08"])
        dolomite, npk = (s["schedule"] for s in first["steps"])
        self.assertEqual((dolomite["status"], dolomite["date"]), ("planned", "2027-01-05"))
        # 14 days after dolomite is past the 10-day forecast: a "not before", to run again then
        self.assertEqual((npk["status"], npk["not_before"]), ("not_before", "2027-01-19"))

    def test_spread_step_sets_the_next_date(self):
        today = date(2027, 1, 12)
        request = {**BASIC, "applications": [{"fertilizerId": DOLOMITE_ID, "date": "2027-01-02"}]}
        schedule = ft.build_schedule(request, STAGE_A, today, daily_weather=weather(forecast_from(today, [1] * 10)))
        dolomite, npk = (s["schedule"] for s in schedule["rounds"][0]["steps"])
        self.assertEqual((dolomite["status"], dolomite["date"]), ("done", "2027-01-02"))
        self.assertEqual((npk["status"], npk["date"]), ("planned", "2027-01-16"))

    def test_urea_any_later_safe_day_after_npk(self):
        request = request_with([{"name": "NPK", "composition": {"N": 15, "P": 15, "K": 15, "Mg": 0},
                                 "id": "npk", "overallDoseWeightInKg": 30},
                                {"name": "Urea", "id": "urea", "overallDoseWeightInKg": 30}])
        today = date(2027, 1, 5)
        schedule = ft.build_schedule(request, STAGE_A, today, daily_weather=weather(forecast_from(today, [1] * 10)))
        npk, urea = (s["schedule"]["date"] for s in schedule["rounds"][0]["steps"])
        self.assertEqual((npk, urea), ("2027-01-05", "2027-01-06"))

    def test_water_stress_waits(self):
        today = date(2027, 1, 5)
        dry = {**P30_OK, "p30_mm": 40.0, "water_stress": True}
        schedule = ft.build_schedule(BASIC, STAGE_A, today,
                                     daily_weather=weather(forecast_from(today, [1] * 10), dry))
        first = schedule["rounds"][0]
        self.assertIn("wait for rain", first["daily"]["wait"])
        self.assertEqual({s["schedule"]["status"] for s in first["steps"]}, {"wait"})

    def test_closed_month_waits(self):
        # only November is open: on 3 Dec the round is under way, but December averages 318.7 mm (L8 judges the
        # month from its average: over 300 mm is tunda)
        only_nov = {m: {**row, "gate": "open" if m == 11 else "cek"} for m, row in STAGE_A.items()}
        today = date(2026, 12, 3)
        schedule = ft.build_schedule(request(freq=1), only_nov, today,
                                     daily_weather=weather(forecast_from(today, [1] * 10)))
        first = schedule["rounds"][0]
        self.assertEqual(first["daily"]["l8"]["state"], "tunda")
        self.assertEqual(first["steps"][0]["schedule"]["status"], "wait")

    def test_no_forecast_waits_and_flags(self):
        today = date(2027, 1, 5)
        schedule = ft.build_schedule(BASIC, STAGE_A, today, daily_weather=weather({}, None, ["no keys"]))
        self.assertEqual(schedule["rounds"][0]["daily"]["l8"]["state"], "cek_hujan")
        self.assertIn("no keys", schedule["flags"])
        self.assertTrue(any("no 10-day forecast" in f for f in schedule["flags"]))

    def test_no_active_round_fetches_nothing(self):
        def fail():
            raise AssertionError("weather fetched with no active round")

        schedule = ft.build_schedule(BASIC, STAGE_A, date(2026, 11, 8), daily_weather=fail)
        self.assertEqual({r["daily"] is None for r in schedule["rounds"]}, {True})
        self.assertNotIn("schedule", schedule["rounds"][0]["steps"][0])

    def test_round_starting_within_the_forecast_is_active(self):
        today = date(2026, 12, 25)  # 7 days before 1 January
        schedule = ft.build_schedule(BASIC, STAGE_A, today, daily_weather=weather(forecast_from(today, [1] * 10)))
        first = schedule["rounds"][0]
        self.assertEqual(first["daily"]["reference_day"], "2027-01-01")  # December is tunda, January is judged
        self.assertEqual(first["steps"][0]["schedule"]["date"], "2027-01-01")

    def test_default_plan_year(self):
        self.assertEqual(ft.build_schedule(BASIC, STAGE_A, date(2026, 11, 8))["plan_year"], 2027)
        self.assertEqual(ft.build_schedule(BASIC, STAGE_A, date(2027, 1, 20))["plan_year"], 2027)

    def test_application_before_the_first_round_is_flagged(self):
        only_nov = {m: {**row, "gate": "open" if m == 11 else "cek"} for m, row in STAGE_A.items()}
        marked = {**request(freq=1), "applications": [{"fertilizerId": DOLOMITE_ID, "date": "2026-03-10"}]}
        schedule = ft.build_schedule(marked, only_nov, date(2026, 11, 8))
        self.assertTrue(any("before the year's first round" in f for f in schedule["flags"]))


def request_with(fertilizers, freq=3, applications=None):
    return {**request(freq, fertilizers),
            "applications": [{"fertilizerId": i, "date": d} for i, d in (applications or {}).items()]}


# The card in the design: Dolomit → (14–28 days) → NPK → (any later day) → Urea, round 1 = January–February 2027
THREE = [{"name": "Dolomite", "id": "dol", "overallDoseWeightInKg": 1200},
         {"name": "NPK", "composition": {"N": 15, "P": 15, "K": 15, "Mg": 0}, "id": "npk",
          "overallDoseWeightInKg": 360},
         {"name": "Urea", "id": "urea", "overallDoseWeightInKg": 300}]


def first_card(today, applications=None, daily_weather=None):
    """(header, {fertilizer id: card}, [wait after each step]) of round 1."""
    schedule = ft.build_schedule(request_with(THREE, applications=applications), STAGE_A, today,
                                 daily_weather=daily_weather)
    first = schedule["rounds"][0]
    return (first["card"], {s["fertilizer_id"]: s["card"] for s in first["steps"]},
            [s["wait_after_days"] for s in first["steps"]])


class CardTest(unittest.TestCase):
    def test_upcoming_image_4(self):
        header, cards, waits = first_card(date(2026, 11, 8))
        self.assertEqual(header, {"state": "upcoming", "days_until_start": 54})
        self.assertEqual({c["state"] for c in cards.values()}, {"locked"})
        self.assertEqual(waits[0], {"min": 14, "max": 28, "state": "idle", "until": None})
        self.assertIsNone(waits[2])

    def test_ongoing_first_step_ready_image_5(self):
        header, cards, _ = first_card(date(2027, 1, 5))
        self.assertEqual(header, {"state": "ongoing", "day_of_round": 5, "total_days": 59})
        self.assertEqual(cards["dol"], {"state": "ready", "available_from": "2027-01-01", "deadline": "2027-02-28"})
        self.assertEqual((cards["npk"]["state"], cards["urea"]["state"]), ("locked", "locked"))

    def test_waiting_after_dolomite_image_6(self):
        header, cards, waits = first_card(date(2027, 1, 5), {"dol": "2027-01-05"})
        self.assertEqual(header, {"state": "started", "started_on": "2027-01-05"})
        self.assertEqual(cards["dol"], {"state": "done", "done_on": "2027-01-05"})
        self.assertEqual((waits[0]["state"], waits[0]["until"]), ("counting", "2027-01-19"))  # "· sampai 19 Januari"
        self.assertEqual(cards["npk"], {"state": "waiting", "available_from": "2027-01-19"})

    def test_npk_ready_image_7(self):
        _, cards, waits = first_card(date(2027, 1, 19), {"dol": "2027-01-05"})
        self.assertEqual(waits[0]["state"], "done")  # "· selesai"
        # overdue after dolomite + 28 days
        self.assertEqual(cards["npk"], {"state": "ready", "available_from": "2027-01-19", "deadline": "2027-02-02"})

    def test_urea_waiting_image_8(self):
        _, cards, waits = first_card(date(2027, 1, 19), {"dol": "2027-01-05", "npk": "2027-01-19"})
        self.assertEqual((waits[1]["state"], waits[1]["until"]), ("counting", "2027-01-20"))  # "· mulai 20 Januari"
        self.assertEqual(cards["urea"], {"state": "waiting", "available_from": "2027-01-20"})

    def test_urea_ready_image_9(self):
        _, cards, waits = first_card(date(2027, 1, 20), {"dol": "2027-01-05", "npk": "2027-01-19"})
        self.assertEqual(waits[1]["state"], "done")
        # no maximum before urea: it is due by the end of the round
        self.assertEqual(cards["urea"], {"state": "ready", "available_from": "2027-01-20", "deadline": "2027-02-28"})

    def test_period_passed_image_10(self):
        header, cards, waits = first_card(date(2027, 3, 1), {"dol": "2027-01-05", "npk": "2027-01-19"})
        self.assertEqual(header, {"state": "period_passed", "period_end": "2027-02-28"})
        self.assertEqual(cards["urea"], {"state": "overdue", "overdue_since": "2027-02-28"})
        self.assertEqual({w["state"] for w in waits if w}, {"idle"})  # plain "Tunggu …" again

    def test_npk_overdue_while_the_round_runs(self):
        header, cards, _ = first_card(date(2027, 2, 3), {"dol": "2027-01-05"})
        self.assertEqual(header["state"], "started")
        self.assertEqual(cards["npk"], {"state": "overdue", "overdue_since": "2027-02-02"})
        self.assertEqual(cards["urea"]["state"], "locked")

    def test_first_step_overdue_when_the_round_ends(self):
        header, cards, _ = first_card(date(2027, 3, 1))
        self.assertEqual(header["state"], "period_passed")
        self.assertEqual(cards["dol"], {"state": "overdue", "overdue_since": "2027-02-28"})
        self.assertEqual(cards["npk"]["state"], "locked")

    def test_completed_card_needs_no_weather(self):
        def fail():
            raise AssertionError("weather fetched for a completed round")

        header, cards, waits = first_card(date(2027, 1, 25), {"dol": "2027-01-05", "npk": "2027-01-19",
                                                              "urea": "2027-01-22"}, daily_weather=fail)
        self.assertEqual(header, {"state": "completed", "completed_on": "2027-01-22"})  # "Selesai 22 Januari 2027"
        self.assertEqual({w["state"] for w in waits if w}, {"idle"})

    def test_step_with_no_id_does_not_block_completed(self):
        fertilizers = THREE[:2] + [{**THREE[2], "id": None}]
        schedule = ft.build_schedule(request_with(fertilizers, applications={"dol": "2027-01-05",
                                                                             "npk": "2027-01-19"}),
                                     STAGE_A, date(2027, 1, 25))
        first = schedule["rounds"][0]
        self.assertEqual(first["card"], {"state": "completed", "completed_on": "2027-01-19"})
        self.assertEqual(first["steps"][2]["card"]["state"], "ready")  # urea is shown, just can't be marked
        self.assertIn("Urea: no id, so it can't be marked as spread", schedule["flags"])

    def test_marked_after_the_window_still_counts(self):
        # "Tandai Sudah Dipupuk" on 2 March: after the window, before round 2 (May)
        header, cards, _ = first_card(date(2027, 3, 2), {"dol": "2027-01-05", "npk": "2027-01-19",
                                                         "urea": "2027-03-02"})
        self.assertEqual(header, {"state": "completed", "completed_on": "2027-03-02"})
        self.assertEqual(cards["urea"], {"state": "done", "done_on": "2027-03-02"})


if __name__ == "__main__":
    unittest.main()
