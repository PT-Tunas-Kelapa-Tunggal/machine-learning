"""Runs every case in rule_cases.json against krops_rules.py.

    cd weather && python -m unittest discover -s tests -v
"""
import json
import math
import random
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import krops_rules as rules  # noqa: E402

CASES = json.loads((Path(__file__).with_name("rule_cases.json")).read_text())["cases"]


# ------------------------------------------------------------ decoding the JSON encoding

def to_date(s):
    return date.fromisoformat(s)


def to_month(s):
    year, month = s.split("-")
    return int(year), int(month)


def to_daily(enc):
    """{start, values} (or a list of them) → {date: mm or None}. Overlapping segments give a date twice, and
    rules.daily_from_pairs settles it: once if the values agree, missing if they don't."""
    if enc is None:
        return None
    pairs = []
    for seg in enc if isinstance(enc, list) else [enc]:
        start = to_date(seg["start"])
        pairs += [(start + timedelta(days=i), v) for i, v in enumerate(seg["values"])]
    return rules.daily_from_pairs(pairs)


def to_table(avgs):
    return {m: {"avg_mm": avg} for m, avg in enumerate(avgs, start=1)}


# ------------------------------------------------------------ encoding results back to JSON values

def encode(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def run(case):
    f, x = case["function"], case["input"]
    if f == "daily_from_pairs":
        return rules.daily_from_pairs([(to_date(d), v) for d, v in x["pairs"]])
    if f == "missing_in_p30_window":
        return rules.missing_in_p30_window(to_daily(x["daily"]))
    if f == "compute_p30":
        return rules.compute_p30(to_daily(x["daily"]), x["source"])
    if f == "choose_p30":
        sources = {k: to_daily(v) for k, v in x["sources"].items()}
        order = tuple(x["order"]) if "order" in x else rules.P30_SOURCE_ORDER
        return rules.choose_p30(sources, to_date(x["today"]), order=order)
    if f == "monthly_normals":
        return rules.monthly_normals(to_daily(x["daily"]), x["first"], x["last"])
    if f == "suspect_months":
        normals = {int(k): v for k, v in x["normals"].items()}
        months = rules.suspect_months(to_daily(x["nasa"]), normals, to_daily(x["weatherkit"]))
        return [f"{y}-{m:02d}" for y, m in sorted(months)]
    if f == "unchecked_months":
        return [f"{y}-{m:02d}" for y, m in sorted(rules.unchecked_months(to_daily(x["daily"])))]
    if f == "drop_months":
        return rules.drop_months(to_daily(x["daily"]), {to_month(m) for m in x["months"]})
    if f == "month_table":
        return rules.month_table(to_daily(x["daily"]), x["first"], x["last"])
    if f == "month_gate":
        return rules.month_gate(x["avg_mm"])
    if f == "stage_b_day_pick":
        window, removed = rules.stage_b_day_pick(to_daily(x["forecast"]), to_date(x["today"]))
        return {"window": window, "removed": removed}
    if f == "l8_decision":
        cut = x.get("cut_at_closed_month", rules.CUT_WINDOW_AT_CLOSED_MONTH)
        return rules.l8_decision(to_daily(x["forecast"]), to_table(x["table"]), to_date(x["today"]), cut_at_closed_month=cut)
    raise ValueError(f"unknown function {f}")


def assert_same(test, actual, expected, path="result"):
    if isinstance(expected, float) or (isinstance(expected, int) and isinstance(actual, float)
                                       and not isinstance(expected, bool)):
        test.assertTrue(math.isclose(actual, expected, abs_tol=1e-6), f"{path}: {actual} != {expected}")
    elif isinstance(expected, dict):
        test.assertIsInstance(actual, dict, path)
        test.assertEqual(sorted(actual), sorted(expected), f"{path}: keys differ")
        for k in expected:
            assert_same(test, actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected), f"{path}: {actual} != {expected}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(test, a, e, f"{path}[{i}]")
    else:
        test.assertEqual(actual, expected, path)


class RuleCases(unittest.TestCase):
    def test_cases(self):
        for case in CASES:
            with self.subTest(case["name"]):
                assert_same(self, encode(run(case)), case["expected"])

    def test_every_function_is_covered(self):
        covered = {c["function"] for c in CASES}
        public = {"daily_from_pairs", "missing_in_p30_window", "compute_p30", "choose_p30", "monthly_normals",
                  "suspect_months", "unchecked_months", "drop_months", "month_table", "month_gate", "stage_b_day_pick",
                  "l8_decision"}
        self.assertEqual(public - covered, set())

    def test_the_pending_cut_is_tested_both_ways_and_still_on_by_default(self):
        flags = {c["input"].get("cut_at_closed_month") for c in CASES if c["function"] == "l8_decision"}
        self.assertTrue({True, False} <= flags)
        self.assertIs(rules.CUT_WINDOW_AT_CLOSED_MONTH, True)  # today's behaviour until the team decides


def suspect_months_before(nasa, normals, weatherkit=None, ratio=3.0):
    """The monthly check as it was before pro-rating (frozen copy), for the stricter-only test."""
    wk = rules.monthly_totals(weatherkit) if weatherkit else {}
    out = set()
    for (y, m), total in rules.monthly_totals(nasa).items():
        if total <= ratio * normals[m]:
            continue
        if (y, m) in wk and total <= ratio * wk[(y, m)]:
            continue
        out.add((y, m))
    return out


class MonthlyCheckOnlyGetsStricter(unittest.TestCase):
    def test_no_window_rejected_before_is_accepted_now(self):
        rng, made_stricter = random.Random(2026), 0
        for trial in range(500):
            first = date(2026, rng.randint(1, 12), 1)
            n = rng.randint(1, 75)  # one to three months, the last one often part-filled
            nasa = {first + timedelta(days=i): (None if rng.random() < 0.03 else rng.choice([0.0, rng.uniform(0, 15),
                                                                                              rng.uniform(0, 120)]))
                    for i in range(n)}
            present = {d: v for d, v in nasa.items() if v is not None}
            normals = {m: rng.uniform(40.0, 350.0) for m in range(1, 13)}
            wk = None if rng.random() < 0.3 else {first + timedelta(days=i): rng.uniform(0, 30)
                                                   for i in range(rng.randint(0, n + 5)) if rng.random() > 0.05}
            before = rules.drop_months(present, suspect_months_before(present, normals, wk))
            after = rules.drop_months(present, rules.suspect_months(present, normals, wk) | rules.unchecked_months(present))
            self.assertLessEqual(set(after), set(before), trial)
            for end in present:
                window = [end - timedelta(days=i) for i in range(rules.P30_WINDOW_DAYS)]
                if not all(d in before for d in window):  # rejected before…
                    self.assertFalse(all(d in after for d in window), (trial, end))  # …still rejected
            made_stricter += len(after) < len(before)
        self.assertGreater(made_stricter, 50)  # the scenarios do reach the new tests


if __name__ == "__main__":
    unittest.main()
