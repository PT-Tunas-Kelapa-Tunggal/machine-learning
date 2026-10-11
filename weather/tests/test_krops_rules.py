"""Runs every case in rule_cases.json against krops_rules.py.

    cd weather && python -m unittest discover -s tests -v
"""
import json
import math
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
    """{start, values} (or a list of them) → {date: mm or None}."""
    if enc is None:
        return None
    daily = {}
    for seg in enc if isinstance(enc, list) else [enc]:
        start = to_date(seg["start"])
        for i, v in enumerate(seg["values"]):
            daily[start + timedelta(days=i)] = v
    return daily


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
        return rules.l8_decision(to_daily(x["forecast"]), to_table(x["table"]), to_date(x["today"]))
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
        public = {"compute_p30", "choose_p30", "monthly_normals", "suspect_months", "drop_months",
                  "month_table", "month_gate", "stage_b_day_pick", "l8_decision"}
        self.assertEqual(public - covered, set())


if __name__ == "__main__":
    unittest.main()
