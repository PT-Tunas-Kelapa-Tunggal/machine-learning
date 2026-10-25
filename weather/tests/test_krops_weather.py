"""Tests for the notebook plumbing in krops_weather.py that touches P30. Skipped without the notebook libraries.

    cd weather && python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import pandas as pd

    import krops_weather as kw
except ImportError:  # pandas, matplotlib, jwt or dotenv missing: the job and rule tests still run
    kw = None


@unittest.skipUnless(kw, "notebook libraries not installed (pip install -r requirements.txt)")
class RainFileDuplicates(unittest.TestCase):
    def load(self, extra_rows):
        rows = ["date,rain_mm"] + [f"{date(2026, 9, 1) + timedelta(days=i)},2.0" for i in range(30)] + extra_rows
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gauge.csv"
            path.write_text("\n".join(rows) + "\n")
            return kw.load_rain_file(path)

    def test_a_date_listed_twice_with_the_same_reading_counts_once(self):
        rain = self.load(["2026-09-10,2.0"])
        self.assertEqual((len(rain), rain[date(2026, 9, 10)]), (30, 2.0))

    def test_a_date_listed_twice_with_different_readings_is_missing(self):
        rain = self.load(["2026-09-20,7.5"])
        self.assertNotIn(date(2026, 9, 20), rain.index)
        rolling = rain.copy()
        rolling.index = pd.to_datetime(rolling.index)
        self.assertTrue(pd.isna(rolling.rolling("30D", min_periods=30).sum().iloc[-1]))  # no P30, not a full window


if __name__ == "__main__":
    unittest.main()
