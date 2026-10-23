"""The no-coordinates rule on the real outputs: every file in outputs/s2_feasibility/ is scanned.

    cd s2-feasibility && python -m unittest discover -s tests -v

Skipped when there are no outputs yet. The .env values are read here, never printed: a failure names the file and
the kind of match only.
"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "s2-feasibility"))
import s2_rules as rules  # noqa: E402

OUT_DIR = ROOT / "outputs" / "s2_feasibility"


def env_values():
    """LON as written in weather/.env, and LAT when it has more than 2 decimals (2 can't be told from an index)."""
    env = ROOT / "weather" / ".env"
    if not env.is_file():
        return []
    values = dict(re.findall(r"^(LAT|LON)\s*=\s*(\S+)", env.read_text(), re.M))
    known = [values.get("LON", "")]
    if len(values.get("LAT", "").split(".")[-1]) > 2:
        known.append(values["LAT"])
    return [k for k in known if k]


@unittest.skipUnless(OUT_DIR.is_dir() and any(OUT_DIR.iterdir()), "no outputs yet (run s2_feasibility.py)")
class OutputsHoldNoLocation(unittest.TestCase):
    def test_every_output_file(self):
        known = env_values()
        files = [p for p in sorted(OUT_DIR.rglob("*")) if p.is_file()]
        self.assertTrue(files)
        for path in files:
            with self.subTest(file=path.name):
                hits = rules.find_coordinates(path.read_text(errors="replace"), known)
                self.assertEqual(hits, [], f"{path.name}: {[h.split(' ')[0] for h in hits]}")


if __name__ == "__main__":
    unittest.main()
