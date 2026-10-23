"""Tests for s2_rules.py: thresholds, SCL masking, offset, indices, month aggregation, Uji 1–3 statistics.

    cd s2-feasibility && python -m unittest discover -s tests -v

Without numpy the pixel tests skip; the thresholds, months, offset decision and coordinate scan still run.
"""
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "weather"))
import krops_rules as weather_rules  # noqa: E402
import l2_scan_rain as l2  # noqa: E402
import s2_rules as rules  # noqa: E402

np = rules.np
needs_numpy = unittest.skipUnless(np, "numpy not installed (conda env create -f environment.yml)")


class Thresholds(unittest.TestCase):
    """Each GUESS pinned: changing one must be a deliberate team decision, made here too."""

    def test_clear_share_min(self):
        self.assertEqual(rules.CLEAR_SHARE_MIN, 0.80)

    def test_inner_buffer(self):
        self.assertEqual(rules.INNER_BUFFER_M, 20)

    def test_min_clean_pixels(self):
        self.assertEqual(rules.MIN_CLEAN_PIXELS, 20)

    def test_min_pixels_per_stat(self):
        self.assertEqual(rules.MIN_PIXELS_PER_STAT, 10)

    def test_age_classes(self):
        self.assertEqual(rules.AGE_CLASSES, ((0, 3, "0-3 immature"), (4, 8, "4-8 young"),
                                             (9, 15, "9-15 prime"), (16, 99, "16+ old")))

    def test_offset_dark_share_max(self):
        self.assertEqual((rules.OFFSET_DARK_SHARE_MAX, rules.DARK_EVIDENCE_MIN_PIXELS), (0.05, 500))

    def test_month_rules_and_dedup(self):
        self.assertEqual(rules.MONTH_POOLING, "every clear interior pixel of every scene in the month")
        self.assertEqual(rules.MONTH_WATER_STRESS, "any scene")
        self.assertEqual(rules.DEDUP, "highest processing baseline per acquisition")

    def test_every_guess_is_marked(self):
        source = (Path(__file__).resolve().parents[1] / "s2_rules.py").read_text()
        for name in ("CLEAR_SHARE_MIN", "INNER_BUFFER_M", "MIN_CLEAN_PIXELS", "MIN_PIXELS_PER_STAT", "AGE_CLASSES",
                     "OFFSET_DARK_SHARE_MAX", "DARK_EVIDENCE_MIN_PIXELS", "MONTH_POOLING", "MONTH_WATER_STRESS",
                     "DEDUP"):
            line = next(ln for ln in source.splitlines() if ln.startswith(f"{name} ="))
            self.assertIn("# GUESS – PENDING TEAM DECISION", line, name)

    def test_fixed_values(self):
        self.assertEqual((rules.CLEAR_CLASSES, rules.LOGGED_CLASSES), ((4, 5), (2, 3, 7, 8, 9, 10)))
        self.assertEqual((rules.OFFSET_DN, rules.OFFSET_FROM_BASELINE, rules.PROXY_BUFFER_M), (1000, "04.00", 100))


class Months(unittest.TestCase):
    def test_latest_complete_month(self):
        self.assertEqual(rules.latest_complete_month(date(2026, 10, 23)), (2026, 9))
        self.assertEqual(rules.latest_complete_month(date(2027, 1, 1)), (2026, 12))

    def test_range_crosses_years(self):
        self.assertEqual(rules.months((2019, 11), (2020, 2)), [(2019, 11), (2019, 12), (2020, 1), (2020, 2)])

    def test_uji1_table(self):
        scenes = [{"date": date(2020, 1, 3), "clear_share": 0.79}, {"date": date(2020, 1, 13), "clear_share": 0.80},
                  {"date": date(2020, 2, 3), "clear_share": 0.5}, {"date": date(2020, 2, 8), "clear_share": None}]
        rows = rules.uji1_table(scenes, (2020, 1), (2020, 3))
        self.assertEqual([(r["month"], r["has_clear"], r["scenes_total"], r["scenes_clear"]) for r in rows],
                         [("2020-01", True, 2, 1), ("2020-02", False, 2, 0), ("2020-03", False, 0, 0)])
        self.assertEqual(rows[1]["best_clear_share"], 0.5)
        self.assertIsNone(rows[2]["best_clear_share"])

    def test_month_of_year_and_longest_gap(self):
        flags = [True, False, False, True, False, False, False, True] + [True] * 16
        rows = [{"month": rules.month_label(m), "has_clear": f}
                for m, f in zip(rules.months((2019, 1), (2020, 12)), flags)]
        self.assertEqual(rules.longest_gap(rows), {"months": 3, "first": "2019-05", "last": "2019-07"})
        summary = {s["month_of_year"]: s for s in rules.month_of_year_summary(rows)}
        self.assertEqual((summary[2]["years"], summary[2]["years_with_clear"], summary[2]["share"]), (2, 1, 0.5))
        self.assertEqual(summary[12]["share"], 1.0)

    def test_gap_at_the_end_counts(self):
        rows = [{"month": "2020-01", "has_clear": True}, {"month": "2020-02", "has_clear": False}]
        self.assertEqual(rules.longest_gap(rows)["months"], 1)


class Catalogue(unittest.TestCase):
    def item(self, id_, day, baseline, updated="2024-01-01", tile="T1"):
        return {"id": id_, "date": day, "tile": tile, "baseline": baseline, "updated": updated}

    def test_highest_baseline_wins(self):
        d = date(2020, 5, 1)
        items = [self.item("a", d, "02.14"), self.item("b", d, "05.00"), self.item("c", d, "05.00", tile="T2"),
                 self.item("d", date(2020, 5, 6), "02.14")]
        kept, dropped = rules.dedup_items(items)
        self.assertEqual(sorted(i["id"] for i in kept), ["b", "c", "d"])
        self.assertEqual([i["id"] for i in dropped], ["a"])

    def test_same_baseline_latest_update_wins(self):
        d = date(2024, 10, 16)
        kept, _ = rules.dedup_items([self.item("old", d, "05.11", "2024-10-17"), self.item("new", d, "05.11", "2025-01-02")])
        self.assertEqual([i["id"] for i in kept], ["new"])


class OffsetHarmonization(unittest.TestCase):
    def test_metadata(self):
        self.assertFalse(rules.offset_in_metadata("03.01", False))  # before 04.00: never had it
        self.assertTrue(rules.offset_in_metadata("04.00", False))
        self.assertTrue(rules.offset_in_metadata("05.11", None))     # flag missing: the metadata can't say it was removed
        self.assertFalse(rules.offset_in_metadata("05.11", True))
        self.assertFalse(rules.offset_in_metadata(None, False))

    def test_decision(self):
        self.assertEqual(rules.offset_decision(False, 0.0), ("none", False))       # pixels never add the offset
        self.assertEqual(rules.offset_decision(True, 0.0), ("offset", False))
        self.assertEqual(rules.offset_decision(True, 0.05), ("offset", False))     # at the line: still offset
        self.assertEqual(rules.offset_decision(True, 0.051), ("none", True))       # metadata contradicted
        self.assertEqual(rules.offset_decision(True, None), ("unresolved", False))

    def test_decision_on_the_observed_scenes(self):
        """The dark shares measured on 23 Oct 2026 (peek6): Jan–Feb 2022 keep the offset, Oct 2022 and 05.11 don't."""
        for share in (0.0, 0.021):
            self.assertEqual(rules.offset_decision(True, share)[0], "offset")
        for share in (0.115, 0.309, 0.365, 0.867):
            self.assertEqual(rules.offset_decision(True, share)[0], "none")

    @needs_numpy
    def test_harmonize_one_scale(self):
        dn = np.array([1500, 0, 2000], dtype="uint16")
        with_offset, without = rules.harmonize(dn, True), rules.harmonize(dn, False)
        np.testing.assert_allclose(with_offset[[0, 2]], [0.05, 0.10])
        np.testing.assert_allclose(without[[0, 2]], [0.15, 0.20])
        self.assertTrue(np.isnan(with_offset[1]) and np.isnan(without[1]))  # DN 0 is no data, not −0.1

    @needs_numpy
    def test_same_surface_same_reflectance(self):
        """A surface at reflectance 0.3: DN 3000 without the offset, 4000 with it, read the same after harmonising."""
        a = rules.harmonize(np.array([3000], dtype="uint16"), False)
        b = rules.harmonize(np.array([4000], dtype="uint16"), True)
        np.testing.assert_allclose(a, b)

    @needs_numpy
    def test_dark_share(self):
        scl = np.array([6] * 300 + [4] * 300 + [9] * 400)
        b5 = np.array([300] * 300 + [2500] * 300 + [500] * 400)  # the clouds (SCL 9) are left out
        self.assertAlmostEqual(rules.dark_share(b5, scl), 0.5)
        self.assertIsNone(rules.dark_share(b5[:400], scl[:400]))  # 400 valid pixels < 500


@needs_numpy
class SclMasking(unittest.TestCase):
    def test_counts_inside_interior_only(self):
        scl = np.array([[4, 5, 9], [3, 4, 8], [10, 2, 7]], dtype="uint8")
        interior = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=bool)
        counts = rules.scl_counts(scl, interior)
        self.assertEqual({c: n for c, n in counts.items() if n}, {3: 1, 4: 2, 5: 1, 9: 1})
        self.assertEqual(set(counts), set(range(12)))
        self.assertAlmostEqual(rules.clear_share(counts), 3 / 5)

    def test_water_and_unclassified_are_not_clear(self):
        counts = rules.scl_counts(np.array([4, 4, 4, 6, 7], dtype="uint8"), np.ones(5, dtype=bool))
        self.assertAlmostEqual(rules.clear_share(counts), 0.6)
        self.assertFalse(rules.is_clear_scene(0.6))
        self.assertTrue(rules.is_clear_scene(0.8))

    def test_empty_interior(self):
        counts = rules.scl_counts(np.array([4, 5], dtype="uint8"), np.zeros(2, dtype=bool))
        self.assertIsNone(rules.clear_share(counts))
        self.assertFalse(rules.is_clear_scene(None))

    def test_class_totals_by_name(self):
        totals = rules.class_totals([{c: 1 for c in range(12)}, {c: (2 if c == 9 else 0) for c in range(12)}])
        self.assertEqual((totals["cloud_high"], totals["vegetation"]), (3, 1))


@needs_numpy
class Indices(unittest.TestCase):
    def test_ndre(self):
        np.testing.assert_allclose(rules.ndre(np.array([0.40, 0.30]), np.array([0.10, 0.10])), [0.6, 0.5])

    def test_cire(self):
        np.testing.assert_allclose(rules.cire(np.array([0.35, 0.20]), np.array([0.10, 0.10])), [2.5, 1.0])

    def test_no_data_gives_nan(self):
        self.assertTrue(np.isnan(rules.ndre(np.array([0.0]), np.array([0.0]))[0]))
        self.assertTrue(np.isnan(rules.cire(np.array([0.3]), np.array([0.0]))[0]))
        self.assertTrue(np.isnan(rules.ndre(np.array([np.nan]), np.array([0.1]))[0]))

    def test_offset_matters_for_both(self):
        """The same surface with and without harmonising: the indices differ, so the offset must be removed first."""
        raw = rules.ndre(np.array([4000.0]), np.array([2000.0]))
        fixed = rules.ndre(rules.harmonize(np.array([4000]), True), rules.harmonize(np.array([2000]), True))
        self.assertNotAlmostEqual(float(raw[0]), float(fixed[0]), places=3)


class MonthAggregation(unittest.TestCase):
    @needs_numpy
    def test_pooled_over_scenes(self):
        stats = rules.month_stats({(2024, 1): [np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0.6, 0.7, 0.8, 0.9, 1.0])]})
        s = stats[(2024, 1)]
        self.assertEqual((s["n"], s["median"]), (10, 0.55))
        self.assertAlmostEqual(s["iqr"], 0.45)

    @needs_numpy
    def test_min_pixels_per_stat_boundary(self):
        stats = rules.month_stats({(2024, 1): [np.arange(10.0)], (2024, 2): [np.arange(9.0)],
                                   (2024, 3): [np.array([np.nan] * 5 + list(range(9)))]})
        self.assertEqual(sorted(stats), [(2024, 1)])  # NaNs don't count towards the 10

    def test_month_water_stress(self):
        self.assertTrue(rules.month_water_stress([False, True]))
        self.assertTrue(rules.month_water_stress([None, True]))
        self.assertIsNone(rules.month_water_stress([False, None]))
        self.assertFalse(rules.month_water_stress([False, False]))
        self.assertIsNone(rules.month_water_stress([]))

    def test_water_stress_comes_from_weather(self):
        """Annotation goes through weather's own rule: 30 days of 1 mm is P30 30 mm, under its 60 mm line."""
        scan = date(2024, 3, 1)
        dry = {scan - timedelta(days=i): 1.0 for i in range(1, 31)}
        wet = {scan - timedelta(days=i): 5.0 for i in range(1, 31)}
        dry_p30 = l2.p30_at({"weatherkit-history": dry}, scan)
        wet_p30 = l2.p30_at({"weatherkit-history": wet}, scan)
        self.assertEqual((dry_p30["p30_mm"], wet_p30["p30_mm"]), (30.0, 150.0))
        self.assertEqual(weather_rules.P30_THRESHOLD_MM, 60.0)
        self.assertTrue(rules.month_water_stress([dry_p30["water_stress"], wet_p30["water_stress"]]))
        self.assertFalse(rules.month_water_stress([wet_p30["water_stress"]]))


class AgeSignalAndNoise(unittest.TestCase):
    def test_plot_age(self):
        self.assertEqual(rules.plot_age({"year_planted": 2010}, 2020, 2026), 10)
        self.assertEqual(rules.plot_age({"age_years": 12}, 2020, 2026), 6)  # 12 in 2026 → 6 in 2020
        self.assertIsNone(rules.plot_age({}, 2020, 2026))

    def test_age_class(self):
        self.assertEqual([rules.age_class(a) for a in (0, 3, 4, 8, 9, 15, 16, None)],
                         ["0-3 immature", "0-3 immature", "4-8 young", "4-8 young", "9-15 prime", "9-15 prime",
                          "16+ old", None])

    @needs_numpy
    def test_spearman(self):
        self.assertEqual(rules.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(rules.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(rules.spearman([1, 2, 2, 3], [1, 2, 3, 4]), math.sqrt(0.9), places=4)  # tied ranks
        self.assertIsNone(rules.spearman([1, 2], [1, 2]))
        self.assertIsNone(rules.spearman([5, 5, 5], [1, 2, 3]))

    @needs_numpy
    def test_signal_vs_noise_known_values(self):
        r = rules.signal_vs_noise({"a": [1, 3], "b": [5, 7], "c": [9, 11]})
        self.assertEqual((r["plots"], r["months"]), (3, 6))
        self.assertEqual((r["between_plot_variance"], r["within_plot_variance"], r["ratio"]), (16.0, 2.0, 8.0))
        self.assertEqual(r["icc"], round(30 / 34, 3))  # MSB 32, MSW 2, n0 2

    @needs_numpy
    def test_fewer_than_three_plots_not_reported(self):
        r = rules.signal_vs_noise({"a": [1, 3], "b": [5, 7], "c": [9]})  # c has one month: left out
        self.assertEqual(r["plots"], 2)
        self.assertNotIn("ratio", r)
        self.assertIn("fewer than 3 plots", r["not_reported"])


class Uji2Rows(unittest.TestCase):
    def test_pass_mark(self):
        self.assertTrue(rules.uji2_row(20_000, 50, 20)["pass"])
        self.assertFalse(rules.uji2_row(20_000, 50, 19)["pass"])

    def test_shape_failure(self):
        """A 1000 × 40 m strip: 4 ha, nothing left after 20 m inward, yet a round 4 ha plot keeps ~66 pixels."""
        row = rules.uji2_row(40_000, 100, 0)
        self.assertEqual((row["area_ha"], row["pass"], row["fails_for_shape"]), (4.0, False, True))
        self.assertFalse(rules.uji2_row(1_000, 2, 0)["fails_for_shape"])  # just small: not a shape problem

    def test_compact_expectation(self):
        self.assertEqual(rules.compact_clean_pixels(math.pi * 100 ** 2), int(math.pi * 80 ** 2 // 400))
        self.assertEqual(rules.compact_clean_pixels(100), 0)


class NoCoordinates(unittest.TestCase):
    def test_degree_pairs_either_order(self):
        self.assertTrue(rules.find_coordinates("site at 0.92, 104.45"))
        self.assertTrue(rules.find_coordinates("104.4512 0.9234"))
        self.assertTrue(rules.find_coordinates("-6.21;106.85"))

    def test_index_values_are_not_coordinates(self):
        self.assertEqual(rules.find_coordinates("p1,2024-01,0.6123,0.0412,2.5134,0.3012,104.5"), [])
        self.assertEqual(rules.find_coordinates('{"ndre_median": 0.61, "p30_mm_min": 104.5}'), [])

    def test_projected_pairs(self):
        self.assertTrue(rules.find_coordinates("399960.0 101820.0"))
        self.assertEqual(rules.find_coordinates("scenes 120345,100234"), [])  # counts are integers

    def test_markers(self):
        for text in ('{"geometry": {}}', '{"bbox": [1]}', "POLYGON ((1 2, 3 4))", "EPSG:32648", "MGRS-48NWG",
                     "S2B_48NWG_20240101_0_L2A", "https://sentinel-cogs.s3/x.tif", "/vsicurl/x"):
            self.assertTrue(rules.find_coordinates(text), text)
        self.assertEqual(rules.find_coordinates("a 100 m buffer around the .env point (no plots.geojson)"), [])

    def test_known_values(self):
        self.assertTrue(rules.find_coordinates("lon was 104.45", known=["104.45"]))
        self.assertTrue(rules.find_coordinates("lon was 104.4567", known=["104.45"]))
        self.assertEqual(rules.find_coordinates("1104.45", known=["104.45"]), [])


if __name__ == "__main__":
    unittest.main()
