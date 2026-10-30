"""Zone confidence and the boost on small made-up zones (no photo, no Core ML)."""
import math
import unittest

from krops_vari.boost import add_zone_boost_and_fertilizer, product_mass_rows
from krops_vari.config import Settings
from krops_vari.confidence import flight_gate_flag, zone_confidence


def zone(zone_id, trees, leaf_px, mean, box=(100, 100, 200, 200)):
    x1, y1, x2, y2 = box
    return {"zone_id": zone_id, "tree_count": trees, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "vari": {"vegetation_pixel_count": leaf_px, "mean_vari": mean}}


def trees(n, leaf_px=1000):
    return [{"vari": {"vegetation_pixel_count": leaf_px}} for _ in range(n)]


class ZoneConfidence(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def run_zones(self, zones, flight=None):
        return zone_confidence(zones, trees(20), (1000, 1000), self.s, flight)

    def test_tree_factor_is_sqrt_of_n_over_9(self):
        zones = [zone(1, 9, 9000, 0.2), zone(2, 4, 4000, 0.2), zone(3, 1, 1000, 0.2)]
        self.run_zones(zones)
        self.assertEqual([z["confidence"]["tree_factor"] for z in zones], [1.0, round(2 / 3, 4), round(1 / 3, 4)])

    def test_reference_needs_zone_factor_0_70(self):
        zones = [zone(1, 5, 5000, 0.2), zone(2, 4, 4000, 0.2)]  # sqrt(5/9) = 0.745, sqrt(4/9) = 0.667
        _, reference, _ = self.run_zones(zones)
        self.assertEqual(reference, [1])

    def test_leaf_factor_catches_a_failed_mask(self):
        zones = [zone(1, 9, 9000 * 0.35, 0.2)]  # half the 0.70 ratio
        self.run_zones(zones)
        self.assertAlmostEqual(zones[0]["confidence"]["leaf_pixel_factor"], 0.5)

    def test_flight_times_zone_factor_clamped_to_the_floor(self):
        zones = [zone(1, 9, 9000, 0.2), zone(2, 1, 1000, 0.2)]
        by_id, _, flags = self.run_zones(zones, flight=0.2)
        self.assertAlmostEqual(by_id[1], 0.2)
        self.assertAlmostEqual(by_id[2], 0.10)  # 0.2 × 0.333 = 0.067 → the floor
        self.assertNotIn("flight confidence (L6) not computed", " ".join(flags))

    def test_default_stands_in_and_flags(self):
        zones = [zone(1, 9, 9000, 0.2)]
        by_id, _, flags = self.run_zones(zones)
        self.assertAlmostEqual(by_id[1], 0.60)
        self.assertTrue(zones[0]["confidence"]["may_be_less_accurate"])
        self.assertTrue(any("not computed" in f for f in flags))


class FlightGate(unittest.TestCase):
    """Team decision (23 Oct 2026): a low or incomplete flight is flagged on every zone; the dose is still given."""

    def setUp(self):
        self.s = Settings()

    def run_flight(self, flight_result):
        zones = [zone(1, 9, 9000, 0.2), zone(2, 9, 9000, 0.3)]
        by_id, _, flags = zone_confidence(zones, trees(20), (1000, 1000), self.s,
                                          flight_result["confidence"], flight_result)
        return zones, by_id, flags

    def test_low_flight_flags_every_zone_with_its_reasons(self):
        zones, by_id, flags = self.run_flight({"confidence": 0.35, "label": "rendah", "final": True,
                                               "reasons": [{"code": "wet_rain_before"}, {"code": "cloud_overcast"}]})
        for z in zones:  # full 9-tree zones, which would otherwise carry no flag
            self.assertTrue(z["confidence"]["may_be_less_accurate"])
            self.assertTrue(any("low flight confidence 0.35 (wet_rain_before, cloud_overcast)" in f
                                for f in z["confidence"]["flags"]), z["confidence"]["flags"])
        self.assertAlmostEqual(by_id[1], 0.35)  # the dose is still given, scaled by the confidence
        self.assertTrue(any("doses still given" in f for f in flags))

    def test_incomplete_flight_is_flagged(self):
        zones, _, _ = self.run_flight({"confidence": 0.6, "label": "sedang", "final": False, "reasons": []})
        self.assertTrue(any("not final" in f for f in zones[0]["confidence"]["flags"]))
        self.assertTrue(zones[0]["confidence"]["may_be_less_accurate"])

    def test_good_flight_is_not_flagged(self):
        zones, _, flags = self.run_flight({"confidence": 0.85, "label": "tinggi", "final": True,
                                           "reasons": [{"code": "cloud_overcast"}]})
        self.assertEqual(zones[0]["confidence"]["flags"], [])
        self.assertFalse(zones[0]["confidence"]["may_be_less_accurate"])
        self.assertFalse(any("flight gate" in f for f in flags))

    def test_flagged_even_when_not_multiplied_in(self):
        zones = [zone(1, 9, 9000, 0.2)]
        zone_confidence(zones, trees(20), (1000, 1000), self.s, None,
                        {"confidence": 0.3, "label": "rendah", "final": True, "reasons": []})
        self.assertTrue(any("low flight confidence" in f for f in zones[0]["confidence"]["flags"]))

    def test_no_result_no_flag(self):
        self.assertIsNone(flight_gate_flag(None))


class Boost(unittest.TestCase):
    def setUp(self):
        self.s = Settings()

    def test_small_zone_doesnt_set_p95(self):
        zones = [zone(1, 9, 9000, 0.10), zone(2, 9, 9000, 0.20), zone(3, 1, 1000, 0.40)]
        p95 = add_zone_boost_and_fertilizer(zones, self.s, {1: 0.6, 2: 0.6, 3: 0.2}, reference_zone_ids=[1, 2])
        self.assertAlmostEqual(p95, 0.10 + 0.95 * 0.10)  # zone 3's 0.40 would have raised it
        self.assertEqual(zones[2]["recommendation"]["boost_fraction"], 0.0)  # above P95: no boost, still a record

    def test_boost_formula_and_nutrient(self):
        zones = [zone(1, 9, 9000, 0.10), zone(2, 9, 9000, 0.20)]
        add_zone_boost_and_fertilizer(zones, self.s, {1: 0.5, 2: 0.5}, reference_zone_ids=[1, 2])
        rec = zones[0]["recommendation"]
        p95 = 0.10 + 0.95 * 0.10
        self.assertAlmostEqual(rec["boost_fraction"], (1 - 0.10 / p95) * 0.5)
        mg = rec["fertilizers"]["kieserite_Mg_expensive"]
        self.assertAlmostEqual(mg["annual_total_kg"], rec["boost_fraction"] * 22.0 * 0.09)  # 9 palms × 0.01 ha

    def test_blanket_mode_when_zones_are_alike(self):
        zones = [zone(1, 9, 9000, 0.200), zone(2, 9, 9000, 0.210)]  # gap ≤ 0.03
        add_zone_boost_and_fertilizer(zones, self.s, {}, reference_zone_ids=[1, 2])
        self.assertEqual(zones[0]["recommendation"]["boost_mode"], "blanket_p25")
        self.assertEqual(zones[0]["recommendation"]["boost_fraction"], zones[1]["recommendation"]["boost_fraction"])

    def test_product_mass_divides_by_the_label_fraction(self):
        zones = [zone(1, 9, 9000, 0.10), zone(2, 9, 9000, 0.20)]
        add_zone_boost_and_fertilizer(zones, self.s, {1: 0.5, 2: 0.5}, reference_zone_ids=[1, 2])
        rows = {(r["zone_id"], r["product"]): r for r in product_mass_rows(zones, self.s)}
        urea = rows[(1, "urea_N_expensive")]
        self.assertTrue(math.isclose(urea["product_kg_per_zone_per_year"],
                                     urea["elemental_nutrient_kg_per_zone_per_year"] / 0.46))


if __name__ == "__main__":
    unittest.main()
