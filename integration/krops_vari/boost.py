"""The zone boost and the annual Mg/N targets, then the product mass that carries them.

    P95, P25 = percentiles of the reference zones' mean VARI (every valid zone if none qualify)
    V_i      = the zone's mean VARI, or P25 for every zone when P95 − P25 ≤ the blanket cutoff
    B_i      = max(0, 1 − V_i / P95) × zone confidence
    nutrient = B_i × max rate (kg/ha/year) × zone area (ha)
    product  = nutrient ÷ the label's nutrient fraction (never fed back into the boost)
"""
import numpy as np


def _valid_mean(zone):
    mean = zone.get("vari", {}).get("mean_vari")
    return mean is not None and np.isfinite(mean)


def load_zone_areas(settings):
    """Supplied ground area in hectares by zone id (as strings); never inferred from pixels."""
    areas = {str(k): float(v) for k, v in settings.zone_area_ha_by_id.items()}
    if settings.uniform_zone_area_ha is not None and float(settings.uniform_zone_area_ha) <= 0:
        raise ValueError("uniform_zone_area_ha must be positive.")
    for key, value in areas.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Area for zone {key} must be a finite positive number of hectares.")
    return areas


def add_zone_boost_and_fertilizer(zone_json, settings, confidence_by_id, reference_zone_ids=None):
    """Attach zone["recommendation"] to every zone; returns P95 (the reference VARI) or None.

    confidence_by_id maps zone id → confidence in 0.10–1.0 (zone ids missing from it use
    settings.default_zone_confidence). Only reference_zone_ids set P95 and P25; every zone still gets a boost.
    None, or no valid zone among them, = every valid zone.
    """
    valid_means = [z["vari"]["mean_vari"] for z in zone_json if _valid_mean(z)]
    reference_zones = [z for z in zone_json
                       if reference_zone_ids is not None and z["zone_id"] in reference_zone_ids and _valid_mean(z)]
    reference_fallback = reference_zone_ids is not None and not reference_zones
    if not reference_zones:
        reference_zones = [z for z in zone_json if _valid_mean(z)]
    reference_means = [z["vari"]["mean_vari"] for z in reference_zones]
    reference = float(np.percentile(reference_means, 95)) if reference_means else None
    p25 = float(np.percentile(reference_means, 25)) if reference_means else None
    gap = reference - p25 if reference is not None and p25 is not None else None
    cutoff = settings.min_p95_p25_vari_gap_for_blanket_boost
    blanket_mode = gap is not None and gap <= cutoff
    areas = load_zone_areas(settings)
    for nutrient, rate in settings.max_nutrient_kg_ha_per_year.items():
        if nutrient not in ("Mg", "N") or not np.isfinite(float(rate)) or float(rate) < 0:
            raise ValueError(f"Maximum annual elemental nutrient target for {nutrient} must be nonnegative.")
    for product in settings.fertilizer_nutrient_by_product:
        price = settings.fertilizer_price_per_kg.get(product)
        if price is not None and (not np.isfinite(float(price)) or float(price) < 0):
            raise ValueError(f"Product price for {product} must be nonnegative.")
    for zone in zone_json:
        zid = str(zone["zone_id"])
        confidence = float(confidence_by_id.get(zone["zone_id"], confidence_by_id.get(zid, settings.default_zone_confidence)))
        if not np.isfinite(confidence) or not 0.10 <= confidence <= 1.0:
            raise ValueError(f"Confidence for zone {zid} must be in the range 0.10 to 1.0.")
        mean = zone.get("vari", {}).get("mean_vari")
        percentile_rank = (100.0 * sum(value <= float(mean) for value in valid_means) / len(valid_means)
                           if _valid_mean(zone) and valid_means else None)
        boost_input = p25 if blanket_mode else mean
        if reference is None or reference <= 0 or boost_input is None or not np.isfinite(boost_input):
            deficit = boost = None
        else:
            deficit = max(0.0, 1.0 - float(boost_input) / reference)
            boost = deficit * confidence
        if reference is None or reference <= 0 or not _valid_mean(zone):
            zone_specific_deficit = zone_specific_boost = None
        else:
            zone_specific_deficit = max(0.0, 1.0 - float(mean) / reference)
            zone_specific_boost = zone_specific_deficit * confidence
        spacing_area = (settings.use_spacing_based_zone_area_default and zid not in areas
                        and settings.uniform_zone_area_ha is None)
        if zid in areas:
            area = areas[zid]
        elif settings.uniform_zone_area_ha is not None:
            area = float(settings.uniform_zone_area_ha)
        elif settings.use_spacing_based_zone_area_default:
            if settings.default_tree_spacing_m <= 0:
                raise ValueError("default_tree_spacing_m must be positive.")
            area = zone.get("tree_count", 0) * (settings.default_tree_spacing_m ** 2) / 10000.0
        else:
            area = None
        recommendation = {
            "reference_zone_mean_vari": reference,
            "reference_zone_mean_vari_percentile": 95,
            "p25_zone_mean_vari": p25,
            "p95_p25_vari_gap": gap,
            "reference_zone_ids": [z["zone_id"] for z in reference_zones],
            "reference_fallback_all_zones": reference_fallback,
            "zone_mean_vari_percentile_rank": percentile_rank,
            "blanket_vari_gap_cutoff": cutoff,
            "boost_mode": "blanket_p25" if blanket_mode else "zone_specific",
            "boost_vari_input": boost_input,
            "zone_confidence": confidence,
            "zone_specific_vari_deficit_fraction": zone_specific_deficit,
            "zone_specific_boost_fraction": zone_specific_boost,
            "vari_deficit_fraction": deficit,
            "boost_fraction": boost,
            "zone_area_ha": area,
            "fertilizers": {},
            "area_note": ("Approximate area from detected tree count × spacing²; default spacing is 10 m (midpoint of "
                          "9–11 m). Replace with measured zone areas when available." if spacing_area
                          else (None if area is not None
                                else "Provide zone ground area in hectares; pixel dimensions are not converted to area.")),
        }
        for product, nutrient in settings.fertilizer_nutrient_by_product.items():
            max_rate = settings.max_nutrient_kg_ha_per_year.get(nutrient)
            if max_rate is None:
                continue
            if nutrient not in ("Mg", "N"):
                raise ValueError(f"Set fertilizer_nutrient_by_product for {product} to 'Mg' or 'N'.")
            annual_rate = float(max_rate) * boost if boost is not None else None
            annual_total = annual_rate * area if area is not None and annual_rate is not None else None
            item = {
                "base_rate_kg_ha_per_year": float(max_rate),
                "boost_added_rate_kg_ha_per_year": annual_rate,
                "boosted_rate_kg_ha_per_year": annual_rate,
                "recommended_nutrient": nutrient,
                "recommended_nutrient_kg_ha_per_year": annual_rate,
                "annual_total_kg": annual_total,
                "kg_per_palm_per_year": (annual_total / zone["tree_count"]
                                         if annual_total is not None and zone.get("tree_count", 0) > 0 else None),
            }
            item[f"{nutrient.lower()}_kg_ha_per_year"] = annual_rate
            item[f"{nutrient.lower()}_total_kg_per_year"] = annual_total
            recommendation["fertilizers"][product] = item
        zone["recommendation"] = recommendation
    return reference


def product_mass_rows(zone_json, settings):
    """One row per zone and product: the annual elemental target ÷ the label fraction, per zone and per palm."""
    rows = []
    for zone in zone_json:
        rec = zone.get("recommendation", {})
        for product, nutrient in settings.fertilizer_nutrient_by_product.items():
            if settings.max_nutrient_kg_ha_per_year.get(nutrient) is None:
                continue
            item = rec.get("fertilizers", {}).get(product, {})
            nutrient = item.get("recommended_nutrient", nutrient)
            nutrient_total = item.get("annual_total_kg")
            fraction = settings.fertilizer_product_nutrient_fractions.get(product)
            if fraction is not None:
                fraction = float(fraction)
                if not np.isfinite(fraction) or not 0 < fraction <= 1:
                    raise ValueError(f"Product nutrient fraction for {product} must be greater than 0 and at most 1.")
            product_total = nutrient_total / fraction if nutrient_total is not None and fraction is not None else None
            per_palm = (product_total / zone["tree_count"]
                        if product_total is not None and zone.get("tree_count", 0) > 0 else None)
            price = settings.fertilizer_price_per_kg.get(product)
            rows.append({
                "zone_id": zone["zone_id"], "product": product, "nutrient": nutrient,
                "elemental_nutrient_kg_per_zone_per_year": nutrient_total,
                "label_nutrient_fraction": fraction, "tree_count": zone.get("tree_count"),
                "product_kg_per_zone_per_year": product_total,
                "product_kg_per_palm_per_year": per_palm,
                **({"price_per_kg_product": float(price),
                    "annual_product_cost": product_total * float(price) if product_total is not None else None}
                   if price is not None else {}),
            })
    return rows
