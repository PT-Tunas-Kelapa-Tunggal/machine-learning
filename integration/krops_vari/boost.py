"""The zone boost and the annual Mg/N targets, then the product mass that carries them.

    P95, P25 = percentiles of the reference zones' mean VARI (every valid zone if none qualify)
    V_i      = the zone's mean VARI, or P25 for every zone when P95 − P25 ≤ the blanket cutoff
    B_i      = max(0, 1 − V_i / P95) × zone confidence
    nutrient = B_i × max rate (kg/ha/year) × zone area (ha)
    product  = nutrient ÷ the label's nutrient fraction (never fed back into the boost)
"""
import numpy as np

MG_TO_MGO_FACTOR = 40.304 / 24.305


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

    # Iteration 2: lowest min VARI diagnostics
    valid_mins = [z["vari"].get("min_vari") for z in zone_json
                  if z.get("vari", {}).get("min_vari") is not None
                  and np.isfinite(z["vari"]["min_vari"])]
    lowest_zone_min_vari = min(valid_mins) if valid_mins else None

    lowest_zone_mean_vari = min(valid_means) if valid_means else None

    total_detected_trees = sum(z.get("tree_count", 0) for z in zone_json)
    for nutrient, rate in settings.max_nutrient_kg_ha_per_year.items():
        if nutrient not in ("Mg", "N") or not np.isfinite(float(rate)) or float(rate) < 0:
            raise ValueError(f"Maximum annual elemental nutrient target for {nutrient} must be nonnegative.")
    for product in settings.fertilizer_nutrient_by_product:
        price = settings.fertilizer_price_per_kg.get(product)
        if price is not None and (not np.isfinite(float(price)) or float(price) < 0):
            raise ValueError(f"Product price for {product} must be nonnegative.")
    for zone in zone_json:
        zid = str(zone["zone_id"])
        if settings.include_iteration2_fields:
            zone_min = zone.get("vari", {}).get("min_vari")
            is_lowest_min = (
                zone_min is not None
                and np.isfinite(zone_min)
                and lowest_zone_min_vari is not None
                and np.isclose(zone_min, lowest_zone_min_vari, rtol=0.0, atol=1e-12)
            )
            zone_mean_val = zone.get("vari", {}).get("mean_vari")
            is_lowest_mean = (
                zone_mean_val is not None
                and np.isfinite(zone_mean_val)
                and lowest_zone_mean_vari is not None
                and np.isclose(zone_mean_val, lowest_zone_mean_vari, rtol=0.0, atol=1e-12)
            )
            zone["vari"]["lowest_min_vari_zone"] = bool(is_lowest_min)
            zone["vari"]["lowest_zone_min_vari_value"] = (
                float(lowest_zone_min_vari) if lowest_zone_min_vari is not None else None
            )
            zone["vari"]["min_vari_review_note"] = (
                "Zone has the lowest valid min VARI; review imagery/trees for non-canopy pixels, shade, or stress."
                if is_lowest_min else None
            )
            zone["vari"]["lowest_mean_vari_zone"] = bool(is_lowest_mean)

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
                        and settings.uniform_zone_area_ha is None and settings.total_farm_area_ha is None)
        if zid in areas:
            area = areas[zid]
            area_note = "User-supplied area for this zone."
        elif settings.uniform_zone_area_ha is not None:
            area = float(settings.uniform_zone_area_ha)
            area_note = "Configured uniform area applied to every zone."
        elif settings.total_farm_area_ha is not None:
            tree_count = max(0, int(zone.get("tree_count", 0)))
            area = (float(settings.total_farm_area_ha) * tree_count / total_detected_trees
                    if total_detected_trees > 0 else None)
            area_note = (
                "Total farm area split across zones in proportion to detected tree counts; "
                "replace with measured zone areas when available."
            )
        elif settings.use_spacing_based_zone_area_default:
            if settings.default_tree_spacing_m <= 0:
                raise ValueError("default_tree_spacing_m must be positive.")
            area = zone.get("tree_count", 0) * (settings.default_tree_spacing_m ** 2) / 10000.0
            area_note = ("Approximate area from detected tree count × spacing²; default spacing is 10 m (midpoint of "
                         "9–11 m). Replace with measured zone areas when available.")
        else:
            area = None
            area_note = "Provide zone ground area in hectares; pixel dimensions are not converted to area."
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
            "area_note": area_note if settings.include_iteration2_fields else (
                "Approximate area from detected tree count × spacing²; default spacing is 10 m (midpoint of "
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
            label_info = settings.fertilizer_product_label_fractions.get(product, {})
            label_basis = label_info.get("basis")
            label_fraction = label_info.get("fraction")
            if label_fraction is not None:
                label_fraction = float(label_fraction)

            if nutrient_total is None or fraction is None:
                product_total = None
            elif nutrient == "Mg" and label_basis == "MgO" and label_fraction is not None:
                mgo_target = nutrient_total * MG_TO_MGO_FACTOR
                product_total = mgo_target / label_fraction
            else:
                product_total = nutrient_total / fraction

            per_palm = (product_total / zone["tree_count"]
                        if product_total is not None and zone.get("tree_count", 0) > 0 else None)
            price = settings.fertilizer_price_per_kg.get(product)

            base_product_total = None
            combined_product_total = None
            combined_per_palm = None
            base_rate = settings.fertilizer_base_product_kg_per_palm_per_year.get(product)
            if base_rate is not None:
                base_product_total = float(base_rate) * zone.get("tree_count", 0)
                combined_product_total = (base_product_total + product_total
                                          if product_total is not None else None)
                combined_per_palm = (
                    combined_product_total / zone["tree_count"]
                    if combined_product_total is not None and zone.get("tree_count", 0) > 0 else None
                )

            if settings.include_iteration2_fields:
                row = {
                    "zone_id": zone["zone_id"], "product": product, "nutrient": nutrient,
                    "elemental_nutrient_kg_per_zone_per_year": nutrient_total,
                    "product_label_basis": label_basis,
                    "product_label_fraction": label_fraction,
                    "elemental_nutrient_fraction_per_kg_product": fraction,
                    "tree_count": zone.get("tree_count"),
                    "product_kg_per_zone_per_year": product_total,
                    "product_kg_per_palm_per_year": per_palm,
                    "base_product_kg_per_zone_per_year": base_product_total,
                    "base_plus_supplement_product_kg_per_zone_per_year": combined_product_total,
                    "base_plus_supplement_product_kg_per_palm_per_year": combined_per_palm,
                }
            else:
                row = {
                    "zone_id": zone["zone_id"], "product": product, "nutrient": nutrient,
                    "elemental_nutrient_kg_per_zone_per_year": nutrient_total,
                    "label_nutrient_fraction": fraction, "tree_count": zone.get("tree_count"),
                    "product_kg_per_zone_per_year": product_total,
                    "product_kg_per_palm_per_year": per_palm,
                }

            if price is not None:
                row["price_per_kg_product"] = float(price)
                row["annual_product_cost"] = product_total * float(price) if product_total is not None else None
            rows.append(row)
    return rows
