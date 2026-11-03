"""Field-wide NPK / NPKMg baseline comparison and UI data generation.

Implements Part 2 of the VARI analysis pipeline:
  - Parses user submission requests (farm profile, fertilizer history, harvest).
  - Calculates annual fertilizer nutrient supply from dosePerTree × frequency × tree count ÷ farm area.
  - Compares supply against elemental baseline (N, P, K, Mg).
  - Calculates standalone top-up options (SP-36 for P, KCl/MOP for K, ZA/Urea for N, Kieserite/Dolomite for Mg).
  - Routes suggestions based on NPK shortfall threshold (e.g. 30%).
  - Formats combined recommendations and app-facing ui_data.
"""
from __future__ import annotations

import math
from typing import Any

# Atomic masses for oxide-to-element stoichiometry
ATOMIC_MASS_G_MOL = {"P": 30.973761998, "K": 39.0983, "Mg": 24.305, "O": 15.999}

OXIDE_TO_ELEMENT = {
    "P2O5": {
        "element": "P",
        "factor": (2 * ATOMIC_MASS_G_MOL["P"]) / (2 * ATOMIC_MASS_G_MOL["P"] + 5 * ATOMIC_MASS_G_MOL["O"]),
    },
    "K2O": {
        "element": "K",
        "factor": (2 * ATOMIC_MASS_G_MOL["K"]) / (2 * ATOMIC_MASS_G_MOL["K"] + ATOMIC_MASS_G_MOL["O"]),
    },
    "MgO": {
        "element": "Mg",
        "factor": ATOMIC_MASS_G_MOL["Mg"] / (ATOMIC_MASS_G_MOL["Mg"] + ATOMIC_MASS_G_MOL["O"]),
    },
}

MG_PER_MGO = OXIDE_TO_ELEMENT["MgO"]["factor"]
MGO_PER_MG = 1.0 / MG_PER_MGO


def parse_analysis(label: str) -> dict[str, float]:
    """Read N-P2O5-K2O[-MgO] percentages from a string such as '12-6-22-3' or '15-15-15'."""
    try:
        values = [float(value.strip()) for value in label.replace(",", "-").split("-")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("Set NPK label as e.g. '15-15-15' or '12-6-22-3'.") from exc
    if len(values) == 3:
        values.append(0.0)
    if len(values) != 4 or any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError("Enter three or four label percentages, each from 0 to 100.")
    return dict(zip(("N", "P2O5", "K2O", "MgO"), values))


def base_product_rates_from_history(submission_data: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, str]]:
    """Extract annual base product rates per palm (kg/palm/year) from user submission history."""
    history = (submission_data or {}).get("fertilizerHistory") or {}
    freq_value = history.get("freq")
    freq_per_year = float(1.0 if freq_value is None else freq_value)
    products = history.get("fertilizers", [])

    def _rate_for_aliases(aliases: tuple[str, ...], default_rate: float = 1.0) -> tuple[float, str]:
        alias_set = {a.casefold() for a in aliases}
        entries = [
            item for item in products
            if str(item.get("name", "")).strip().casefold() in alias_set
            and item.get("dosePerTree") is not None
        ]
        if not entries:
            return float(default_rate), f"default: {default_rate} kg/palm/year (no entered product dose)"
        annual_rate = sum(float(item["dosePerTree"]) * freq_per_year for item in entries)
        dose_terms = " + ".join(f"{float(item['dosePerTree']):g} × {freq_per_year:g}" for item in entries)
        return annual_rate, f"from UI history: ({dose_terms}) kg/palm/year"

    kieserite_rate, kieserite_src = _rate_for_aliases(("kieserite", "kieserit"), default_rate=1.0)
    dolomite_rate, dolomite_src = _rate_for_aliases(("dolomite", "dolomit"), default_rate=1.0)

    rates = {
        "kieserite_Mg_expensive": kieserite_rate,
        "dolomite_Mg_cheaper": dolomite_rate,
        "ZA_N_cheaper": None,
        "urea_N_expensive": None,
    }
    sources = {
        "kieserite_Mg_expensive": kieserite_src,
        "dolomite_Mg_cheaper": dolomite_src,
    }
    return rates, sources


def build_annual_nutrients(
    label_percent: dict[str, float],
    bags_per_ha_year: float,
    bag_size_kg: float = 50.0,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Return elemental supply per hectare/year, oxide/element breakdown, and product kg/ha/year."""
    effective_bag_size = float(bag_size_kg)
    if not math.isfinite(effective_bag_size) or effective_bag_size <= 0:
        raise ValueError("bag_size_kg must be greater than zero.")
    product_kg_ha_year = bags_per_ha_year * effective_bag_size
    oxide_or_element = {
        nutrient: product_kg_ha_year * percent / 100.0
        for nutrient, percent in label_percent.items()
    }
    elemental = {
        "N": oxide_or_element.get("N", 0.0),
        "P": oxide_or_element.get("P2O5", 0.0) * OXIDE_TO_ELEMENT["P2O5"]["factor"],
        "K": oxide_or_element.get("K2O", 0.0) * OXIDE_TO_ELEMENT["K2O"]["factor"],
        "Mg": oxide_or_element.get("MgO", 0.0) * OXIDE_TO_ELEMENT["MgO"]["factor"],
    }
    return elemental, oxide_or_element, product_kg_ha_year


def build_recorded_fertilizer_history_supply(
    submission_data: dict[str, Any],
    tree_count: int,
    farm_area_ha: float,
) -> dict[str, Any] | None:
    """Calculate elemental nutrient supply from entered product doses, per ha/year."""
    history = (submission_data or {}).get("fertilizerHistory") or {}
    entries = history.get("fertilizers", [])
    if not entries:
        return None
    frequency = history.get("freq")
    if frequency is None:
        raise ValueError("fertilizerHistory.freq is required to annualize dosePerTree.")
    frequency = float(frequency)
    tree_count = int(tree_count)
    area_ha = float(farm_area_ha)
    if area_ha <= 0:
        raise ValueError("Farm area must be positive to calculate kg/ha/year.")

    totals = {nutrient: 0.0 for nutrient in ("N", "P", "K", "Mg")}
    breakdown: list[dict[str, Any]] = []
    ignored: list[dict[str, str]] = []
    oxide_factors = {
        "P2O5": OXIDE_TO_ELEMENT["P2O5"]["factor"],
        "K2O": OXIDE_TO_ELEMENT["K2O"]["factor"],
        "MgO": OXIDE_TO_ELEMENT["MgO"]["factor"],
    }
    simple_product_labels = {
        "za": {"N": ("N", 0.21, 1.0)},
        "urea": {"N": ("N", 0.46, 1.0)},
        "dolomite": {"Mg": ("MgO", 0.22, oxide_factors["MgO"])},
        "dolomit": {"Mg": ("MgO", 0.22, oxide_factors["MgO"])},
        "kieserite": {"Mg": ("MgO", 0.255, oxide_factors["MgO"])},
        "kieserit": {"Mg": ("MgO", 0.255, oxide_factors["MgO"])},
        "sp-36": {"P": ("P2O5", 0.36, oxide_factors["P2O5"])},
        "kcl": {"K": ("K2O", 0.60, oxide_factors["K2O"])},
        "kcl (mop)": {"K": ("K2O", 0.60, oxide_factors["K2O"])},
        "kcl/mop": {"K": ("K2O", 0.60, oxide_factors["K2O"])},
        "mop": {"K": ("K2O", 0.60, oxide_factors["K2O"])},
    }

    for entry in entries:
        name = str(entry.get("name", "Unknown product"))
        dose = entry.get("dosePerTree")
        if dose is None:
            ignored.append({"product": name, "reason": "dosePerTree is missing"})
            continue
        dose = float(dose)
        annual_product_kg_per_tree = dose * frequency
        annual_product_kg_farm = annual_product_kg_per_tree * tree_count
        product_kg_ha = annual_product_kg_farm / area_ha
        normalized = name.strip().casefold()
        contribution = {nutrient: 0.0 for nutrient in totals}
        label_basis_kg_ha: dict[str, float] = {}

        if normalized == "npk":
            comp = entry.get("composition") or {}
            if not all(key in comp for key in ("N", "P", "K", "Mg")):
                ignored.append({"product": name, "reason": "NPK composition is missing"})
                continue
            labels = {
                "N": float(comp["N"]),
                "P2O5": float(comp["P"]),
                "K2O": float(comp["K"]),
                "MgO": float(comp["Mg"]),
            }
            label_basis_kg_ha = {key: product_kg_ha * pct / 100.0 for key, pct in labels.items()}
            contribution = {
                "N": label_basis_kg_ha["N"],
                "P": label_basis_kg_ha["P2O5"] * oxide_factors["P2O5"],
                "K": label_basis_kg_ha["K2O"] * oxide_factors["K2O"],
                "Mg": label_basis_kg_ha["MgO"] * oxide_factors["MgO"],
            }
        elif normalized in simple_product_labels:
            nutrient, (basis, label_fraction, factor) = next(iter(simple_product_labels[normalized].items()))
            label_basis_kg_ha[basis] = product_kg_ha * label_fraction
            contribution[nutrient] = label_basis_kg_ha[basis] * factor
        else:
            ignored.append({"product": name, "reason": "no configured nutrient-label conversion"})
            continue

        for nutrient in totals:
            totals[nutrient] += contribution[nutrient]
        breakdown.append({
            "product": name,
            "dose_kg_per_tree_per_application": dose,
            "applications_per_year": frequency,
            "detected_tree_count": tree_count,
            "annual_product_kg_per_farm": annual_product_kg_farm,
            "farm_area_ha": area_ha,
            "annual_product_kg_ha": product_kg_ha,
            "label_basis_kg_ha": label_basis_kg_ha,
            "elemental_supply_kg_ha_year": contribution,
        })

    return {
        "elemental_supply_kg_ha_year": totals,
        "products": breakdown,
        "ignored_products": ignored,
    }


def build_field_npk_json_record(
    label_percent: dict[str, float],
    bags_per_ha_year: float,
    bag_size_kg: float,
    history_supply: dict[str, Any] | None,
    baseline_elemental: dict[str, float],
    dose_per_tree: float | None,
    applications_per_year: float | None,
    tree_count: int,
    farm_area_ha: float,
    standalone_products: dict[str, Any] | None = None,
    route_threshold_percent: float = 30.0,
) -> dict[str, Any]:
    """Return machine-readable field-wide blend comparison and standalone product options."""
    npk_elemental_year, label_basis_year, product_kg = build_annual_nutrients(
        label_percent, bags_per_ha_year, bag_size_kg
    )
    elemental_year = (
        history_supply["elemental_supply_kg_ha_year"]
        if history_supply is not None else npk_elemental_year
    )
    comparisons = {}
    gaps = {}
    for nutrient, target in baseline_elemental.items():
        supplied = float(elemental_year.get(nutrient, 0.0))
        target_f = float(target)
        difference = supplied - target_f
        gap = max(0.0, target_f - supplied)
        gaps[nutrient] = gap
        comparisons[nutrient] = {
            "baseline_elemental_kg_ha_year": target_f,
            "recorded_fertilizer_supply_elemental_kg_ha_year": supplied,
            "npk_blend_only_elemental_kg_ha_year": float(npk_elemental_year.get(nutrient, 0.0)),
            "difference_elemental_kg_ha_year": difference,
            "shortfall_elemental_kg_ha_year": gap,
            "shortfall_percent_of_baseline": gap / target_f * 100.0 if target_f > 0 else 0.0,
            "surplus_percent_of_baseline": max(0.0, difference) / target_f * 100.0 if target_f > 0 else 0.0,
        }

    # Default standalone products if not passed
    standalone_cfg = standalone_products or {
        "N": [("ZA", "N", 0.21, 1.0, 50.0), ("Urea", "N", 0.46, 1.0, 50.0)],
        "P": [("SP-36", "P2O5", 0.36, OXIDE_TO_ELEMENT["P2O5"]["factor"], 50.0)],
        "K": [("KCl / MOP", "K2O", 0.60, OXIDE_TO_ELEMENT["K2O"]["factor"], 50.0)],
        "Mg": [
            ("Kieserite", "MgO", 0.255, OXIDE_TO_ELEMENT["MgO"]["factor"], 50.0),
            ("Dolomite", "MgO", 0.22, OXIDE_TO_ELEMENT["MgO"]["factor"], 50.0),
        ],
    }

    standalone: dict[str, Any] = {}
    for nutrient, products in standalone_cfg.items():
        gap = gaps.get(nutrient, 0.0)
        options = []
        for name, label_basis, label_fraction, oxide_factor, bag_size in products:
            elemental_fraction = label_fraction * oxide_factor
            product_needed = gap / elemental_fraction if gap > 0 and elemental_fraction > 0 else 0.0
            exact_bags = product_needed / bag_size if bag_size > 0 else 0.0
            options.append({
                "product": name,
                "label_basis": label_basis,
                "label_guarantee_percent": label_fraction * 100.0,
                "label_guarantee_fraction_kg_label_basis_per_kg_product": label_fraction,
                "oxide_to_element_factor": oxide_factor,
                "elemental_nutrient_fraction_kg_per_kg_product": elemental_fraction,
                "elemental_gap_kg_ha_year": gap,
                "product_needed_kg_ha_year": product_needed,
                "bag_size_kg": bag_size,
                "exact_bags_ha_year": exact_bags,
                "full_bags_to_buy_ha_year": math.ceil(exact_bags - 1e-12) if exact_bags > 0 else 0,
            })
        standalone[nutrient] = {
            "elemental_shortfall_kg_ha_year": gap,
            "options_are_alternatives": len(options) > 1,
            "options": options,
        }

    largest_npk_gap = max(
        (gaps[n] / float(baseline_elemental[n]) * 100.0 if baseline_elemental.get(n) else 0.0)
        for n in ("N", "P", "K")
    )
    all_met = all(gaps.get(n, 0.0) == 0 for n in baseline_elemental)
    if all_met:
        route = "no_field_wide_top_up_indicated"
    elif largest_npk_gap <= route_threshold_percent:
        route = "standalone_top_up_route_with_zone_boosts_kept_separate"
    else:
        route = "review_blend_quantity_or_alternative_npk_before_top_up"

    return {
        "scope": (
            "all recognized recorded fertilizer-history products compared with the configured "
            "elemental baseline; dose-based annual use"
        ),
        "user_blend": {
            "label_order": "N-P2O5-K2O-MgO",
            "label_percent": {key: float(value) for key, value in label_percent.items()},
            "bags_per_ha_year": float(bags_per_ha_year),
            "bag_size_kg": float(bag_size_kg),
            "product_kg_ha_year": float(product_kg),
            "label_basis_kg_ha_year": {key: float(value) for key, value in label_basis_year.items()},
            "elemental_supply_kg_ha_year": {key: float(value) for key, value in npk_elemental_year.items()},
            "recorded_history_total_elemental_supply_kg_ha_year": {
                key: float(value) for key, value in elemental_year.items()
            },
            "dose_per_tree_kg_per_application": dose_per_tree,
            "fertilizer_applications_per_year": applications_per_year,
            "detected_tree_count_used": tree_count,
            "farm_area_ha_used": farm_area_ha,
            "npk_product_kg_farm_per_year": (
                dose_per_tree * applications_per_year * tree_count
                if dose_per_tree is not None and applications_per_year is not None else None
            ),
        },
        "recorded_fertilizer_history_supply": history_supply,
        "baseline_elemental_kg_ha_year": {key: float(value) for key, value in baseline_elemental.items()},
        "nutrient_comparison": comparisons,
        "standalone_top_up_options": standalone,
        "routing": {
            "largest_npk_shortfall_percent": largest_npk_gap,
            "small_gap_threshold_percent": float(route_threshold_percent),
            "suggested_route": route,
        },
        "formulas": {
            "annual_product_kg_per_farm": "dosePerTree kg/tree/application × fertilizerHistory.freq applications/year × detected tree count",
            "annual_product_kg_per_ha": "annual product kg per farm ÷ farm areaHa",
            "bags_per_ha_year": "annual product kg/ha/year ÷ bagKg; derived for reference, not entered by the farmer",
            "elemental_supply": "annual product kg/ha/year × (label guarantee % ÷ 100) × oxide-to-element factor; N factor=1",
            "elemental_gap": "max(0, baseline elemental kg/ha/year − total recognized recorded fertilizer supply kg/ha/year)",
            "product_needed": "elemental gap ÷ (label fraction × oxide-to-element factor)",
            "bags_to_buy": "ceil(product needed kg ÷ bag size kg)",
        },
    }


def build_zone_boost_json_records(zone_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select existing per-zone VARI boost amounts for the combined JSON section."""
    records = []
    for zone in zone_records:
        rec = zone.get("recommendation", {})
        records.append({
            "zone_id": zone.get("zone_id"),
            "tree_count": zone.get("tree_count"),
            "area_ha": rec.get("zone_area_ha"),
            "boost_mode": rec.get("boost_mode"),
            "boost_fraction": rec.get("boost_fraction"),
            "overall_field_confidence": rec.get("overall_field_confidence", rec.get("zone_confidence")),
            "fertilizer_boosts": rec.get("fertilizers", {}),
        })
    return records


def build_ui_data(
    submission_data: dict[str, Any],
    combined_result: dict[str, Any],
    image_name: str | None = None,
) -> dict[str, Any]:
    """Build the client/app-facing view matching Iteration 2 notebook output."""
    farm_input = submission_data.get("farm") or {}
    farm_profile = {
        "name": farm_input.get("name"),
        "location_label": farm_input.get("location_label"),
        "area_ha": farm_input.get("areaHa"),
        "perimeter_m": farm_input.get("perimeterM"),
        "planting_year": farm_input.get("year"),
        "soil_type": farm_input.get("soil"),
        "farm_boundary_geojson": farm_input.get("farmBoundaryGeojson"),
        "review_month": submission_data.get("month"),
    }
    fertilizer_history = submission_data.get("fertilizerHistory")
    harvest_history = submission_data.get("harvest")

    npk = combined_result.get("combined_fertilizer_calculations", {}).get("field_wide_npk_baseline_comparison", {})
    zone_product_rows = combined_result.get("product_mass", [])

    zone_app_rows = []
    for zone in combined_result.get("zones", []):
        zid = zone.get("zone_id")
        zrec = zone.get("recommendation", {})
        nutrient_boosts = {}
        for product, item in zrec.get("fertilizers", {}).items():
            nutrient = item.get("recommended_nutrient")
            if nutrient and nutrient not in nutrient_boosts:
                nutrient_boosts[nutrient] = {
                    "boost_added_rate_kg_ha_year": item.get("boost_added_rate_kg_ha_per_year"),
                    "elemental_boost_kg_zone_year": item.get("annual_total_kg"),
                    "elemental_boost_kg_palm_year": item.get("kg_per_palm_per_year"),
                }
        zone_products = [
            {
                **row,
                "select_one_product_for_this_nutrient": sum(
                    1 for cand in zone_product_rows
                    if cand.get("zone_id") == zid and cand.get("nutrient") == row.get("nutrient")
                ) > 1,
            }
            for row in zone_product_rows if row.get("zone_id") == zid
        ]
        zone_app_rows.append({
            "area_id": f"Z{zid}",
            "zone_id": zid,
            "tree_count": zone.get("tree_count"),
            "area_ha": zrec.get("zone_area_ha"),
            "image_bbox_px": zone.get("bbox"),
            "map_boundary_geojson": None,
            "images": {
                "base_crop": zone.get("base_image_crop"),
                "exg_crop": zone.get("exg_image_crop"),
            },
            "vari_summary": zone.get("vari", {}),
            "boost": {
                "mode": zrec.get("boost_mode"),
                "fraction": zrec.get("boost_fraction"),
                "percent": (
                    zrec.get("boost_fraction") * 100.0
                    if zrec.get("boost_fraction") is not None else None
                ),
                "confidence": zrec.get("overall_field_confidence", zrec.get("zone_confidence")),
            },
            "elemental_nutrient_boosts": nutrient_boosts,
            "product_supplement_options": zone_products,
            "application_status": None,
        })

    # Read NPK product name
    npk_product_name = None
    if fertilizer_history and "fertilizers" in fertilizer_history:
        for f in fertilizer_history["fertilizers"]:
            if str(f.get("name", "")).strip().upper() == "NPK":
                npk_product_name = f.get("name", "NPK")
                break

    ui_blend = dict(npk.get("user_blend", {}))
    ui_blend["product_name"] = npk_product_name
    ui_blend["price_per_bag_rp"] = None

    ui_standalone_options: dict[str, Any] = {}
    for nutrient, group in npk.get("standalone_top_up_options", {}).items():
        ui_options = []
        for opt in group.get("options", []):
            opt_ui = dict(opt)
            opt_ui["price_per_bag_rp"] = None
            opt_ui["calculated_need_cost_rp"] = None
            opt_ui["full_bag_purchase_cost_rp"] = None
            ui_options.append(opt_ui)
        ui_standalone_options[nutrient] = {
            **group,
            "options": ui_options,
        }

    nutrient_cards = []
    for nutrient, cmp_data in npk.get("nutrient_comparison", {}).items():
        gap = cmp_data["shortfall_elemental_kg_ha_year"]
        diff = cmp_data["difference_elemental_kg_ha_year"]
        nutrient_cards.append({
            "nutrient": nutrient,
            "current_recorded_fertilizer_supply_kg_ha_year": cmp_data["recorded_fertilizer_supply_elemental_kg_ha_year"],
            "baseline_need_kg_ha_year": cmp_data["baseline_elemental_kg_ha_year"],
            "difference_kg_ha_year": diff,
            "shortfall_kg_ha_year": gap,
            "status_vs_baseline": "below" if gap > 0 else ("above" if diff > 0 else "matched"),
            "basis_note": (
                "Annual nutrient supply calculated from entered per-tree product doses × frequency "
                "× detected tree count ÷ farm area; not a soil/leaf measurement."
            ),
        })

    missing_app_inputs = []
    for label, val in [
        ("farm name", farm_profile.get("name")),
        ("location label", farm_profile.get("location_label")),
        ("farm perimeter", farm_profile.get("perimeter_m")),
        ("planting year", farm_profile.get("planting_year")),
        ("georeferenced farm boundary", farm_profile.get("farm_boundary_geojson")),
        ("measured soil/leaf nutrient results", None),
        ("georeferenced zone boundaries", None),
        ("fertilizer product prices", None),
        ("fertilizer application history", fertilizer_history),
        ("harvest history", harvest_history),
    ]:
        if val is None:
            missing_app_inputs.append(label)

    total_zone_area = sum(z.get("recommendation", {}).get("zone_area_ha") or 0.0 for z in combined_result.get("zones", []))

    return {
        "schema_version": "1.0",
        "submission_month": submission_data.get("month"),
        "farm_overview": {
            **farm_profile,
            "area_ha": farm_profile.get("area_ha") or (total_zone_area if total_zone_area > 0 else None),
            "analysis_area_ha": total_zone_area if total_zone_area > 0 else farm_profile.get("area_ha"),
            "area_source": "user_submission_request.json" if farm_profile.get("area_ha") is not None else "spacing_estimate",
            "source_image": image_name,
            "zone_overlay_image": None,
            "map_coordinate_basis": "Image pixel coordinates only unless GeoJSON boundaries are supplied.",
        },
        "configured_annual_nutrient_comparison": {
            "blend": ui_blend,
            "fertilizer_history_supply": npk.get("recorded_fertilizer_history_supply"),
            "nutrient_cards": nutrient_cards,
            "measured_soil_leaf_status": None,
            "standalone_field_top_up_options": ui_standalone_options,
            "routing": npk.get("routing", {}),
        },
        "zone_recommendations": zone_app_rows,
        "fertilizer_cost_inputs": {
            "currency": "IDR",
            "price_per_bag_rp": {},
            "prices_complete": False,
            "note": "Enter verified price per bag for each listed product to enable a complete budget; missing prices remain null.",
        },
        "fertilizer_application_history": fertilizer_history,
        "harvest_history": harvest_history,
        "missing_inputs_for_mockup_fields": missing_app_inputs,
        "interpretation_note": (
            "Field-wide NPK top-ups and per-zone VARI boosts are supplied as separate recommendations. "
            "The client must not combine them into a single dose until an allocation policy is configured. "
            "Zone product options are alternatives for the same nutrient; select one option per nutrient."
        ),
        "data_sources": {
            "farm_overview": (
                "Name, areaHa, perimeterM, planting year, soil, and review month come from "
                "user_submission_request.json; unentered profile values stay null."
            ),
            "configured_annual_nutrient_comparison.blend": (
                "NPK label and per-bag composition come from the NPK submission record. NPK annual product "
                "kg/ha is calculated from dosePerTree × fertilizerHistory.freq × detected tree count ÷ farm "
                "areaHa; bag equivalents are derived from bagKg."
            ),
            "configured_annual_nutrient_comparison.nutrient_cards": (
                "Computed recognized fertilizer-history nutrient supply from dosePerTree × annual frequency "
                "× detected tree count, divided by farm area, compared with baseline; P2O5, K2O, and MgO "
                "label quantities are converted to elemental P, K, and Mg."
            ),
            "configured_annual_nutrient_comparison.measured_soil_leaf_status": (
                "Optional measured results; null means no test data was entered."
            ),
            "configured_annual_nutrient_comparison.standalone_field_top_up_options": (
                "Calculated from the baseline shortfall and configured product label fractions, oxide conversion "
                "factors, and bag sizes."
            ),
            "zone_recommendations.zone, boost, and nutrient values": (
                "Generated by image detection, tree-to-zone grouping, VARI summaries, confidence, zone-area "
                "configuration, and nutrient boost formula."
            ),
            "zone_recommendations.image_bbox_px and images": (
                "Pixel bounding boxes and crop paths generated from the input image; pixel boxes are not geospatial map boundaries."
            ),
            "zone_recommendations.map_boundary_geojson": (
                "Optional manually supplied zone boundaries; null if none was supplied."
            ),
            "zone_recommendations.product_supplement_options": (
                "Rows from fertilizer_product_mass. The reported Dolomite application is included as a base amount "
                "(dosePerTree × fertilizer frequency/year) in the Dolomite row; VARI boost product mass is added "
                "separately. Kieserite is a separate alternative Mg source and does not inherit the Dolomite base. "
                "Alternatives for the same nutrient are not additive."
            ),
            "zone_recommendations.application_status": (
                "Optional zone application status; null if no status was entered."
            ),
            "fertilizer_cost_inputs": (
                "Optional price per bag; missing prices are not inferred."
            ),
            "submission_month": (
                "The month value is read from the top-level month field in submission JSON."
            ),
            "fertilizer_application_history and harvest_history": (
                "fertilizerHistory and harvest objects are read directly from submission JSON. Recognized fertilizer "
                "products with dosePerTree are included in the annual nutrient supply calculation; unsupported or "
                "incomplete product entries are preserved but listed as excluded."
            ),
        },
    }

