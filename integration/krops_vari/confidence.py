"""Zone confidence (team decisions, 23 Oct 2026): tree factor × leaf factor × the flight confidence.

    tree factor = min(1, sqrt(n / zone_confidence_full_trees))
    leaf factor = min(1, leaf ratio / zone_leaf_pixel_full_ratio),
                  leaf ratio = zone leaf px ÷ (n × the photo's median leaf px per tree)
    zone factor = tree factor × leaf factor; under zone_reference_min_factor the zone doesn't set P95/P25
    confidence  = clamp(flight × zone factor, zone_confidence_floor, 1.0)

A flag never withholds a dose; it says the result may be less accurate. The values are GUESSes until calibrated.

The flight gate (team decision, 23 Oct 2026: flag only): a flight labelled low (L6 "rendah", under 0.5) or with an
input still unknown (L6 final: false) flags every zone, with L6's reason codes, and still gets its dose.
"""
import numpy as np

LOW_FLIGHT_LABEL = "rendah"  # L6's label under 0.5 (l6_vari_confidence.LABELS)


def flight_gate_flag(flight_result):
    """The flag text for a low or incomplete flight, or None. flight_result is the L6 result (or None)."""
    if not flight_result:
        return None
    problems = []
    if flight_result.get("label") == LOW_FLIGHT_LABEL:
        problems.append(f"low flight confidence {flight_result['confidence']:.2f}")
    if flight_result.get("final") is False:
        problems.append("some flight weather unknown (L6 not final)")
    if not problems:
        return None
    codes = [r["code"] for r in flight_result.get("reasons", [])]
    return "; ".join(problems) + (f" ({', '.join(codes)})" if codes else "")


def zone_confidence(zone_json, trees, image_size, settings, flight_confidence=None, flight_result=None):
    """({zone_id: confidence}, [reference zone ids], [run flags]); each zone gets zone["confidence"].

    Without a flight confidence (L6), settings.default_zone_confidence stands in and every zone is flagged. With the
    L6 result, a low or incomplete flight flags every zone too (flight_gate_flag), whether or not it is multiplied in.
    """
    gate_flag = flight_gate_flag(flight_result)
    width, height = image_size
    leaf_per_tree = [t["vari"]["vegetation_pixel_count"] for t in trees if t["vari"]["vegetation_pixel_count"]]
    median_leaf = float(np.median(leaf_per_tree)) if leaf_per_tree else 0.0
    flight = settings.default_zone_confidence if flight_confidence is None else float(flight_confidence)
    flight_source = ("DEFAULT_ZONE_CONFIDENCE, standing in: flight confidence not computed"
                     if flight_confidence is None else "flight confidence (L6)")
    margin, floor = settings.border_margin_px, settings.zone_confidence_floor
    by_id, reference_ids = {}, []
    for zone in zone_json:
        n = zone["tree_count"]
        tree_factor = min(1.0, float(np.sqrt(n / settings.zone_confidence_full_trees))) if n > 0 else 0.0
        leaf_ratio = zone["vari"]["vegetation_pixel_count"] / (n * median_leaf) if n > 0 and median_leaf > 0 else 0.0
        leaf_factor = min(1.0, leaf_ratio / settings.zone_leaf_pixel_full_ratio)
        zone_factor = tree_factor * leaf_factor
        product = flight * zone_factor
        confidence = min(1.0, max(floor, product))
        in_reference = (zone_factor >= settings.zone_reference_min_factor
                        and zone["vari"].get("mean_vari") is not None)
        box = zone["bbox"]
        touches_edge = (box["x1"] <= margin or box["y1"] <= margin
                        or box["x2"] >= width - margin or box["y2"] >= height - margin)
        flags = []
        if flight_confidence is None:
            flags.append("flight confidence not computed; DEFAULT_ZONE_CONFIDENCE stands in")
        if gate_flag:
            flags.append(gate_flag)
        if tree_factor < 1.0:
            flags.append(f"{n} tree(s): tree factor {tree_factor:.2f}")
        if leaf_factor < 1.0:
            flags.append(f"few leaf pixels for its trees (ratio {leaf_ratio:.2f})")
        if not in_reference:
            flags.append(f"not a P95/P25 reference zone (zone factor {zone_factor:.2f} < "
                         f"{settings.zone_reference_min_factor})")
        if product < floor:
            flags.append(f"confidence {product:.3f} raised to the floor {floor}")
        if touches_edge:
            flags.append("touches the photo edge (recorded only)")
        zone["confidence"] = {
            "tree_factor": round(tree_factor, 4),
            "leaf_pixel_ratio": round(leaf_ratio, 4),
            "leaf_pixel_factor": round(leaf_factor, 4),
            "zone_factor": round(zone_factor, 4),
            "flight_confidence": flight,
            "flight_confidence_source": flight_source,
            "confidence": round(confidence, 4),
            "reference_zone": in_reference,
            "touches_photo_edge": touches_edge,
            "may_be_less_accurate": flight_confidence is None or gate_flag is not None or zone_factor < 1.0,
            "flags": flags,
        }
        by_id[zone["zone_id"]] = confidence
        if in_reference:
            reference_ids.append(zone["zone_id"])
    run_flags = []
    if flight_confidence is None:
        run_flags.append("flight confidence (L6) not computed: every zone uses DEFAULT_ZONE_CONFIDENCE as its flight part")
    if gate_flag:
        run_flags.append(f"flight gate: {gate_flag}; every zone flagged, doses still given")
    if not reference_ids:
        run_flags.append("no zone qualifies as a reference: P95/P25 fall back to every valid zone")
    elif len(reference_ids) < settings.min_reference_zones_suggested:
        run_flags.append(f"P95/P25 from {len(reference_ids)} reference zone(s), under the suggested "
                         f"{settings.min_reference_zones_suggested} (proposal, not decided)")
    return by_id, reference_ids, run_flags
