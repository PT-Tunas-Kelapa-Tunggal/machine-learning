"""run_scan: one drone photo through every stage, returning a JSON-ready result.

    photo → leaf mask → palm detection → VARI → zones → zone VARI → zone confidence → boost → product mass

The result holds pixel boxes only, never coordinates.
"""
from datetime import datetime, timezone
from pathlib import Path

import cv2

from . import __version__
from .boost import add_zone_boost_and_fertilizer, product_mass_rows
from .confidence import zone_confidence
from .config import Settings
from .detection import Detector
from .segmentation import segment_vegetation_exg
from .vari import stats_for_box, vari_map
from .zoning import group_trees_into_zones, zone_records

SCHEMA_VERSION = 1


def read_photo(photo):
    """BGR image from a path, or the array itself."""
    if isinstance(photo, (str, Path)):
        image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {Path(photo).name}")
        return image
    return photo


def analyse_photo(image_bgr, settings, detector, flight_confidence=None):
    """Stages 1–5 and the zone confidence, boost and product mass. Returns (zone_json, products, run_flags, counts)."""
    height, width = image_bgr.shape[:2]
    exg_mask, segmented_bgr = segment_vegetation_exg(image_bgr)
    trees, skipped = detector.detect(cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB), settings)
    vmap = vari_map(image_bgr)
    for tree in trees:
        tree["vari"] = stats_for_box(vmap, exg_mask, tree["bbox"])
    zone_json = zone_records(group_trees_into_zones(trees, settings, width, height), vmap, exg_mask, width, height)
    computed, reference_ids, run_flags = zone_confidence(zone_json, trees, (width, height), settings, flight_confidence)
    confidence_by_id = settings.zone_confidence_by_id or computed
    if settings.zone_confidence_by_id:
        run_flags.append("zone confidence from settings.zone_confidence_by_id, not computed")
    add_zone_boost_and_fertilizer(zone_json, settings, confidence_by_id, reference_zone_ids=reference_ids)
    counts = {"width": width, "height": height, "trees": len(trees), "trees_dropped_partial": skipped}
    return zone_json, product_mass_rows(zone_json, settings), run_flags, counts


def summary(zone_json):
    """The run-wide part of the recommendation (the same on every zone)."""
    if not zone_json:
        return None
    rec = zone_json[0]["recommendation"]
    return {k: rec[k] for k in ("reference_zone_ids", "reference_fallback_all_zones", "p25_zone_mean_vari",
                                "reference_zone_mean_vari", "p95_p25_vari_gap", "boost_mode", "blanket_vari_gap_cutoff")}


def run_scan(photo, settings=None, *, detector=None, scan_id=None, flight_confidence=None, computed_at=None):
    """The full result for one photo. Pass a loaded `detector` to reuse the model across scans (the API does)."""
    settings = settings or Settings()
    detector = detector or Detector(settings.model_path)
    zone_json, products, run_flags, counts = analyse_photo(read_photo(photo), settings, detector, flight_confidence)
    if not zone_json:
        run_flags.append("no trees detected: no zones, no boost")
    return {
        "schema": SCHEMA_VERSION,
        "package_version": __version__,
        "scan_id": scan_id,
        "computed_at": computed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": {"name": Path(photo).name if isinstance(photo, (str, Path)) else None, **counts},
        "summary": summary(zone_json),
        "zones": zone_json,
        "product_mass": products,
        "run_flags": run_flags,
        "settings": {k: v for k, v in settings.to_dict().items() if k != "model_path"},
    }
