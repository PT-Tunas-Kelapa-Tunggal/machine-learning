"""run_scan: one drone photo through every stage, returning a JSON-ready result.

    photo → leaf mask → palm detection → VARI → zones → zone VARI → zone confidence → boost → product mass
    plot + weather → (L6 flight confidence) → L5 P30, L8 month gate and safe days → the spreading plan

The result holds pixel boxes only, never coordinates.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from . import __version__
from .boost import add_zone_boost_and_fertilizer, product_mass_rows
from .confidence import zone_confidence
from .config import Settings
from .detection import Detector
from .segmentation import segment_vegetation_exg
from .spreading import spreading_plan
from .vari import stats_for_box, vari_map
from .zoning import group_trees_into_zones, zone_records

SCHEMA_VERSION = 1
import os

if "PYTHONTZPATH" not in os.environ and Path("/usr/share/zoneinfo").is_dir():
    os.environ["PYTHONTZPATH"] = "/usr/share/zoneinfo"

try:
    TZ = ZoneInfo("Asia/Jakarta")
except Exception:
    import zoneinfo
    zoneinfo.reset_tzpath(to=["/usr/share/zoneinfo"])
    TZ = ZoneInfo("Asia/Jakarta")

log = logging.getLogger("krops_vari.pipeline")


def stage_a_table(climate):
    """{month: {"avg_mm", "in_band_share", "gate"}} with int months, from a climate.json."""
    return {int(m): row for m, row in climate["stage_a"].items()}


def l8_decision(forecast, stage_a, today):
    from .weather import rules  # weather/ is only imported when weather is used

    return rules.l8_decision(forecast, stage_a, today)


def l6_flight_day(flight):
    """The local (WIB) date of take-off."""
    start = datetime.fromisoformat(flight["flight_start"].replace("Z", "+00:00"))
    return start.astimezone(TZ).date()


def read_photo(photo):
    """BGR image from a path, or the array itself."""
    if isinstance(photo, (str, Path)):
        image = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {Path(photo).name}")
        return image
    return photo


def analyse_photo(image_bgr, settings, detector, flight_confidence=None, flight_result=None):
    """Stages 1–5 and the zone confidence, boost and product mass. Returns (zone_json, products, run_flags, counts).

    flight_confidence is multiplied in; flight_result (the L6 result) raises the flight gate flag when low or
    incomplete."""
    height, width = image_bgr.shape[:2]
    exg_mask, segmented_bgr = segment_vegetation_exg(image_bgr)
    trees, skipped = detector.detect(cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB), settings)
    vmap = vari_map(image_bgr)
    for tree in trees:
        tree["vari"] = stats_for_box(vmap, exg_mask, tree["bbox"])
    zone_json = zone_records(group_trees_into_zones(trees, settings, width, height), vmap, exg_mask, width, height)
    computed, reference_ids, run_flags = zone_confidence(zone_json, trees, (width, height), settings, flight_confidence,
                                                         flight_result)
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


def _weather_failed(run_flags, what, error):
    """Record a weather failure by its type only: the request URL in the message carries the coordinates."""
    run_flags.append(f"{what} unavailable ({type(error).__name__})")
    log.warning("%s unavailable (%s)", what, type(error).__name__)


def run_scan(photo, settings=None, *, detector=None, scan_id=None, plot=None, flight=None, weather=None, today=None,
             computed_at=None):
    """The full result for one photo. Pass a loaded `detector` to reuse the model across scans (the API does).

    With `weather` (a WeatherClient) and `plot` ({"plot_id", "lat", "lon"}, kept on the server) the result adds the
    spreading plan; with `flight` ({"flight_start", "flight_end"} in UTC) also the L6 flight confidence, which then
    replaces DEFAULT_ZONE_CONFIDENCE as the flight part of every zone's confidence. Without them the result equals
    integration.ipynb's. A weather failure never stops the scan: it becomes a run flag.
    """
    settings = settings or Settings()
    detector = detector or Detector(settings.model_path)
    today = today or datetime.now(TZ).date()
    run_flags, climate, flight_result = [], None, None
    if weather is not None and plot is not None:
        lat, lon = float(plot["lat"]), float(plot["lon"])
        try:
            climate = weather.climate(lat, lon, today)
        except Exception as error:  # noqa: BLE001 – reported, never fatal
            _weather_failed(run_flags, "climate (NASA POWER 1991–2020)", error)
        if climate is not None and flight:
            try:
                manifest = {"scan_id": scan_id or "scan", "plot_id": plot.get("plot_id"), **flight}
                flight_result = weather.flight_confidence(manifest, plot, climate["normals_mm"])
            except Exception as error:  # noqa: BLE001
                _weather_failed(run_flags, "flight confidence (L6)", error)
    elif flight:
        run_flags.append("flight times given without a plot and weather: flight confidence not computed")
    flight_confidence = (flight_result["confidence"]
                         if flight_result is not None and settings.use_flight_confidence else None)

    zone_json, products, photo_flags, counts = analyse_photo(read_photo(photo), settings, detector, flight_confidence,
                                                             flight_result)
    run_flags += photo_flags
    if not zone_json:
        run_flags.append("no trees detected: no zones, no boost")

    plan = None
    if climate is not None:
        try:
            p30 = None
            if flight_result is not None and l6_flight_day(flight) == today:
                p30 = flight_result["factors"]["water"]["p30"]  # the same L5 record: no second history fetch
            if p30 is None:
                p30 = weather.p30(lat, lon, today, climate["normals_mm"])
            stage_a = stage_a_table(climate)
            l8 = l8_decision(weather.forecast(lat, lon, today), stage_a, today)
            plan = spreading_plan(products, stage_a, l8, p30, today, settings)
        except Exception as error:  # noqa: BLE001
            _weather_failed(run_flags, "spreading plan (L5/L8)", error)
    return {
        "schema": SCHEMA_VERSION,
        "package_version": __version__,
        "scan_id": scan_id,
        "computed_at": computed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": {"name": Path(photo).name if isinstance(photo, (str, Path)) else None, **counts},
        "summary": summary(zone_json),
        "zones": zone_json,
        "product_mass": products,
        "flight_confidence": flight_result,
        "spreading": plan,
        "run_flags": run_flags,
        "settings": {k: v for k, v in settings.to_dict().items() if k != "model_path"},
    }
