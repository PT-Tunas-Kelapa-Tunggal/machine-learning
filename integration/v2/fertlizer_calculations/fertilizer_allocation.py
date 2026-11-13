#!/usr/bin/env python3
"""Oil-palm zoning, VARI, and fertilizer analysis converted from new_readable.ipynb.

Run from the project root and pass the full submission JSON as one option value, plus the image path:
    python outputs/new_readable.py --user-request-submission-json '{"farm":{"areaHa":1.0},"fertilizerHistory":{"freq":2,"fertilizers":[]}}' --image webodm_new_2nd_flight.jpeg

The CoreML model path is configured by the MODEL_PATH constant below; it is not a CLI argument.
"""

import argparse as _cli_argparse
import json as _cli_json

_cli_parser = _cli_argparse.ArgumentParser(description="Run the oil-palm VARI and fertilizer analysis.")
_cli_parser.add_argument(
    "--user-request-submission-json",
    required=True,
    help="The complete user submission as a single JSON string (not a path to a JSON file).",
)
_cli_parser.add_argument(
    "--image",
    required=True,
    help="Path to the input RGB/WebODM image.",
)
_cli_parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("outputs"),
    help="Destination directory for output files (default: outputs).",
)
_cli_args = _cli_parser.parse_args()
try:
    _USER_SUBMISSION_DATA = _cli_json.loads(_cli_args.user_request_submission_json)
except _cli_json.JSONDecodeError as _json_error:
    raise SystemExit(f"Invalid submission JSON string: {_json_error}") from _json_error
if not isinstance(_USER_SUBMISSION_DATA, dict):
    raise SystemExit("The submission JSON argument must decode to a JSON object.")


# # Oil-palm zoning and VARI analysis
#
# This notebook processes a WebODM RGB image into tree detections, zones, VARI summaries, fertilizer calculations, and output images/files.
#
# Run the notebook **from top to bottom**. Set the image/model paths and calculation settings in the configuration cell first. Generated images, JSON/CSV files, and zone-crop folders are saved under `outputs/`. Each numbered section below has its own code cell so you can follow the pipeline stage by stage.
# **Flight confidence:** with actual flight times and farm coordinates configured, the notebook scores sun, cloud, pre-flight rain, and 30-day rainfall; that score feeds the existing zone/field confidence calculation.


# ## Imports


# ===== Notebook code cell 2 =====
import csv
import json
import math
import calendar
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vari_confidence")
import os
import time
from datetime import date, datetime, timedelta, timezone
from statistics import median
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
import html
from pathlib import Path
import coremltools as ct
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    from IPython.display import HTML
except ImportError:
    class HTML:
        def __init__(self, data=None, *args, **kwargs):
            self.data = data

# Standalone runs save plots/images to disk but never open viewer windows.
def display(*_args, **_kwargs):
    return None

# ## Configuration
#
# Edit the input paths, output paths, detection settings, zoning rules, and fertilizer settings here.


# ===== Notebook code cell 4 =====
# ------------------------- Configuration -------------------------
# All generated files (images, tables, JSON, and crop folders) go here.
OUTPUT_DIR = Path(_cli_args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = _cli_args.image  # Input RGB image supplied with --image.
# IMAGE_PATH = "webodm_last_image_1st_flight.jpeg"
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "krops_model_v0.2_yolo26l.mlpackage"
MODEL_PATH = str(_DEFAULT_MODEL_PATH if _DEFAULT_MODEL_PATH.exists() else Path("krops_model_v0.2_yolo26l.mlpackage"))  # CoreML object-detection model used to detect palm trees.
# MODEL_PATH = "yolo26m-krops-v0.1.mlpackage"  # CoreML object-detection model used to detect palm trees.
OUTPUT_JSON = str(OUTPUT_DIR / "oil_palm_zones_vari.json")  # JSON with metadata, zone/tree statistics, and recommendations.
OUTPUT_FERTILIZER_HISTORY_JSON = str(OUTPUT_DIR / "user_fertilizer_summary.json")  # Coordinates and annual total product mass from submitted fertilizer history.
OUTPUT_DECIMALS = 6  # Shared decimal precision for fertilizer and nutrient outputs.
OUTPUT_TREE_VARI_HEATMAP_DIR = str(OUTPUT_DIR / "oil_palm_tree_vari_heatmaps")  # One ExG-masked VARI heatmap crop per detected tree.
OUTPUT_TREE_VARI_HEATMAP_MONTAGE = str(OUTPUT_DIR / "oil_palm_tree_vari_heatmap_montage.png")  # Tiled preview of all per-tree heatmaps.
OUTPUT_FIELD_TREE_VARI_QUARTILE = str(OUTPUT_DIR / "oil_palm_field_tree_vari_quartiles.jpg")  # Full-field tree overlay colored by each tree's VARI quartile.

# Submission is provided as one JSON-string command-line argument (not loaded from a file).
UI_SUBMISSION_DATA = _USER_SUBMISSION_DATA
UI_FARM_INPUT = UI_SUBMISSION_DATA.get("farm", {})
FERTILIZER_BUDGET_RP = UI_SUBMISSION_DATA.get("fertilizerBudgetRp")  # Optional annual fertilizer budget in Indonesian Rupiah.

# Screen-entered values are loaded from the submission JSON; image/map fields remain optional.
PULSE_CHECK_FARM_PROFILE = {
    "name": UI_FARM_INPUT.get("name"),
    "location_label": UI_FARM_INPUT.get("location_label"),
    "area_ha": UI_FARM_INPUT.get("areaHa"),
    "perimeter_m": UI_FARM_INPUT.get("perimeterM"),
    "latitude": UI_FARM_INPUT.get("latitude"),
    "longitude": UI_FARM_INPUT.get("longitude"),
    "planting_year": UI_FARM_INPUT.get("year"),
    "planting_month": UI_FARM_INPUT.get("plantingMonth"),
    "soil_type": UI_FARM_INPUT.get("soil"),
    "farm_boundary_geojson": UI_FARM_INPUT.get("farmBoundaryGeojson"),
    "review_month": UI_SUBMISSION_DATA.get("month"),
    "fertilizer_budget_rp": FERTILIZER_BUDGET_RP,
}
PULSE_CHECK_ZONE_BOUNDARIES_GEOJSON = {}  # Optional {"Z1": GeoJSON geometry, ...}; image bboxes are not map polygons.
PULSE_CHECK_NUTRIENT_MEASUREMENTS = None  # Optional measured soil/leaf results; not part of the current screens.
PULSE_CHECK_FERTILIZER_HISTORY = UI_SUBMISSION_DATA.get("fertilizerHistory")
PULSE_CHECK_HARVEST_HISTORY = UI_SUBMISSION_DATA.get("harvest")
PULSE_CHECK_ZONE_APPLICATION_STATUS = {}  # Optional {"Z1": {"status": "done", "completed_at": ...}, ...}.
OUTPUT_ZONE_CSV = str(OUTPUT_DIR / "oil_palm_zone_vari_summary.csv")  # CSV with one row per zone and its VARI/fertilizer values.
OUTPUT_PRODUCT_MASS_CSV = str(OUTPUT_DIR / "oil_palm_zone_fertilizer_product_mass.csv")  # CSV converting elemental nutrient totals to product mass.
OUTPUT_EXG_MASK = str(OUTPUT_DIR / "oil_palm_exg_mask.png")  # Binary vegetation mask produced by ExG thresholding.
OUTPUT_EXG_SEGMENTED = str(OUTPUT_DIR / "oil_palm_exg_segmented.jpg")  # RGB image showing pixels retained by the ExG vegetation mask.
OUTPUT_COREML_ANNOTATED = str(OUTPUT_DIR / "oil_palm_coreml_detections.jpg")  # ExG-segmented image with detected palm boxes and labels.
OUTPUT_ZONE_ANNOTATED = str(OUTPUT_DIR / "oil_palm_zones_only.jpg")  # Black background with detected tree pixels colored by zone.
OUTPUT_ANNOTATED_IMAGE = str(OUTPUT_DIR / "oil_palm_zones_annotated.jpg")  # Original image with zone colors, tree labels, and boost overlays.
OUTPUT_PLAIN_ZONE_ANNOTATED = str(OUTPUT_DIR / "oil_palm_zones_plain.jpg")  # Zone-colored tree masks with small zone IDs/counts only; no centered boost labels.
OUTPUT_PLAIN_ZONE_WITH_SEPARATOR = str(OUTPUT_DIR / "oil_palm_zones_plain_with_separator.jpg")  # Plain zone overlay with shared white zone boundaries.
OUTPUT_WHITE_ZONE_ANNOTATED = str(OUTPUT_DIR / "oil_palm_zones_white_separators.jpg")  # Original image with white zone boundaries and no zone-color fill.
OUTPUT_WHITE_ZONE_NUMBERS = str(OUTPUT_DIR / "oil_palm_zones_white_separators_with_zone_numbers.jpg")  # White separators with Z1/Z2/... labels at zone centers.
WHITE_ZONE_BOUNDARY_WIDTH_PX = 9  # Thickness of the white-only shared zone boundaries in the separator image.
OUTPUT_ZONE_BASE_CROPS_DIR = str(OUTPUT_DIR / "oil_palm_zone_base_crops")  # Folder for original-image crops, one per zone.
OUTPUT_ZONE_EXG_CROPS_DIR = str(OUTPUT_DIR / "oil_palm_zone_exg_crops")  # Folder for ExG-segmented image crops, one per zone.
OUTPUT_PIPELINE_PLOT = str(OUTPUT_DIR / "oil_palm_pipeline_stages.jpg")  # Combined visualization of the analysis pipeline outputs.
OUTPUT_VARI_DISTRIBUTION = str(OUTPUT_DIR / "oil_palm_vari_distribution.png")  # Zone mean VARI values with the P25–P95 range highlighted.
CONF_THRESHOLD = 0.10  # Temporary recall-focused cutoff; inspect lower-confidence detections for false positives.
IOU_THRESHOLD = 0.45  # Non-maximum suppression overlap threshold for removing duplicate detections.
MODEL_INPUT_SIZE = 640  # Width and height, in pixels, of the square image passed to the CoreML model.
ZONE_GRID_ROWS = 3  # Target number of palm rows grouped into one zone band.
ZONE_GRID_COLS = 3  # Target number of palms per row within a zone; nominal capacity is rows × columns.
TREES_PER_ZONE = ZONE_GRID_ROWS * ZONE_GRID_COLS  # Nominal tree capacity used to mark zones as undersized.
MIN_TREES_TO_KEEP_ZONE = 2  # Merge a one-tree spatial zone into a directly adjacent zone when possible.
ROW_TOLERANCE_MULTIPLIER = 0.75  # Same-row center tolerance as a fraction of median detected tree height.
MAX_ZONE_DISTANCE_MULTIPLIER = 3.0  # Maximum row/column grouping gap as a fraction of median tree height.
ZONE_PADDING_MULTIPLIER = 0.15  # Padding around each zone bounding box as a fraction of median tree height.
MIN_TREE_COMPLETENESS = 0.70  # Keep a detection only if its width, height, and area are each >=70% of their medians.
MAX_TREE_DIMENSION_MULTIPLIER = 4.0  # Reject extreme boxes wider or taller than 4× the robust typical tree box.
MAX_TREE_AREA_MULTIPLIER = 20.0  # Reject extreme boxes with area greater than 20× the robust typical tree-box area.
BORDER_MARGIN_PX = 5  # Exclude detections whose bounding boxes come within 5 pixels of any image edge.
SHOW_PLOTS = True  # Display the generated pipeline figure interactively after saving it.
# If the spread between zone-mean VARI P95 and P25 is at or below this
# configured absolute VARI gap, use the P25-derived boost across all zones.
# Tune this threshold for the imagery and management policy; set to 0.0 to
# effectively disable the blanket mode for a nonzero spread.
MIN_P95_P25_VARI_GAP_FOR_BLANKET_BOOST = 0.03
# Maximum annual elemental Mg and N nutrient rates per hectare/year, supplied
# by the user. These are nutrient quantities, independent of product choice.
# Zone targets are B_i × the corresponding nutrient rate.
MAX_NUTRIENT_KG_HA_PER_YEAR = {
    "Mg": 6.0,
    "N": 44.0,
}
FERTILIZER_NUTRIENT_BY_PRODUCT = {
    "kieserite_Mg_expensive": "Mg", "dolomite_Mg_cheaper": "Mg",
    "ZA_N_cheaper": "N", "urea_N_expensive": "N",
}
# Product label fractions in the form printed on the bag. Mg fertilizers use
# MgO labels; N fertilizers use elemental N labels. Set each product guarantee
# here once. Product kg is calculated in two steps: nutrient -> label basis -> product.
FERTILIZER_PRODUCT_LABEL_FRACTIONS = {
    "kieserite_Mg_expensive": {"basis": "MgO", "fraction": 0.255},
    "dolomite_Mg_cheaper": {"basis": "MgO", "fraction": 0.22},
    "ZA_N_cheaper": {"basis": "N", "fraction": 0.21},
    "urea_N_expensive": {"basis": "N", "fraction": 0.46},
}
# Product bag sizes used only to convert the zone product quantities into bag equivalents.
# Update these values if the available product packaging differs.
FERTILIZER_BAG_SIZE_KG = {
    "kieserite_Mg_expensive": 50.0,
    "dolomite_Mg_cheaper": 50.0,
    "ZA_N_cheaper": 50.0,
    "urea_N_expensive": 50.0,
}
MGO_TO_MG_FACTOR = 24.305 / 40.304  # Elemental Mg mass contained in one unit of MgO mass.
MG_TO_MGO_FACTOR = 40.304 / 24.305  # MgO mass equivalent to one unit of elemental Mg mass.
# Derived elemental nutrient per kg product fractions. These are calculated
# from the single label configuration above and should not be edited separately.
FERTILIZER_PRODUCT_NUTRIENT_FRACTIONS = {
    product: config["fraction"] * (MGO_TO_MG_FACTOR if config["basis"] == "MgO" else 1.0)
    for product, config in FERTILIZER_PRODUCT_LABEL_FRACTIONS.items()
}
# Annual base product rates, kept separate from VARI-derived supplements.
# These fixed defaults are used regardless of Kieserite or Dolomite entries in
# the submitted fertilizer history. Units are kg product/palm/year.
DEFAULT_KIESERITE_BASE_KG_PER_PALM_PER_YEAR = 1.0
DEFAULT_DOLOMITE_BASE_KG_PER_PALM_PER_YEAR = 1.0
KIESERITE_BASE_PRODUCT_KG_PER_PALM_PER_YEAR = DEFAULT_KIESERITE_BASE_KG_PER_PALM_PER_YEAR
KIESERITE_BASE_RATE_SOURCE = "configured default; user fertilizer history ignored"
DOLOMITE_BASE_PRODUCT_KG_PER_PALM_PER_YEAR = DEFAULT_DOLOMITE_BASE_KG_PER_PALM_PER_YEAR
DOLOMITE_BASE_RATE_SOURCE = "configured default; user fertilizer history ignored"
ZA_BASE_PRODUCT_KG_PER_PALM_PER_YEAR = None
UREA_BASE_PRODUCT_KG_PER_PALM_PER_YEAR = None
FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR = {
    "kieserite_Mg_expensive": KIESERITE_BASE_PRODUCT_KG_PER_PALM_PER_YEAR,
    "dolomite_Mg_cheaper": DOLOMITE_BASE_PRODUCT_KG_PER_PALM_PER_YEAR,
    "ZA_N_cheaper": ZA_BASE_PRODUCT_KG_PER_PALM_PER_YEAR,
    "urea_N_expensive": UREA_BASE_PRODUCT_KG_PER_PALM_PER_YEAR,
}
# Estimated retail prices, in Indonesian rupiah per configured fertilizer bag.
# Prices supplied for this project assume 50 kg bags; adjust bag sizes above if needed.
FERTILIZER_PRICE_PER_BAG_RP = {
    "kieserite_Mg_expensive": None,  # No price supplied.
    "dolomite_Mg_cheaper": 100_000,
    "ZA_N_cheaper": None,           # No price supplied.
    "urea_N_expensive": 500_000,
    "16-16-16": 890_000,
    "16-16-16-0": 890_000,
    "15-15-15": 740_000,
    "15-15-15-0": 740_000,
    "13-6-27-4": 550_000,
    "13-8-27-4": 550_000,
    "12-6-22-3": 630_000,
    "Urea": 500_000,
    "urea": 500_000,
    "Dolomite": 100_000,
    "dolomite": 100_000,
}
# Existing tables use product cost per kg, so derive it from bag price ÷ bag size.
FERTILIZER_PRICE_PER_KG = {
    product: (price / FERTILIZER_BAG_SIZE_KG[product]
              if price is not None and FERTILIZER_BAG_SIZE_KG.get(product) else None)
    for product, price in FERTILIZER_PRICE_PER_BAG_RP.items()
    if product in FERTILIZER_BAG_SIZE_KG
}
# Optional measured ground areas per zone in hectares, supplied directly or by JSON.
# If omitted, the active estimate uses tree spacing × tree spacing for each detected palm;
# TOTAL_FARM_AREA_HA is retained for field-wide comparisons, not zone-area allocation.
ZONE_AREA_HA_BY_ID = {}
ZONE_AREA_JSON_PATH = None
UNIFORM_ZONE_AREA_HA = None  # Optional override applied to every zone
TOTAL_FARM_AREA_HA = float(UI_FARM_INPUT.get("areaHa") or 1.0)  # Use the farm area entered in the profile JSON; default to 1 ha if blank.
# Current zone-area estimate: assume 9 m × 9 m ground spacing per detected palm.
# Each detected tree therefore contributes 81 m² = 0.0081 ha to its zone.
DEFAULT_TREE_SPACING_M = 9.0
USE_SPACING_BASED_ZONE_AREA_DEFAULT = True
# Optional manual final-confidence overrides by zone ID; leave empty to use the
# confidence calculation below. Each configured value must be in [0.10, 1.0].
ZONE_CONFIDENCE_BY_ID = {}
# Weather-derived flight confidence requires the actual flight window and farm coordinates.
# Enter timezone-aware local datetimes (WIB) and latitude/longitude here, or leave None to use the existing fallback.
WIB = ZoneInfo("Asia/Jakarta")
FLIGHT_START_WIB = None  # Example: datetime(2026, 11, 6, 10, 0, tzinfo=WIB); enter actual take-off time.
FLIGHT_END_WIB = None    # Example: datetime(2026, 11, 6, 10, 20, tzinfo=WIB); enter actual landing time.
FARM_LATITUDE = UI_FARM_INPUT.get("latitude")  # Farm latitude in decimal degrees from user_submission_request.json.
FARM_LONGITUDE = UI_FARM_INPUT.get("longitude")  # Farm longitude in decimal degrees from user_submission_request.json.
WEATHER_CONFIDENCE_ENV_PATHS = (Path.cwd() / ".env", Path.cwd().parent / "weather" / ".env")
WEATHER_CONFIDENCE_OUTPUT = OUTPUT_DIR / "flight_confidence_factors.json"
FLIGHT_CONFIDENCE_OVERRIDE = None  # Optional manual override [0,1]; when set it takes precedence over weather.

# Flight-level confidence (L6), when assessed. Until it is assessed, the
# default below stands in as the flight factor and the run/each zone is flagged.
FLIGHT_CONFIDENCE = None  # Filled from weather below when flight inputs are configured.
FLIGHT_CONFIDENCE_SOURCE = "unassessed fallback"
WEATHER_FLIGHT_CONFIDENCE = None
DEFAULT_ZONE_CONFIDENCE = 0.60
# Notebook confidence model (team proposal, values marked GUESS until calibrated).
ZONE_CONFIDENCE_FULL_TREES = 9  # Tree factor reaches 1.0 at this many palms.
ZONE_LEAF_PIXEL_FULL_RATIO = 0.70  # Leaf factor reaches 1.0 at this zone/photo leaf-pixel ratio.
ZONE_REFERENCE_MIN_FACTOR = 0.70  # Zones below this zone factor do not set P25/P95.
ZONE_CONFIDENCE_FLOOR = 0.10  # Keep nonzero boost confidence at or above this floor.
MIN_REFERENCE_ZONES_SUGGESTED = 5  # Emit a metadata note when fewer valid reference zones qualify.
# -----------------------------------------------------------------

# ## Shared analysis functions
#
# These helper functions handle detections, tree filtering, zoning, and pixel-coordinate records.


# ===== Notebook code cell 6 =====
def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.45):
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    selected = []
    while len(order) > 0:
        current = int(order[0])
        selected.append(current)
        order = np.asarray([
            int(j) for j in order[1:]
            if iou(boxes[current], boxes[int(j)]) <= iou_threshold
        ], dtype=int)
    return selected


def segment_vegetation_exg(image_bgr):
    """Create an ExG/Otsu mask. This is a heuristic vegetation mask, not a crown model."""
    b, g, r = cv2.split(image_bgr)
    exg = 2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return mask, segmented


def parse_coreml_predictions(model, rgb_image):
    """Parse the [1, 5, N] / [1, N, 5] single-class output assumed by the source script.
    Expected prediction rows: [cx, cy, width, height, confidence] in 640x640 pixels.
    """
    input_names = list(model.input_description)
    if not input_names:
        raise RuntimeError("The CoreML model has no declared input.")
    pil_image = Image.fromarray(rgb_image).convert("RGB").resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    )
    prediction = model.predict({input_names[0]: pil_image})
    output_key = list(prediction.keys())[0]
    raw = np.asarray(prediction[output_key])
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(f"Unexpected model output shape {raw.shape}; expected a 2D array after squeeze.")
    if raw.shape[0] == 5:
        rows = raw.T
    elif raw.shape[1] == 5:
        rows = raw
    else:
        raise ValueError(
            f"Model output shape {raw.shape} does not have a 5-value detection axis. "
            "Inspect the CoreML output schema and update parse_coreml_predictions()."
        )
    boxes, scores = [], []
    for row in rows:
        cx, cy, width, height, confidence = map(float, row[:5])
        if not np.isfinite([cx, cy, width, height, confidence]).all() or confidence < CONF_THRESHOLD:
            continue
        box = [
            np.clip((cx - width / 2) / MODEL_INPUT_SIZE, 0, 1),
            np.clip((cy - height / 2) / MODEL_INPUT_SIZE, 0, 1),
            np.clip((cx + width / 2) / MODEL_INPUT_SIZE, 0, 1),
            np.clip((cy + height / 2) / MODEL_INPUT_SIZE, 0, 1),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append([float(v) for v in box])
            scores.append(confidence)
    keep = nms(boxes, scores, IOU_THRESHOLD)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def normalized_box_to_pixels(box, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0, min(width, int(round(x1 * width)))),
        max(0, min(height, int(round(y1 * height)))),
        max(0, min(width, int(round(x2 * width)))),
        max(0, min(height, int(round(y2 * height)))),
    ]


def stats_for_box(vari_map, exg_mask, bbox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        values = np.asarray([], dtype=np.float32)
    else:
        vari_crop = vari_map[y1:y2, x1:x2]
        mask_crop = exg_mask[y1:y2, x1:x2] > 0
        values = vari_crop[mask_crop & np.isfinite(vari_crop)]
    if values.size == 0:
        return {
            "vegetation_pixel_count": 0,
            "mean_vari": None,
            "median_vari": None,
            "std_vari": None,
            "mad_vari": None,
            "min_vari": None,
            "max_vari": None,
        }
    median = float(np.median(values))
    return {
        "vegetation_pixel_count": int(values.size),
        "mean_vari": float(np.mean(values)),
        "median_vari": median,
        "std_vari": float(np.std(values)),
        "mad_vari": float(np.median(np.abs(values - median))),
        "min_vari": float(np.min(values)),
        "max_vari": float(np.max(values)),
    }


def group_trees_into_zones(
    trees,
    grid_rows=3,
    grid_cols=3,
    row_tolerance_multiplier=0.75,
    max_zone_distance_multiplier=3.0,
    zone_padding_multiplier=0.15,
    image_width=None,
    image_height=None,
    ):
    """Group detected palms by approximate row bands and fixed spatial column blocks."""
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError("grid_rows and grid_cols must be at least 1.")
    if not trees:
        return []
    target_trees_per_zone = grid_rows * grid_cols
# 1. Estimate typical tree size
    heights = [max(1, tree["bbox"][3] - tree["bbox"][1]) for tree in trees]
    median_height = float(np.median(heights))
    row_tolerance = max(1.0, median_height * row_tolerance_multiplier)
    max_zone_distance = max(1.0, median_height * max_zone_distance_multiplier)
    padding = max(0, int(median_height * zone_padding_multiplier))
# 2. Cluster trees into approximate horizontal rows
    ordered = sorted(trees, key=lambda tree: (tree["center"][1], tree["center"][0]))
    rows = []
    for tree in ordered:
        cy = tree["center"][1]
        best_row = None
        best_distance = float("inf")
        for row in rows:
            row_center = float(np.median([item["center"][1] for item in row]))
            distance = abs(cy - row_center)
            if distance <= row_tolerance and distance < best_distance:
                best_row = row
                best_distance = distance
        if best_row is None:
            rows.append([tree])
        else:
            best_row.append(tree)
    rows.sort(key=lambda row: float(np.median([tree["center"][1] for tree in row])))
    for row in rows:
        row.sort(key=lambda tree: tree["center"][0])
# 3. Group consecutive rows into bands of up to `grid_rows` (3 rows)
    row_bands = []
    current_band = []
    for row in rows:
        if not current_band:
            current_band.append(row)
        else:
            prev_row_y = float(np.median([t["center"][1] for t in current_band[-1]]))
            curr_row_y = float(np.median([t["center"][1] for t in row]))
            if len(current_band) >= grid_rows or (curr_row_y - prev_row_y) > max_zone_distance:
                row_bands.append(current_band)
                current_band = [row]
            else:
                current_band.append(row)
    if current_band:
        row_bands.append(current_band)
# 4. Within each row band, assign trees to fixed spatial column blocks.
# Missing detections leave empty positions instead of shifting later trees.
    zones = []
    row_gaps = []
    for row in rows:
        row_x = sorted(float(tree["center"][0]) for tree in row)
        row_gaps.extend(
            gap for gap in np.diff(row_x)
            if gap > 0 and gap <= max_zone_distance
        )
    typical_column_spacing = float(np.median(row_gaps)) if row_gaps else max(1.0, median_height * 2.0)
    global_x_origin = min(float(tree["center"][0]) for tree in trees)
    spatial_block_width = max(1.0, typical_column_spacing * grid_cols)
    for band_idx, band in enumerate(row_bands, start=1):
        band_trees = [tree for row in band for tree in row]
# Assign by fixed x intervals, not by chunks or gaps in detected-tree counts.
        spatial_blocks = {}
        for tree in band_trees:
            block_id = int(np.floor((tree["center"][0] - global_x_origin) / spatial_block_width))
            spatial_blocks.setdefault(block_id, []).append(tree)
        for block_id in sorted(spatial_blocks):
            members = spatial_blocks[block_id]
            x1 = min(t["bbox"][0] for t in members) - padding
            y1 = min(t["bbox"][1] for t in members) - padding
            x2 = max(t["bbox"][2] for t in members) + padding
            y2 = max(t["bbox"][3] for t in members) + padding
            if image_width is not None:
                x1 = max(0, x1)
                x2 = min(image_width, x2)
            if image_height is not None:
                y1 = max(0, y1)
                y2 = min(image_height, y2)
            zones.append({
                "zone_id": len(zones) + 1,
                "approx_row": band_idx,
                "spatial_column_block": block_id,
                "trees": members,
                "bbox": [x1, y1, x2, y2],
                "is_undersized": len(members) < target_trees_per_zone,
            })
    return zones


def merge_small_adjacent_zones(zones, min_tree_count=MIN_TREES_TO_KEEP_ZONE):
    """Merge undersized zones into the nearest zone in an adjacent grid cell."""
    zones = list(zones)
    for source in sorted(zones, key=lambda item: (len(item["trees"]), item["zone_id"])):
        if source not in zones or len(source["trees"]) >= min_tree_count:
            continue
        source_row = source.get("approx_row")
        source_col = source.get("spatial_column_block")
        source_center = np.mean([tree["center"] for tree in source["trees"]], axis=0)
        candidates = []
        for target in zones:
            if target is source:
                continue
            row_delta = abs(target.get("approx_row", source_row) - source_row)
            col_delta = abs(target.get("spatial_column_block", source_col) - source_col)
            if row_delta <= 1 and col_delta <= 1:
                target_center = np.mean([tree["center"] for tree in target["trees"]], axis=0)
                distance = float(np.linalg.norm(source_center - target_center))
                candidates.append((distance, target))
        if not candidates:
            continue
        _, target = min(candidates, key=lambda item: item[0])
        target["trees"].extend(source["trees"])
        target["bbox"] = [
            min(target["bbox"][0], source["bbox"][0]),
            min(target["bbox"][1], source["bbox"][1]),
            max(target["bbox"][2], source["bbox"][2]),
            max(target["bbox"][3], source["bbox"][3]),
        ]
        target.setdefault("merged_from_zone_ids", []).extend(
            source.get("merged_from_zone_ids", [source["zone_id"]])
        )
        zones.remove(source)
    zones.sort(key=lambda item: (item.get("approx_row", 0), item.get("spatial_column_block", 0)))
    for new_zone_id, zone in enumerate(zones, start=1):
        zone["zone_id"] = new_zone_id
        zone["is_undersized"] = len(zone["trees"]) < TREES_PER_ZONE
    return zones


def pixel_bbox_record(bbox, image_width, image_height):
    x1, y1, x2, y2 = map(int, bbox)
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "normalized_xyxy": [
            x1 / image_width, y1 / image_height,
            x2 / image_width, y2 / image_height,
        ],
    }


def show_preview_table(title, headers, rows):
    """Render a compact HTML table so intermediate records are readable in Jupyter."""
    header_html = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    display(HTML(
        f"<h4>{html.escape(str(title))}</h4>"
        f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;white-space:nowrap'>"
        f"<thead><tr style='background:#263238;color:white'>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table></div>"
    ))

# ## Zone summary and nutrient calculations
#
# Functions for the annual zone table and its detailed nutrient calculation output.


# ===== Notebook code cell 8 =====
def print_zone_vari_table(zone_json):
    """Print annual VARI-derived elemental nutrient supplements by zone."""
    nutrients = [nutrient for nutrient, rate in MAX_NUTRIENT_KG_HA_PER_YEAR.items() if rate is not None]
    headers = ["Zone", "Trees", "Mean VARI", "Min VARI", "Min VARI review",
               "Zone factor", "Zone confidence", "Overall field confidence (applied)",
               "Qualifies for P25/P95?", "Raw zone boost", "Boost mode / basis",
               *[f"Additional {nutrient} boost only" for nutrient in nutrients]]
    valid_means = [float(z["vari"]["mean_vari"]) for z in zone_json
                   if z.get("vari", {}).get("mean_vari") is not None
                   and np.isfinite(z["vari"]["mean_vari"])]
    rows = []
    def fmt(value, digits=3):
        if value is None:
            return "N/A"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"
        return str(value)
    for zone in zone_json:
        rec = zone.get("recommendation", {})
        boost = rec.get("boost_fraction")
        mean = zone.get("vari", {}).get("mean_vari")
        if boost is None:
            boost_mode_text = "Unavailable"
        elif rec.get("boost_mode") == "blanket_p25":
            boost_mode_text = f"Blanket (P25={rec.get('p25_zone_mean_vari'):.3f}): {100 * boost:.1f}%"
        else:
            boost_mode_text = f"Zone-specific: {100 * boost:.1f}%"
        row = [f"Z{zone['zone_id']}", fmt(zone.get("tree_count")),
               fmt(mean),
               fmt(zone.get("vari", {}).get("min_vari")),
               ("CHECK: lowest zone min" if zone.get("vari", {}).get("lowest_min_vari_zone") else "—"),
               fmt(zone.get("confidence", {}).get("zone_factor"), digits=4),
               fmt(zone.get("confidence", {}).get("confidence"), digits=4),
               fmt(rec.get("overall_field_confidence"), digits=4),
               ("yes" if zone.get("confidence", {}).get("reference_zone") else "no"),
               (f"{100 * rec['zone_specific_boost_fraction']:.1f}%" if rec.get("zone_specific_boost_fraction") is not None else "N/A"),
               boost_mode_text]
        for nutrient in nutrients:
            product = next((name for name, mapped_nutrient in FERTILIZER_NUTRIENT_BY_PRODUCT.items()
                            if mapped_nutrient == nutrient and name in rec.get("fertilizers", {})), None)
            item = rec.get("fertilizers", {}).get(product, {}) if product is not None else {}
            total = item.get("annual_total_kg")
            per_palm = item.get("kg_per_palm_per_year")
            row.append(f"{fmt(total, digits=4)} kg/zone/year; "
                       f"{fmt(per_palm, digits=4)} kg/palm/year")
        rows.append(row)
    widths = [max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
              for i, header in enumerate(headers)]
    print("\n--- Zone VARI-Derived Nutrient Supplement Summary (annual) ---")
    overall_field_confidence = (zone_json[0].get("recommendation", {}).get("overall_field_confidence")
                                if zone_json else None)
    print("Notebook confidence thresholds are proposals/guesses pending calibration against field observations.")
    print("Min VARI review flags every zone tied for the lowest valid zone minimum; it is a screening flag for follow-up, not a diagnosis.")
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    if not rows:
        print("(No zones were created.)")
    else:
        for row in rows:
            print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    print("\nConfidence calculations (section 5b):")
    if zone_json:
        first_confidence = zone_json[0].get("confidence", {})
        median_leaf_px = first_confidence.get("median_leaf_pixels_per_tree", 0.0)
        for zone in zone_json:
            c = zone.get("confidence", {})
            n = max(0, int(zone.get("tree_count", 0)))
            leaf_px = int(c.get("leaf_pixel_count", zone.get("vari", {}).get("vegetation_pixel_count", 0)) or 0)
            tree_factor = float(c.get("tree_factor", 0.0))
            leaf_ratio = float(c.get("leaf_pixel_ratio", 0.0))
            leaf_factor = float(c.get("leaf_pixel_factor", 0.0))
            zone_factor = float(c.get("zone_factor", 0.0))
            local_confidence = float(c.get("confidence", 0.0))
            flight = float(c.get("flight_confidence", 0.0))
            print(f"  Z{zone['zone_id']} (weight = {n} trees):")
            print(f"    Tree factor = min(1, sqrt({n} ÷ {ZONE_CONFIDENCE_FULL_TREES})) = {tree_factor:.4f}")
            if n > 0 and median_leaf_px:
                print(f"    Leaf ratio = {leaf_px} ÷ ({n} × {median_leaf_px:.2f} median leaf pixels/tree) = {leaf_ratio:.4f}")
            else:
                print(f"    Leaf ratio = 0.0000 (no valid tree/leaf-pixel denominator)")
            print(f"    Leaf factor = min(1, {leaf_ratio:.4f} ÷ {ZONE_LEAF_PIXEL_FULL_RATIO:.4f}) = {leaf_factor:.4f}")
            print(f"    Zone factor = {tree_factor:.4f} × {leaf_factor:.4f} = {zone_factor:.4f}")
            if c.get("manual_override") is not None:
                print(f"    Zone confidence = manual override {local_confidence:.4f}")
            else:
                print(f"    Zone confidence = clamp({flight:.4f} flight × {zone_factor:.4f} zone factor, "
                      f"{ZONE_CONFIDENCE_FLOOR:.2f}, 1.00) = {local_confidence:.4f}")
        overall_raw = first_confidence.get("overall_field_confidence_raw")
        total_weight = sum(max(0, int(z.get("tree_count", 0))) for z in zone_json)
        weighted_terms = " + ".join(
            f"({z.get('confidence', {}).get('confidence', 0.0):.4f} × "
            f"{max(0, int(z.get('tree_count', 0)))})" for z in zone_json
        )
        print("  Overall field confidence = tree-count-weighted mean of zone confidences:")
        if overall_raw is not None and total_weight > 0:
            print(f"    ({weighted_terms}) ÷ {total_weight} total trees = {overall_raw:.6f}")
            print(f"    Clamp to [{ZONE_CONFIDENCE_FLOOR:.2f}, 1.00] = {overall_field_confidence:.4f}")
        else:
            print(f"    No detected trees; use confidence floor {ZONE_CONFIDENCE_FLOOR:.2f}")
        print("    This one overall value is used in every zone's boost calculation.")
    print("\nBoost calculation by zone:")
    print("  Vref = 95th percentile of valid zone mean VARI values from reference zones only")
    first_rec = zone_json[0].get("recommendation", {}) if zone_json else {}
    boost_mode = first_rec.get("boost_mode", "zone_specific")
    p25 = first_rec.get("p25_zone_mean_vari")
    p95 = first_rec.get("reference_zone_mean_vari")
    gap = first_rec.get("p95_p25_vari_gap")
    reference_id_set = {str(value) for value in first_rec.get("reference_zone_ids", [])}
    reference_means = [float(z["vari"]["mean_vari"]) for z in zone_json
                       if str(z.get("zone_id")) in reference_id_set
                       and z.get("vari", {}).get("mean_vari") is not None
                       and np.isfinite(z["vari"]["mean_vari"])]
    if first_rec.get("reference_fallback_all_zones"):
        reference_means = valid_means
    sorted_means = sorted(reference_means)
    print("  Reference zone IDs = " + (", ".join(f"Z{zid}" for zid in first_rec.get("reference_zone_ids", [])) or "none"))
    if first_rec.get("reference_fallback_all_zones"):
        print("  No zones qualified, so P25/P95 fall back to all valid zone means.")
    for flag in first_rec.get("zone_confidence_run_flags", []):
        print(f"  Confidence note: {flag}")
    print("  Sorted reference-zone mean VARI values = " + ", ".join(f"{value:.3f}" for value in sorted_means))
    print("  Sorted values use zero-based indexes: " + ", ".join(f"sorted[{i}]={value:.6f}" for i, value in enumerate(sorted_means)))
    if p25 is not None and p95 is not None:
        print(f"  Quantile references (linear interpolation): P25={p25:.3f}; P95={p95:.3f}")
        def print_quantile_steps(label, percentile):
            if not sorted_means:
                return
            position = (len(sorted_means) - 1) * percentile
            lower_index = int(np.floor(position))
            upper_index = int(np.ceil(position))
            fraction = position - lower_index
            lower_value = sorted_means[lower_index]
            upper_value = sorted_means[upper_index]
            result = lower_value + fraction * (upper_value - lower_value)
            if lower_index == upper_index:
                print(f"  {label}: position=({len(sorted_means)}-1)×{percentile:.2f}={position:.2f}; "
                      f"sorted[{lower_index}]={lower_value:.6f}")
            else:
                lower_weight = 1.0 - fraction
                lower_part = lower_weight * lower_value
                upper_part = fraction * upper_value
                print(f"  {label}: position=({len(sorted_means)}-1)×{percentile:.2f}={position:.2f}; "
                      f"interpolate sorted[{lower_index}]={lower_value:.6f} and "
                      f"sorted[{upper_index}]={upper_value:.6f} with fraction {fraction:.2f}: "
                      f"({lower_weight:.2f}×{lower_value:.6f})+({fraction:.2f}×{upper_value:.6f}) "
                      f"={lower_part:.6f}+{upper_part:.6f}={result:.6f}")
        print_quantile_steps("P25 calculation", 0.25)
        print_quantile_steps("P95 calculation", 0.95)
    print(f"  P95-P25 zone-mean VARI gap = {gap:.3f}" if gap is not None else "  P95-P25 zone-mean VARI gap unavailable")
    print(f"  Boost mode = {boost_mode}; blanket cutoff = {MIN_P95_P25_VARI_GAP_FOR_BLANKET_BOOST:.3f}")
    print("  B_i = max(0, 1 - V_input / Vref) × overall field confidence")
    if boost_mode == "blanket_p25" and p25 is not None:
        print(f"  Blanket mode uses V_input=P25={p25:.3f} for every zone")
    else:
        print("  Zone-specific mode uses each zone's own mean VARI as V_input")
    print("  Annual zone supplement = B_i × whole-farm nutrient amount × zone area share")
    print("  These are additional VARI-scaled Mg/N quantities, not the full annual fertilizer program.")
    effective_area = sum(
        zone.get("recommendation", {}).get("zone_area_ha") or 0.0 for zone in zone_json
    )
    print(f"  Sum of zone areas used in these totals = {effective_area:.4f} ha")
    farm_rate_totals = {
        nutrient: float(rate) * effective_area
        for nutrient, rate in MAX_NUTRIENT_KG_HA_PER_YEAR.items()
        if rate is not None
    }
    if farm_rate_totals:
        print("  Full-rate reference for this area (before VARI boost): " + "; ".join(
            f"{nutrient} = {float(MAX_NUTRIENT_KG_HA_PER_YEAR[nutrient]):.3f} kg/ha/year × "
            f"{effective_area:.4f} ha = {total:.3f} kg/year"
            for nutrient, total in farm_rate_totals.items()
        ))
    print("Detailed additional Mg and N calculations by zone:")
    for zone in zone_json:
        rec = zone.get("recommendation", {})
        mean = zone.get("vari", {}).get("mean_vari")
        reference = rec.get("reference_zone_mean_vari")
        deficit = rec.get("vari_deficit_fraction")
        confidence = rec.get("zone_confidence")
        boost = rec.get("boost_fraction")
        if None in (mean, reference, deficit, confidence, boost) or reference <= 0:
            print(f"  Z{zone['zone_id']}: cannot calculate (missing/invalid VARI reference or zone mean)")
        else:
            boost_input = rec.get("boost_vari_input", mean)
            print(f"  Z{zone['zone_id']}: zone mean VARI={mean:.9f}; V_input={boost_input:.9f}; Vref(P95)={reference:.9f}; "
                  f"B_i=max(0, 1 - {boost_input:.9f}/{reference:.9f}) × {confidence:.6f}="
                  f"{boost:.9f} ({boost * 100:.6f}%)")
        for nutrient, max_rate in MAX_NUTRIENT_KG_HA_PER_YEAR.items():
            product = next((name for name, mapped_nutrient in FERTILIZER_NUTRIENT_BY_PRODUCT.items()
                            if mapped_nutrient == nutrient and name in rec.get("fertilizers", {})), None)
            item = rec.get("fertilizers", {}).get(product) if product is not None else None
            if not item:
                continue
            if boost is None:
                print(f"    {nutrient}: unavailable because zone boost could not be calculated")
                continue
            zone_total = item["annual_total_kg"]
            per_palm = item["kg_per_palm_per_year"]
            area = rec.get("zone_area_ha")
            zone_text = f"{zone_total:.6f}" if zone_total is not None else "N/A"
            per_palm_text = f"{per_palm:.6f}" if per_palm is not None else "N/A"
            area_text = f"{area:.6f}" if area is not None else "N/A"
            tree_count = zone.get("tree_count", 0)
            if area is not None and effective_area > 0:
                farm_total = float(max_rate) * effective_area
                zone_share = float(area) / effective_area
                print(f"      additional {nutrient} supplement/zone/year = B_i × whole-farm {nutrient} amount × zone area share = "
                      f"{boost:.9f} × {farm_total:.3f} kg {nutrient}/farm/year × "
                      f"({area_text} ha ÷ {effective_area:.4f} ha = {zone_share:.6f}) = "
                      f"{zone_text} kg {nutrient}/zone/year")
                print(f"        equivalent per-hectare form: {boost:.9f} × {max_rate:.3f} kg {nutrient}/ha/year × "
                      f"{area_text} ha = {zone_text} kg {nutrient}/zone/year")
            else:
                print(f"      additional {nutrient} supplement/zone/year = N/A (zone area is unavailable)")
            print(f"      per palm = {zone_text} kg ÷ {tree_count} trees = "
                  f"{per_palm_text} kg {nutrient}/palm/year")
    print("Values above are additional annual elemental Mg or N supplements, not full product rates or full fertilizer-program totals.")

# ## CSV and fertilizer product-mass outputs


# ===== Notebook code cell 10 =====
OUTPUT_DECIMALS = globals().get("OUTPUT_DECIMALS", 6)  # Fallback for running this output cell before the main config cell.

def save_zone_vari_csv(zone_json, output_path):
    """Save one row per zone with VARI and fertilizer recommendation fields."""
    fieldnames = ["zone_id", "approx_row", "tree_count", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                  "vegetation_pixel_count", "mean_vari", "median_vari", "std_vari", "mad_vari", "min_vari", "max_vari",
                  "lowest_min_vari_zone", "lowest_zone_min_vari_value", "min_vari_review_note",
                  "tree_factor", "leaf_pixel_count", "median_leaf_pixels_per_tree", "leaf_pixel_ratio",
                  "leaf_pixel_factor", "zone_factor", "flight_confidence", "flight_confidence_source",
                  "calculated_zone_confidence", "overall_field_confidence_raw", "overall_field_confidence",
                  "confidence_used_for_boost", "field_confidence_weight", "manual_zone_confidence_override", "zone_is_reference",
                  "confidence_flags",
                  "reference_zone_mean_vari", "p25_zone_mean_vari", "p95_p25_vari_gap",
                  "zone_mean_vari_percentile_rank", "boost_mode", "boost_vari_input", "zone_confidence",
                  "zone_local_confidence",
                  "zone_specific_vari_deficit_fraction", "zone_specific_boost_fraction",
                  "vari_deficit_fraction", "boost_fraction", "zone_area_ha"]
    metrics = ("boost_added_rate_kg_ha_per_year", "boosted_rate_kg_ha_per_year",
               "annual_total_kg", "kg_per_palm_per_year")
    fieldnames += [f"{product}_{metric}" for product in FERTILIZER_NUTRIENT_BY_PRODUCT for metric in metrics]
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for zone in zone_json:
            bbox, stats = zone.get("bbox", {}), zone.get("vari", {})
            confidence = zone.get("confidence", {})
            rec = zone.get("recommendation", {})
            row = {"zone_id": zone.get("zone_id"), "approx_row": zone.get("approx_row"),
                   "tree_count": zone.get("tree_count"), "bbox_x1": bbox.get("x1"), "bbox_y1": bbox.get("y1"),
                   "bbox_x2": bbox.get("x2"), "bbox_y2": bbox.get("y2"),
                   **{k: stats.get(k) for k in ("vegetation_pixel_count", "mean_vari", "median_vari", "std_vari", "mad_vari", "min_vari", "max_vari",
                                                   "lowest_min_vari_zone", "lowest_zone_min_vari_value",
                                                   "min_vari_review_note")},
                   "tree_factor": confidence.get("tree_factor"),
                   "leaf_pixel_count": confidence.get("leaf_pixel_count"),
                   "median_leaf_pixels_per_tree": confidence.get("median_leaf_pixels_per_tree"),
                   "leaf_pixel_ratio": confidence.get("leaf_pixel_ratio"),
                   "leaf_pixel_factor": confidence.get("leaf_pixel_factor"),
                   "zone_factor": confidence.get("zone_factor"),
                   "flight_confidence": confidence.get("flight_confidence"),
                   "flight_confidence_source": confidence.get("flight_confidence_source"),
                   "calculated_zone_confidence": confidence.get("calculated_confidence"),
                   "overall_field_confidence_raw": confidence.get("overall_field_confidence_raw"),
                   "overall_field_confidence": confidence.get("overall_field_confidence"),
                   "confidence_used_for_boost": confidence.get("confidence_used_for_boost"),
                   "field_confidence_weight": confidence.get("field_confidence_weight"),
                   "manual_zone_confidence_override": confidence.get("manual_override"),
                   "zone_is_reference": confidence.get("reference_zone"),
                   "confidence_flags": "; ".join(confidence.get("flags", [])),
                   **{k: rec.get(k) for k in ("reference_zone_mean_vari", "p25_zone_mean_vari", "p95_p25_vari_gap",
                                               "zone_mean_vari_percentile_rank", "boost_mode", "boost_vari_input",
                                               "zone_confidence", "zone_local_confidence", "overall_field_confidence",
                                               "zone_specific_vari_deficit_fraction",
                                               "zone_specific_boost_fraction", "vari_deficit_fraction",
                                               "boost_fraction", "zone_area_ha")}}
            for product, item in rec.get("fertilizers", {}).items():
                for metric in metrics:
                    row[f"{product}_{metric}"] = item.get(metric)
            writer.writerow(row)


def build_and_print_product_mass_table(zone_json):
    """Separately convert annual elemental Mg/N supplements to product mass."""
    rows = []
    print("\n--- Separate VARI Supplement Product-Mass Conversion (annual) ---")
    print("These product quantities cover the VARI-derived nutrient supplement only.")
    print("They are not a complete annual fertilizer recommendation.")
    for product, base_rate in FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.items():
        if base_rate is not None and (not np.isfinite(float(base_rate)) or float(base_rate) < 0):
            raise ValueError(f"Base product rate for {product} must be nonnegative or None.")
        if base_rate is not None:
            print(f"Base application included for {product}: {float(base_rate):.4f} kg product/palm/year.")
    print(f"  Dolomite base rate source: {DOLOMITE_BASE_RATE_SOURCE}.")
    print(f"  Kieserite base rate source: {KIESERITE_BASE_RATE_SOURCE}.")
    print("  Dolomite and kieserite are alternative Mg products; choose one, do not add both rows together.")
    if not any(rate is not None for rate in FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.values()):
        print("All optional base product rates are unset; only VARI-derived supplements are calculated.")
    print("  Mg product calculation is shown in two steps: elemental Mg → MgO target → product kg.")
    print(f"  MgO → Mg factor = {MGO_TO_MG_FACTOR:.9f}; Mg → MgO factor = {MG_TO_MGO_FACTOR:.9f}.")
    for zone in zone_json:
        rec = zone.get("recommendation", {})
        for product, nutrient in FERTILIZER_NUTRIENT_BY_PRODUCT.items():
            max_nutrient_rate = MAX_NUTRIENT_KG_HA_PER_YEAR.get(nutrient)
            if max_nutrient_rate is None:
                continue
            item = rec.get("fertilizers", {}).get(product, {})
            nutrient = item.get("recommended_nutrient", FERTILIZER_NUTRIENT_BY_PRODUCT[product])
            nutrient_zone_total = item.get("annual_total_kg")
            fraction = FERTILIZER_PRODUCT_NUTRIENT_FRACTIONS.get(product)
            if fraction is not None:
                fraction = float(fraction)
                if not np.isfinite(fraction) or not 0 < fraction <= 1:
                    raise ValueError(f"Product nutrient fraction for {product} must be greater than 0 and at most 1.")
            label_info = FERTILIZER_PRODUCT_LABEL_FRACTIONS.get(product, {})
            label_basis = label_info.get("basis")
            label_fraction = label_info.get("fraction")
            if label_fraction is not None:
                label_fraction = float(label_fraction)
                if not np.isfinite(label_fraction) or not 0 < label_fraction <= 1:
                    raise ValueError(f"Label fraction for {product} must be greater than 0 and at most 1.")
            if nutrient_zone_total is None or fraction is None:
                product_total = None
            elif nutrient == "Mg" and label_basis == "MgO" and label_fraction is not None:
                mgo_target = nutrient_zone_total * MG_TO_MGO_FACTOR
                product_total = mgo_target / label_fraction
            else:
                product_total = nutrient_zone_total / fraction
            product_per_palm = (product_total / zone["tree_count"]
                                if product_total is not None and zone.get("tree_count", 0) > 0 else None)
            base_product_total = None
            combined_product_total = None
            base_rate = FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.get(product)
            if base_rate is not None:
                base_product_total = float(base_rate) * zone.get("tree_count", 0)
# Base product is required regardless of the VARI boost. If the
# supplement is zero, the combined total remains the base rate.
                combined_product_total = (base_product_total + product_total
                                          if product_total is not None else None)
            price = FERTILIZER_PRICE_PER_KG.get(product)
            rows.append({
                "zone_id": zone["zone_id"], "product": product, "nutrient": nutrient,
                "elemental_nutrient_kg_per_zone_per_year": nutrient_zone_total,
                "product_label_basis": label_basis,
                "product_label_fraction": label_fraction,
                "elemental_nutrient_fraction_per_kg_product": fraction,
                "tree_count": zone.get("tree_count"),
                "product_kg_per_zone_per_year": product_total,
                "product_kg_per_palm_per_year": product_per_palm,
                "base_product_kg_per_zone_per_year": base_product_total,
                "base_plus_supplement_product_kg_per_zone_per_year": combined_product_total,
                "base_plus_supplement_product_kg_per_palm_per_year": (
                    combined_product_total / zone["tree_count"]
                    if combined_product_total is not None and zone.get("tree_count", 0) > 0 else None
                ),
                **({"price_per_kg_product": float(price),
                    "annual_product_cost": product_total * float(price) if product_total is not None else None}
                   if price is not None else {}),
            })
    headers = ["Zone", "Product", "Elemental nutrient kg/zone/year", "Label basis",
               "Label fraction (kg basis/kg product)",
               "Supplement product kg/zone/year", "Trees", "Supplement product kg/palm/year"]
    if any(rate is not None for rate in FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.values()):
        headers.extend(["Base product (always applied) kg/zone/year", "Base+supplement kg/zone/year",
                        "Base+supplement product kg/palm/year"])
    if any(FERTILIZER_PRICE_PER_KG.get(p) is not None for p in FERTILIZER_NUTRIENT_BY_PRODUCT):
        headers.append("Annual product cost")
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    display = []
    for r in rows:
        line = [f"Z{r['zone_id']}", r["product"], fmt(r["elemental_nutrient_kg_per_zone_per_year"]),
                r["product_label_basis"] or "N/A", fmt(r["product_label_fraction"]),
                fmt(r["product_kg_per_zone_per_year"]),
                str(r["tree_count"]), fmt(r["product_kg_per_palm_per_year"])]
        if any(rate is not None for rate in FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.values()):
            line.extend([fmt(r["base_product_kg_per_zone_per_year"]),
                         fmt(r["base_plus_supplement_product_kg_per_zone_per_year"]),
                         fmt(r["base_plus_supplement_product_kg_per_palm_per_year"])])
        if any(FERTILIZER_PRICE_PER_KG.get(product) is not None
               for product in FERTILIZER_NUTRIENT_BY_PRODUCT):
            line.append(fmt(r.get("annual_product_cost")))
        display.append(line)
    widths = [max(len(h), *(len(r[i]) for r in display)) if display else len(h) for i, h in enumerate(headers)]
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    if not display:
        print("(No active fertilizer products.)")
    else:
        for row in display:
            print(" | ".join(v.ljust(widths[i]) for i, v in enumerate(row)))
    print("\nSupplement product-mass calculations:")
    print(f"  Mg supplements: step 1: elemental Mg × {MG_TO_MGO_FACTOR:.9f} = MgO target.")
    print("                step 2: MgO target ÷ product label MgO fraction = product kg.")
    print("  N supplements: product kg = elemental N target ÷ product label N fraction.")
    print("  supplement product kg/palm/year = supplement product kg/zone/year ÷ tree count")
    if any(rate is not None for rate in FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.values()):
        print("  configured base product is counted in every zone, including zones with 0% VARI supplement")
        print("  total product = configured base product + VARI-derived supplement; at 0% supplement, total = base")
    for row in rows:
        target = row["elemental_nutrient_kg_per_zone_per_year"]
        fraction = row["elemental_nutrient_fraction_per_kg_product"]
        if target is None or fraction is None:
            print(f"  Z{row['zone_id']} {row['product']}: enter its verified label fraction to calculate product mass")
        else:
            label_info = FERTILIZER_PRODUCT_LABEL_FRACTIONS.get(row["product"], {})
            if row["nutrient"] == "Mg" and label_info.get("basis") == "MgO":
                mgo_target = target * MG_TO_MGO_FACTOR
                label_mgo_fraction = float(label_info["fraction"])
                print(f"  Z{row['zone_id']} {row['product']}:")
                print(f"    Step 1: {target:.6f} kg elemental Mg × {MG_TO_MGO_FACTOR:.9f} "
                      f"= {mgo_target:.6f} kg MgO needed.")
                print(f"    Step 2: {mgo_target:.6f} kg MgO ÷ {label_mgo_fraction:.6f} "
                      f"kg MgO/kg product = {row['product_kg_per_zone_per_year']:.6f} kg product/zone/year.")
            else:
                print(f"  Z{row['zone_id']} {row['product']}: {target:.6f} kg {row['nutrient']} "
                      f"÷ {fraction:.9f} kg {row['nutrient']}/kg product "
                      f"= {row['product_kg_per_zone_per_year']:.6f} kg product/zone/year.")
        if row.get("base_product_kg_per_zone_per_year") is not None:
            supplement = row.get("product_kg_per_zone_per_year")
            combined = row.get("base_plus_supplement_product_kg_per_zone_per_year")
            if supplement is not None and combined is not None:
                print(f"    {row['product']} base + supplement = "
                      f"{row['base_product_kg_per_zone_per_year']:.4f} + {supplement:.4f} "
                      f"= {combined:.4f} kg product/zone/year")
            else:
                print(f"    base amount = {row['base_product_kg_per_zone_per_year']:.4f} kg product/zone/year; "
                      "total unavailable until supplement product mass can be calculated")
            if row.get("annual_product_cost") is not None:
                print(f"    annual product cost = {row['product_kg_per_zone_per_year']:.4f} kg × "
                      f"{row['price_per_kg_product']:.4f}/kg = {row['annual_product_cost']:.4f}")
    return rows


def build_and_print_zone_bag_purchase_table(product_mass_rows):
    """Convert each zone's base-plus-boost product amount into bag equivalents."""
    print("\n--- Zone Fertilizer Bag Purchase Estimate (annual) ---")
    print("Uses the base + VARI supplement product total per zone whenever it is available.")
    print("If that combined zone total is N/A, fallback = supplement kg/palm/year × zone tree count + base product kg/zone/year.")
    print("Field bags to buy = round up the summed annual product kg across zones ÷ configured bag size.")
    print("Cost uses the configured rupiah price per bag; zone costs are proportional allocations, and field purchase cost uses whole bags.")
    print("Dolomite/kieserite are alternative Mg sources; ZA/urea are alternative N sources. Choose one source per nutrient.")
    products = list(FERTILIZER_NUTRIENT_BY_PRODUCT)
    rows_by_product = {product: [] for product in products}
    for item in product_mass_rows:
        if item["product"] in rows_by_product:
            rows_by_product[item["product"]].append(item)

    output_rows = []
    display_rows = []
    for product in products:
        bag_size = float(FERTILIZER_BAG_SIZE_KG.get(product, 50.0))
        if not np.isfinite(bag_size) or bag_size <= 0:
            raise ValueError(f"Bag size for {product} must be finite and greater than zero.")
        price_per_bag = FERTILIZER_PRICE_PER_BAG_RP.get(product)
        total_product_kg = 0.0
        all_known = bool(rows_by_product[product])
        for item in rows_by_product[product]:
            tree_count = int(item.get("tree_count") or 0)
            combined_zone_kg = item.get("base_plus_supplement_product_kg_per_zone_per_year")
            combined_palm_kg = item.get("base_plus_supplement_product_kg_per_palm_per_year")
            supplement_palm_kg = item.get("product_kg_per_palm_per_year")
            base_zone_kg = item.get("base_product_kg_per_zone_per_year")
            calculation_source = "base_plus_supplement_zone_total"

            # Prefer the explicit combined amount. If it is N/A, rebuild the zone
            # total from supplement-per-palm and the always-applied base product.
            if combined_zone_kg is not None:
                amount_for_bags_kg = float(combined_zone_kg)
            elif supplement_palm_kg is not None and tree_count > 0:
                amount_for_bags_kg = (float(supplement_palm_kg) * tree_count
                                      + float(base_zone_kg or 0.0))
                calculation_source = "supplement_per_palm_times_tree_count_plus_base_zone_total"
                if combined_palm_kg is None:
                    combined_palm_kg = amount_for_bags_kg / tree_count
            else:
                amount_for_bags_kg = None
                all_known = False
                calculation_source = "unavailable"

            exact_bags = amount_for_bags_kg / bag_size if amount_for_bags_kg is not None else None
            if amount_for_bags_kg is not None:
                total_product_kg += amount_for_bags_kg
            output_rows.append({
                "zone_id": item.get("zone_id"),
                "product": product,
                "nutrient": item.get("nutrient"),
                "base_plus_supplement_product_kg_per_zone_per_year": amount_for_bags_kg,
                "base_plus_supplement_product_kg_per_palm_per_year": combined_palm_kg,
                "supplement_product_kg_per_palm_per_year": supplement_palm_kg,
                "base_product_kg_per_zone_per_year": base_zone_kg,
                "bag_size_kg": bag_size,
                "exact_bag_equivalent_for_zone": exact_bags,
                "price_per_bag_rp": price_per_bag,
                "allocated_cost_rp": (exact_bags * price_per_bag
                                       if exact_bags is not None and price_per_bag is not None else None),
                "calculation_source": calculation_source,
                "field_total_product_kg_per_year": None,
                "field_total_bag_equivalent": None,
                "full_bags_to_buy_for_field": None,
            })
            display_rows.append([
                str(item.get("zone_id")), product,
                "N/A" if amount_for_bags_kg is None else f"{amount_for_bags_kg:.{OUTPUT_DECIMALS}f}",
                "N/A" if combined_palm_kg is None else f"{float(combined_palm_kg):.{OUTPUT_DECIMALS}f}",
                f"{bag_size:.{OUTPUT_DECIMALS}f}",
                "N/A" if exact_bags is None else f"{exact_bags:.{OUTPUT_DECIMALS}f}",
                calculation_source,
                "—",
                "—" if price_per_bag is None else f"Rp{price_per_bag:,.0f}",
                "N/A" if exact_bags is None or price_per_bag is None else f"Rp{exact_bags * price_per_bag:,.0f}",
                "—",
            ])

        field_bag_equivalent = total_product_kg / bag_size if all_known else None
        full_bags = (int(math.ceil(field_bag_equivalent - 1e-12))
                     if field_bag_equivalent is not None and field_bag_equivalent > 0
                     else 0 if field_bag_equivalent == 0 else None)
        total_row = {
            "zone_id": "FIELD TOTAL",
            "product": product,
            "nutrient": FERTILIZER_NUTRIENT_BY_PRODUCT[product],
            "base_plus_supplement_product_kg_per_zone_per_year": None,
            "base_plus_supplement_product_kg_per_palm_per_year": None,
            "supplement_product_kg_per_palm_per_year": None,
            "base_product_kg_per_zone_per_year": None,
            "bag_size_kg": bag_size,
            "exact_bag_equivalent_for_zone": None,
            "calculation_source": "sum of selected base + supplement zone amounts",
            "field_total_product_kg_per_year": total_product_kg if all_known else None,
            "field_total_bag_equivalent": field_bag_equivalent,
            "full_bags_to_buy_for_field": full_bags,
            "price_per_bag_rp": price_per_bag,
            "exact_field_cost_rp": (field_bag_equivalent * price_per_bag
                                     if field_bag_equivalent is not None and price_per_bag is not None else None),
            "full_bag_purchase_cost_rp": (full_bags * price_per_bag
                                           if full_bags is not None and price_per_bag is not None else None),
        }
        output_rows.append(total_row)
        display_rows.append([
            "FIELD TOTAL", product,
            "N/A" if not all_known else f"{total_product_kg:.{OUTPUT_DECIMALS}f}",
            "—", f"{bag_size:.{OUTPUT_DECIMALS}f}",
            "N/A" if field_bag_equivalent is None else f"{field_bag_equivalent:.{OUTPUT_DECIMALS}f}",
            "sum zone totals ÷ bag size", "N/A" if full_bags is None else str(full_bags),
            "—" if price_per_bag is None else f"Rp{price_per_bag:,.0f}",
            "N/A" if field_bag_equivalent is None or price_per_bag is None else f"Rp{field_bag_equivalent * price_per_bag:,.0f}",
            "N/A" if full_bags is None or price_per_bag is None else f"Rp{full_bags * price_per_bag:,.0f}",
        ])

    headers = ["Zone", "Product", "Base + supplement kg/zone/year", "Base + supplement kg/palm/year",
               "Bag kg", "Exact bag equivalents", "Quantity calculation", "Full bags to buy (field total)",
               "Price/bag", "Allocated/exact cost", "Whole-bag field cost"]
    widths = [max(len(headers[i]), *(len(row[i]) for row in display_rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in display_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    print("For a zone with a combined total: bags equivalent = base + supplement kg/zone/year ÷ bag kg.")
    print("Fallback when combined total is N/A: (supplement kg/palm/year × tree count + base kg/zone/year) ÷ bag kg.")
    print("Only the FIELD TOTAL is rounded up; per-zone bag equivalents and allocated costs are proportional, not separate purchases.")
    print("Whole-bag field cost = rounded-up field bag count × price per bag. Missing product prices remain N/A.")
    return output_rows


def save_product_mass_csv(rows, output_path):
    """Save product-mass conversions separately from the elemental zone CSV."""
    fields = ["zone_id", "product", "nutrient", "elemental_nutrient_kg_per_zone_per_year",
              "product_label_basis", "product_label_fraction",
              "elemental_nutrient_fraction_per_kg_product", "tree_count",
              "product_kg_per_zone_per_year", "product_kg_per_palm_per_year",
              "base_product_kg_per_zone_per_year",
              "base_plus_supplement_product_kg_per_zone_per_year",
              "base_plus_supplement_product_kg_per_palm_per_year"]
    if any(FERTILIZER_PRICE_PER_KG.get(p) is not None for p in FERTILIZER_NUTRIENT_BY_PRODUCT):
        fields.extend(["price_per_kg_product", "annual_product_cost"])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

# ## VARI distribution plot


# ===== Notebook code cell 12 =====
def save_vari_distribution_plot(zone_json, output_path):
    """Plot zone mean VARI values and identify which fall from P25 to P95."""
    valid_zones = [
        (zone["zone_id"], float(zone["vari"]["mean_vari"]))
        for zone in zone_json
        if zone.get("vari", {}).get("mean_vari") is not None
        and np.isfinite(zone["vari"]["mean_vari"])
    ]
    fig_height = max(4.5, 0.55 * len(valid_zones) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    if not valid_zones:
        ax.text(0.5, 0.5, "No valid zone mean VARI values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        valid_zones.sort(key=lambda item: item[1])
        means = [mean for _, mean in valid_zones]
        p25, p95 = np.percentile(means, [25, 95])
        y_values = np.arange(len(valid_zones))
        ax.axvspan(p25, p95, color="#a5d6a7", alpha=0.35, label="P25–P95 range")
        ax.axvline(p25, color="#ef6c00", linestyle="--", linewidth=1.8, label=f"P25 = {p25:.3f}")
        ax.axvline(p95, color="#1565c0", linestyle="--", linewidth=1.8, label=f"P95 = {p95:.3f}")
        for y, (zone_id, mean) in zip(y_values, valid_zones):
            color = "#ef6c00" if mean < p25 else "#7b1fa2" if mean > p95 else "#2e7d32"
            ax.scatter(mean, y, color=color, s=85, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(f"{mean:.3f}", (mean, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(y_values)
        ax.set_yticklabels([f"Z{zone_id}" for zone_id, _ in valid_zones])
        ax.invert_yaxis()
        min_value, max_value = min(means), max(means)
        span = max(max_value - min_value, 0.02)
        ax.set_xlim(min_value - 0.12 * span, max_value + 0.32 * span)
# Keep regular ticks uncluttered; print exact quantile values in a
# dedicated line below the axis instead of adding crowded tick labels.
        ax.text(0.5, -0.20, f"Exact cutoffs: P25 = {p25:.6f}   |   P95 = {p95:.6f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.set_xlabel("Zone mean VARI")
        ax.set_ylabel("Zone (sorted by mean VARI)")
        ax.set_title("Zone VARI Distribution and P25–P95 Range")
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.legend(loc="best", frameon=True)
        ax.text(0.01, -0.28,
                "Point colors: orange = below P25; green = within P25–P95; purple = above P95.",
                transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ## Zone confidence, area, and boost calculations
#
# The confidence factors and reference-zone rules are defined here; the steps below call these functions.


# ===== Notebook code cell 14 =====
def load_zone_areas():
    """Return supplied ground area in hectares by zone ID; never infer from pixels."""
    areas = {str(k): float(v) for k, v in ZONE_AREA_HA_BY_ID.items()}
    if ZONE_AREA_JSON_PATH:
        with open(ZONE_AREA_JSON_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("ZONE_AREA_JSON_PATH must contain a JSON object mapping zone IDs to hectares.")
        areas.update({str(k): float(v) for k, v in loaded.items()})
    if UNIFORM_ZONE_AREA_HA is not None:
        if float(UNIFORM_ZONE_AREA_HA) <= 0:
            raise ValueError("UNIFORM_ZONE_AREA_HA must be positive.")
    for key, value in areas.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Area for zone {key} must be a finite positive number of hectares.")
    return areas


def calculate_zone_confidence_components(
    tree_count, leaf_pixel_count, median_leaf_pixels_per_tree, flight_confidence, override=None
):
    """Shared confidence formula used for Step 4 preview and final recommendations."""
    n = max(0, int(tree_count))
    leaf_pixels = max(0, int(leaf_pixel_count))
    tree_factor = min(1.0, float(np.sqrt(n / ZONE_CONFIDENCE_FULL_TREES))) if n else 0.0
    leaf_denominator = n * median_leaf_pixels_per_tree
    leaf_ratio = leaf_pixels / leaf_denominator if leaf_denominator > 0 else 0.0
    leaf_factor = min(1.0, leaf_ratio / ZONE_LEAF_PIXEL_FULL_RATIO)
    zone_factor = tree_factor * leaf_factor
    raw_confidence = flight_confidence * zone_factor
    calculated_confidence = min(1.0, max(ZONE_CONFIDENCE_FLOOR, raw_confidence))
    confidence = calculated_confidence if override is None else float(override)
    if not np.isfinite(confidence) or not 0.10 <= confidence <= 1.0:
        raise ValueError(f"Zone confidence override must be in the range 0.10 to 1.0; got {confidence}.")
    return {
        "tree_factor": tree_factor,
        "leaf_pixel_count": leaf_pixels,
        "leaf_denominator": leaf_denominator,
        "leaf_pixel_ratio": leaf_ratio,
        "leaf_pixel_factor": leaf_factor,
        "zone_factor": zone_factor,
        "raw_confidence": raw_confidence,
        "calculated_confidence": calculated_confidence,
        "manual_override": override,
        "confidence": confidence,
    }


def add_zone_boost_and_fertilizer(zone_json, image_size=None):
    """Attach notebook-style confidence and annual VARI-relative nutrient targets."""
    width, height = image_size if image_size is not None else (None, None)
    tree_leaf_pixels = [
        int(tree.get("vari", {}).get("vegetation_pixel_count") or 0)
        for zone in zone_json for tree in zone.get("trees", [])
        if int(tree.get("vari", {}).get("vegetation_pixel_count") or 0) > 0
    ]
    median_leaf_pixels_per_tree = float(np.median(tree_leaf_pixels)) if tree_leaf_pixels else 0.0
    flight_confidence = (DEFAULT_ZONE_CONFIDENCE if FLIGHT_CONFIDENCE is None
                         else float(FLIGHT_CONFIDENCE))
    if not np.isfinite(flight_confidence) or not 0.0 <= flight_confidence <= 1.0:
        raise ValueError("FLIGHT_CONFIDENCE must be None or a finite value from 0 to 1.")
    if ZONE_CONFIDENCE_FULL_TREES <= 0 or ZONE_LEAF_PIXEL_FULL_RATIO <= 0:
        raise ValueError("ZONE_CONFIDENCE_FULL_TREES and ZONE_LEAF_PIXEL_FULL_RATIO must be positive.")
    if not 0.0 < ZONE_CONFIDENCE_FLOOR <= 1.0:
        raise ValueError("ZONE_CONFIDENCE_FLOOR must be greater than 0 and at most 1.")
    if not 0.0 <= ZONE_REFERENCE_MIN_FACTOR <= 1.0:
        raise ValueError("ZONE_REFERENCE_MIN_FACTOR must be between 0 and 1.")

    reference_ids = []
    confidence_run_flags = []
    for zone in zone_json:
        n = max(0, int(zone.get("tree_count", 0)))
        leaf_pixels = int(zone.get("vari", {}).get("vegetation_pixel_count") or 0)
        zone_id = zone.get("zone_id")
        override = ZONE_CONFIDENCE_BY_ID.get(str(zone_id), ZONE_CONFIDENCE_BY_ID.get(zone_id))
        confidence_parts = calculate_zone_confidence_components(
            n, leaf_pixels, median_leaf_pixels_per_tree, flight_confidence, override
        )
        tree_factor = confidence_parts["tree_factor"]
        leaf_ratio = confidence_parts["leaf_pixel_ratio"]
        leaf_factor = confidence_parts["leaf_pixel_factor"]
        zone_factor = confidence_parts["zone_factor"]
        raw_confidence = confidence_parts["raw_confidence"]
        calculated_confidence = confidence_parts["calculated_confidence"]
        confidence = confidence_parts["confidence"]
        mean_vari = zone.get("vari", {}).get("mean_vari")
        in_reference = (zone_factor >= ZONE_REFERENCE_MIN_FACTOR and mean_vari is not None
                        and np.isfinite(mean_vari))
        bbox = zone.get("bbox", {})
        touches_edge = False
        if width is not None and height is not None:
            touches_edge = (
                bbox.get("x1", 0) <= BORDER_MARGIN_PX or bbox.get("y1", 0) <= BORDER_MARGIN_PX
                or bbox.get("x2", width) >= width - BORDER_MARGIN_PX
                or bbox.get("y2", height) >= height - BORDER_MARGIN_PX
            )
        flags = []
        if FLIGHT_CONFIDENCE is None:
            flags.append("flight confidence not assessed; DEFAULT_ZONE_CONFIDENCE used as flight factor")
        if tree_factor < 1.0:
            flags.append(f"{n} tree(s): tree factor {tree_factor:.2f}")
        if leaf_factor < 1.0:
            flags.append(f"low ExG leaf-pixel coverage (ratio {leaf_ratio:.2f})")
        if not in_reference:
            if mean_vari is None or not np.isfinite(mean_vari):
                flags.append("no valid zone mean VARI for the P25/P95 reference set")
            else:
                flags.append(f"not a P25/P95 reference zone (zone factor {zone_factor:.2f} < {ZONE_REFERENCE_MIN_FACTOR})")
        if raw_confidence < ZONE_CONFIDENCE_FLOOR:
            flags.append(f"calculated confidence raised to floor {ZONE_CONFIDENCE_FLOOR:.2f}")
        if override is not None:
            flags.append("manual confidence override applied")
        if touches_edge:
            flags.append("zone bbox touches photo edge (review flag)")
        zone["confidence"] = {
            "tree_factor": round(tree_factor, 4),
            "leaf_pixel_count": leaf_pixels,
            "median_leaf_pixels_per_tree": median_leaf_pixels_per_tree,
            "leaf_pixel_ratio": round(leaf_ratio, 4),
            "leaf_pixel_factor": round(leaf_factor, 4),
            "zone_factor": round(zone_factor, 4),
            "flight_confidence": flight_confidence,
            "flight_confidence_source": ("DEFAULT_ZONE_CONFIDENCE stand-in" if FLIGHT_CONFIDENCE is None
                                         else FLIGHT_CONFIDENCE_SOURCE),
            "calculated_confidence": round(calculated_confidence, 4),
            "manual_override": override,
            "confidence": round(confidence, 4),
            "reference_zone": bool(in_reference),
            "touches_photo_edge": bool(touches_edge),
            "may_be_less_accurate": bool(FLIGHT_CONFIDENCE is None or zone_factor < 1.0),
            "flags": flags,
        }
        if in_reference:
            reference_ids.append(zone_id)

    valid_means = [z["vari"].get("mean_vari") for z in zone_json
                   if z.get("vari", {}).get("mean_vari") is not None
                   and np.isfinite(z["vari"]["mean_vari"])]
    reference_zones = [z for z in zone_json if z.get("confidence", {}).get("reference_zone")]
    reference_fallback = not reference_zones and bool(valid_means)
    if reference_fallback:
        reference_zones = [z for z in zone_json
                           if z.get("vari", {}).get("mean_vari") is not None
                           and np.isfinite(z["vari"]["mean_vari"])]
        confidence_run_flags.append("no zone qualified for the reference set; P25/P95 use all valid zone means")
    if FLIGHT_CONFIDENCE is None:
        confidence_run_flags.append(
            "flight confidence (L6) not assessed; DEFAULT_ZONE_CONFIDENCE is used as the flight factor"
        )
    confidence_weight_total = sum(max(0, int(zone.get("tree_count", 0))) for zone in zone_json)
    if confidence_weight_total > 0:
        field_confidence_raw = sum(
            zone["confidence"]["confidence"] * max(0, int(zone.get("tree_count", 0)))
            for zone in zone_json
        ) / confidence_weight_total
    else:
        field_confidence_raw = 0.0
    field_confidence = min(1.0, max(ZONE_CONFIDENCE_FLOOR, field_confidence_raw))
    for zone in zone_json:
        zone["confidence"]["field_confidence_weight"] = max(0, int(zone.get("tree_count", 0)))
        zone["confidence"]["overall_field_confidence_raw"] = field_confidence_raw
        zone["confidence"]["overall_field_confidence"] = field_confidence
        zone["confidence"]["confidence_used_for_boost"] = field_confidence
    if reference_zones and len(reference_zones) < MIN_REFERENCE_ZONES_SUGGESTED:
        confidence_run_flags.append(
            f"P25/P95 use {len(reference_zones)} reference zone(s), fewer than the suggested "
            f"{MIN_REFERENCE_ZONES_SUGGESTED} (proposal, not a validated threshold)"
        )
    reference_zone_ids_used = [z["zone_id"] for z in reference_zones]
    reference_means = [z["vari"]["mean_vari"] for z in reference_zones]
    valid_mins = [z["vari"].get("min_vari") for z in zone_json
                  if z.get("vari", {}).get("min_vari") is not None
                  and np.isfinite(z["vari"]["min_vari"])]
    lowest_zone_min_vari = min(valid_mins) if valid_mins else None
    reference = float(np.percentile(reference_means, 95)) if reference_means else None
    p25 = float(np.percentile(reference_means, 25)) if reference_means else None
    valid_zone_count = len(valid_means)
    p95_p25_gap = reference - p25 if reference is not None and p25 is not None else None
    blanket_mode = (p95_p25_gap is not None
                    and p95_p25_gap <= MIN_P95_P25_VARI_GAP_FOR_BLANKET_BOOST)
    areas = load_zone_areas()
    mapped_area_configured = bool(areas)
    if mapped_area_configured and UNIFORM_ZONE_AREA_HA is not None:
        raise ValueError("Use either per-zone areas or UNIFORM_ZONE_AREA_HA, not both.")
    if not mapped_area_configured and UNIFORM_ZONE_AREA_HA is None and TOTAL_FARM_AREA_HA is not None:
        if not np.isfinite(float(TOTAL_FARM_AREA_HA)) or float(TOTAL_FARM_AREA_HA) <= 0:
            raise ValueError("TOTAL_FARM_AREA_HA must be a finite positive number or None.")
    total_detected_trees = sum(max(0, int(zone.get("tree_count", 0))) for zone in zone_json)
    if mapped_area_configured:
        missing_zone_ids = [str(zone["zone_id"]) for zone in zone_json if str(zone["zone_id"]) not in areas]
        if missing_zone_ids:
            raise ValueError("Per-zone area mapping is missing zone IDs: " + ", ".join(missing_zone_ids))
    elif (UNIFORM_ZONE_AREA_HA is None and not USE_SPACING_BASED_ZONE_AREA_DEFAULT
          and TOTAL_FARM_AREA_HA is not None and total_detected_trees <= 0):
        raise ValueError("Cannot distribute TOTAL_FARM_AREA_HA because no detected trees are assigned to zones.")
    for nutrient, rate in MAX_NUTRIENT_KG_HA_PER_YEAR.items():
        if nutrient not in ("Mg", "N") or not np.isfinite(float(rate)) or float(rate) < 0:
            raise ValueError(f"Maximum annual elemental nutrient target for {nutrient} must be nonnegative.")
    products = FERTILIZER_NUTRIENT_BY_PRODUCT
    for product in products:
        price = FERTILIZER_PRICE_PER_KG.get(product)
        if price is not None and (not np.isfinite(float(price)) or float(price) < 0):
            raise ValueError(f"Product price for {product} must be nonnegative.")
    for zone in zone_json:
        zone_min_vari = zone.get("vari", {}).get("min_vari")
        is_lowest_min_vari_zone = (
            zone_min_vari is not None
            and np.isfinite(zone_min_vari)
            and lowest_zone_min_vari is not None
            and np.isclose(zone_min_vari, lowest_zone_min_vari, rtol=0.0, atol=1e-12)
        )
        zone["vari"]["lowest_min_vari_zone"] = bool(is_lowest_min_vari_zone)
        zone["vari"]["lowest_zone_min_vari_value"] = (
            float(lowest_zone_min_vari) if lowest_zone_min_vari is not None else None
        )
        zone["vari"]["min_vari_review_note"] = (
            "Zone has the lowest valid min VARI; review imagery/trees/pixels."
            if is_lowest_min_vari_zone else None
        )
        zid = str(zone["zone_id"])
        zone_local_confidence = float(zone["confidence"]["confidence"])
        confidence = field_confidence
        mean = zone.get("vari", {}).get("mean_vari")
        percentile_rank = (100.0 * sum(value <= float(mean) for value in valid_means) / valid_zone_count
                           if mean is not None and np.isfinite(mean) and valid_zone_count else None)
        boost_input = p25 if blanket_mode else mean
        if reference is None or reference <= 0 or boost_input is None or not np.isfinite(boost_input):
            deficit = boost = None
        else:
            deficit = max(0.0, 1.0 - float(boost_input) / reference)
            boost = deficit * confidence
        if reference is None or reference <= 0 or mean is None or not np.isfinite(mean):
            zone_specific_deficit = zone_specific_boost = None
        else:
            zone_specific_deficit = max(0.0, 1.0 - float(mean) / reference)
            zone_specific_boost = zone_specific_deficit * confidence
        if zid in areas:
            area = areas[zid]
            area_note = "User-supplied area for this zone."
        elif UNIFORM_ZONE_AREA_HA is not None:
            area = float(UNIFORM_ZONE_AREA_HA)
            area_note = "Configured uniform area applied to every zone."
        elif USE_SPACING_BASED_ZONE_AREA_DEFAULT:
            if DEFAULT_TREE_SPACING_M <= 0:
                raise ValueError("DEFAULT_TREE_SPACING_M must be positive.")
            tree_count = max(0, int(zone.get("tree_count", 0)))
            area = tree_count * (DEFAULT_TREE_SPACING_M ** 2) / 10000.0
            area_note = (
                f"Estimated as {tree_count} detected trees × {DEFAULT_TREE_SPACING_M:g} m × "
                f"{DEFAULT_TREE_SPACING_M:g} m per tree ÷ 10,000 = {area:.6f} ha."
            )
        elif TOTAL_FARM_AREA_HA is not None:
            tree_count = max(0, int(zone.get("tree_count", 0)))
            area = (float(TOTAL_FARM_AREA_HA) * tree_count / total_detected_trees
                    if total_detected_trees > 0 else None)
            area_note = "Fallback: total farm area split among zones in proportion to detected tree counts."
        else:
            area = None
            area_note = "Provide zone ground area in hectares; image pixels are not converted to ground area."
        recommendation = {
            "reference_zone_mean_vari": reference,
            "reference_zone_mean_vari_percentile": 95,
            "reference_zone_ids": reference_zone_ids_used,
            "reference_fallback_all_zones": reference_fallback,
            "zone_confidence_run_flags": confidence_run_flags,
            "median_leaf_pixels_per_tree": median_leaf_pixels_per_tree,
            "p25_zone_mean_vari": p25,
            "p95_p25_vari_gap": p95_p25_gap,
            "zone_mean_vari_percentile_rank": percentile_rank,
            "blanket_vari_gap_cutoff": MIN_P95_P25_VARI_GAP_FOR_BLANKET_BOOST,
            "boost_mode": "blanket_p25" if blanket_mode else "zone_specific",
            "boost_vari_input": boost_input,
            "zone_confidence": confidence,
            "zone_local_confidence": zone_local_confidence,
            "overall_field_confidence": field_confidence,
            "zone_specific_vari_deficit_fraction": zone_specific_deficit,
            "zone_specific_boost_fraction": zone_specific_boost,
            "vari_deficit_fraction": deficit,
            "boost_fraction": boost,
            "zone_area_ha": area,
            "fertilizers": {},
            "area_note": area_note,
        }
        for product, nutrient in products.items():
            max_nutrient_rate = MAX_NUTRIENT_KG_HA_PER_YEAR.get(nutrient)
            if max_nutrient_rate is None:
                continue
            if nutrient not in ("Mg", "N"):
                raise ValueError(f"Set FERTILIZER_NUTRIENT_BY_PRODUCT for {product} to 'Mg' or 'N'.")
            annual_rate = float(max_nutrient_rate) * boost if boost is not None else None
            annual_total = annual_rate * area if area is not None and annual_rate is not None else None
            item = {
                "base_rate_kg_ha_per_year": float(max_nutrient_rate),
                "boost_added_rate_kg_ha_per_year": annual_rate,
                "boosted_rate_kg_ha_per_year": annual_rate,
                "recommended_nutrient": nutrient,
                "recommended_nutrient_kg_ha_per_year": annual_rate,
                "annual_total_kg": annual_total,
                "kg_per_palm_per_year": annual_total / zone["tree_count"]
                    if annual_total is not None and zone.get("tree_count", 0) > 0 else None,
            }
            item[f"{nutrient.lower()}_kg_ha_per_year"] = annual_rate
            item[f"{nutrient.lower()}_total_kg_per_year"] = annual_total
            recommendation["fertilizers"][product] = item
        zone["recommendation"] = recommendation
    return reference

# ## Weather-based flight confidence (L6)
#
# This section uses the four-factor rule from `vari_confidence.ipynb`: lowest sun elevation × cloud × wet-leaf/rain × recent-water (P30). The score is calculated before zone confidence, then feeds the existing zone and field confidence calculation. WeatherKit is used when configured; NASA POWER supplies documented fallbacks. Unknown inputs remain unknown and make the result non-final.
#
# Most thresholds in the source notebook are marked GUESS and need calibration. Enter the actual flight times and farm coordinates in Configuration before relying on this score.


# ===== Notebook code cell 16 =====
# Copied from weather/l6_vari_confidence.py, weather/krops_rules.py, weather/nasa_power_job.py, weather/solar.py on feat/weather-vari-confidence (6175bca); keep in sync.

# The confidence rule (l6_vari_confidence.py)
SCHEMA_VERSION = 1
TZ = ZoneInfo("Asia/Jakarta")
HOUR = timedelta(hours=1)

SUN_BANDS = ((45.0, 1.0, "high"), (30.0, 0.85, "medium"), (15.0, 0.6, "low"))  # GUESS – no source yet: lowest sun °
SUN_VERY_LOW = 0.3            # GUESS – no source yet: sun under the last band
SUN_UNKNOWN = 0.6             # GUESS – no source yet: flight time or place unknown
CLOUD_CLEAR_MAX = 0.2         # GUESS – no source yet: cloud cover up to this is a clear sky
CLOUD_OVERCAST_MIN = 0.8      # GUESS – no source yet: from this on, overcast
CLOUD_FACTOR = {"clear": 1.0, "broken": 0.7, "overcast": 0.85, "unknown": 0.8}  # GUESS – no source yet
CLEARNESS_CLEAR_MIN = 0.7     # GUESS – no source yet: NASA POWER ALLSKY ÷ FAO-56 clear sky for a clear day
CLEARNESS_OVERCAST_MAX = 0.35 # GUESS – no source yet: … and for an overcast one
WET_HOURS = 6                 # GUESS – no source yet: rain this close before the flight can leave the leaves wet
WET_RAIN_MM = 0.5             # GUESS – no source yet
WET_FACTOR = {"dry": 1.0, "rain_before": 0.7, "heavy_day_before": 0.85, "unknown": 0.85}  # GUESS – no source yet
WATER_FACTOR = {"ok": 1.0, "fallback": 0.9, "water_stress": 0.6, "unknown": 0.75}  # GUESS – no source yet; how far L6
                                                                                   # lowers water_stress is a team value
LABELS = ((0.8, "tinggi"), (0.5, "sedang"))  # GUESS – no source yet; anything lower is "rendah"

# Decided L5 and L8 lines, and the P30 rule's settings (krops_rules.py)
P30_THRESHOLD_MM = 60.0     # L5: water_stress when P30 is under this (proposal)
P30_WINDOW_DAYS = 30
P30_MAX_AGE_DAYS = 7        # a window ending earlier than this many days ago is too old (proposed)
P30_SOURCE_ORDER = ("weatherkit-history", "nasa-power")  # proposed; on 9 Oct NASA POWER came first

SUSPECT_RATIO = 3.0         # NASA POWER monthly check (proposed)
SUSPECT_MIN_DAYS = 5        # GUESS – no source yet: a month with fewer NASA POWER days can't be judged (unchecked);
                            # at 5 or less, last month's checked window stays recent enough while a new month fills in
NORMAL_PERIOD = (1991, 2020)
HEAVY_RAIN_MM = 25.0        # L8 Stage B: a day over this is heavy (proposal)

# Weather sources (nasa_power_job.py)
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RAIN = "PRECTOTCORR"
SUN = "ALLSKY_SFC_SW_DWN"  # logged only (design doc), no rule reads it
GRID_LAT, GRID_LON = 0.5, 0.625  # NASA POWER rain grid (MERRA-2)
WK_AREA = 0.1                    # WeatherKit P30 area, about 11 km: WeatherKit resolves about 10 km
RECENT_LOOKBACK_DAYS = 45        # recent rain starts on the 1st of the month holding today − 45: last month is whole

# Sun (solar.py)
SOLAR_CONSTANT_MJ_MIN = 0.0820          # Gsc, MJ m-2 min-1 (FAO-56 eq. 21)


factors = pd.DataFrame(
    [("sun", f"lowest elevation ≥ {floor:g}°", value) for floor, value, _ in SUN_BANDS]
    + [("sun", f"under {SUN_BANDS[-1][0]:g}°", SUN_VERY_LOW), ("sun", "unknown", SUN_UNKNOWN)]
    + [("cloud", f"{name} (clear ≤ {CLOUD_CLEAR_MAX:g}, overcast ≥ {CLOUD_OVERCAST_MIN:g})", value)
       for name, value in CLOUD_FACTOR.items()]
    + [("wet leaves", {"dry": "dry", "rain_before": f"≥ {WET_RAIN_MM:g} mm in {WET_HOURS} h before",
                       "heavy_day_before": f"> {HEAVY_RAIN_MM:g} mm in 24 h before (decided line)",
                       "unknown": "unknown"}[name], value) for name, value in WET_FACTOR.items()]
    + [("water (L5)", {"ok": "not stressed, WeatherKit", "fallback": "not stressed, NASA POWER fallback",
                       "water_stress": f"P30 under {P30_THRESHOLD_MM:g} mm", "unknown": "unknown"}[name], value)
       for name, value in WATER_FACTOR.items()],
    columns=["input", "when", "factor"]).set_index(["input", "when"])
display(factors)
print("Labels: " + " · ".join(f"{name} ≥ {floor}" for floor, name in LABELS) + " · otherwise rendah")

# ===== Notebook code cell 17 =====
# Copied from weather/l6_vari_confidence.py on feat/weather-vari-confidence (6175bca); keep in sync.
def sun_factor(elevation_deg):
    """(factor, class) for the lowest sun elevation during the flight."""
    if elevation_deg is None:
        return SUN_UNKNOWN, "unknown"
    for floor, factor, name in SUN_BANDS:
        if elevation_deg >= floor:
            return factor, name
    return SUN_VERY_LOW, "very_low"


def cloud_class(cover):
    """'clear', 'broken' or 'overcast' from a cloud cover between 0 and 1; None if unknown."""
    if cover is None:
        return None
    if cover <= CLOUD_CLEAR_MAX:
        return "clear"
    return "overcast" if cover >= CLOUD_OVERCAST_MIN else "broken"


def clearness_class(clearness):
    """The same classes from a day's NASA POWER clearness (ALLSKY ÷ FAO-56 clear sky); None if unknown."""
    if clearness is None:
        return None
    if clearness >= CLEARNESS_CLEAR_MIN:
        return "clear"
    return "overcast" if clearness <= CLEARNESS_OVERCAST_MAX else "broken"


def wet_class(last_6h_mm, last_24h_mm):
    """Whether the leaves may still be wet at take-off. Rain that isn't known is never taken as no rain."""
    if last_6h_mm is not None and last_6h_mm >= WET_RAIN_MM:
        return "rain_before"
    if last_24h_mm is not None and last_24h_mm > HEAVY_RAIN_MM:  # strictly over, as in L8 Stage B
        return "heavy_day_before"
    if last_6h_mm is not None and last_24h_mm is not None:
        return "dry"
    return "unknown"


def water_class(p30):
    """From the L5 P30 record on the scan date: 'ok', 'fallback' (NASA POWER), 'water_stress' or 'unknown'."""
    if not p30 or p30.get("water_stress") is None:
        return "unknown"
    if p30["water_stress"]:
        return "water_stress"
    return "ok" if p30.get("p30_quality") == "ok" else "fallback"


def label_for(confidence):
    for floor, name in LABELS:
        if confidence >= floor:
            return name
    return "rendah"


def _num(x, digits=0):
    """A number the Indonesian way: decimal comma."""
    return f"{x:.{digits}f}".replace(".", ",")


def _reason(name, f):
    """The code and an Indonesian sentence for a factor under 1, built only from what is known."""
    c = f["class"]
    if c == "unknown":
        text = {"sun": "Waktu atau lokasi terbang tidak diketahui.",
                "cloud": "Tutupan awan saat terbang tidak diketahui.",
                "wet": "Hujan sebelum terbang tidak diketahui.",
                "water": "P30 tidak diketahui pada tanggal scan."}[name]
    elif name == "sun":
        text = f"Matahari {_num(f['elevation_deg'])}° di atas cakrawala saat terbang: bayangan memanjang."
    elif name == "cloud":
        cover = f" ({f['cover']:.0%})" if f.get("cover") is not None else " (menurut NASA POWER, sepanjang hari)"
        text = (f"Awan sebagian{cover} saat terbang: terang berganti antar-foto." if c == "broken" else
                f"Langit mendung{cover} saat terbang: cahaya rata tapi redup.")
    elif name == "wet":
        text = (f"Hujan {_num(f['last_6h_mm'], 1)} mm dalam {WET_HOURS} jam sebelum terbang: daun mungkin basah."
                if c == "rain_before" else f"Hujan lebat {_num(f['last_24h_mm'])} mm dalam 24 jam sebelum terbang.")
    elif c == "water_stress":
        text = (f"P30 {_num(f['p30']['p30_mm'])} mm, di bawah {_num(P30_THRESHOLD_MM)} mm: tajuk bisa pucat karena "
                "kering, bukan karena hara.")
    else:
        text = "P30 dari NASA POWER (cadangan), sumber yang cenderung terlalu basah."
    return {"code": f"{name}_{c}", "factor": f["value"], "text": text}


def vari_confidence(sun_elevation_deg, cloud, rain, p30):
    """The confidence for one scan from what is known about its flight and its water.

    cloud = {"cover", "class", "source"[, "clearness"]}, rain = {"last_6h_mm", "last_24h_mm", "source"}, p30 = the L5
    record from choose_p30 (or None). Returns the confidence, its label, both components, every factor with its input
    and source, the reasons for each factor under 1, and `final`: False while any input is still unknown.
    """
    sun_value, sun_cls = sun_factor(sun_elevation_deg)
    cloud_cls = cloud.get("class") or "unknown"
    wet_cls = wet_class(rain.get("last_6h_mm"), rain.get("last_24h_mm"))
    water_cls = water_class(p30)
    factors = {
        "sun": {"value": sun_value, "class": sun_cls, "elevation_deg": sun_elevation_deg},
        "cloud": {"value": CLOUD_FACTOR[cloud_cls], "class": cloud_cls} | {k: v for k, v in cloud.items() if k != "class"},
        "wet": {"value": WET_FACTOR[wet_cls], "class": wet_cls} | rain,
        "water": {"value": WATER_FACTOR[water_cls], "class": water_cls, "p30": p30},
    }
    measurement = round(factors["sun"]["value"] * factors["cloud"]["value"] * factors["wet"]["value"], 3)
    confidence = round(measurement * factors["water"]["value"], 3)
    return {
        "confidence": confidence,
        "label": label_for(confidence),
        "final": all(f["class"] != "unknown" for f in factors.values()),
        "components": {"pengukuran": measurement, "air": factors["water"]["value"]},
        "factors": factors,
        "reasons": [_reason(name, f) for name, f in factors.items() if f["value"] < 1],
    }

# ===== Notebook code cell 18 =====
# Copied from weather/solar.py, weather/l6_vari_confidence.py on feat/weather-vari-confidence (6175bca); keep in sync.
def _sin(deg):
    return math.sin(math.radians(deg))


def _cos(deg):
    return math.cos(math.radians(deg))


def _tan(deg):
    return math.tan(math.radians(deg))


def _clamp(x):
    return min(1.0, max(-1.0, x))  # keeps acos/asin defined when rounding lands just outside [-1, 1]


def julian_day(when_utc):
    """Julian day of a timezone-aware datetime, from its Unix time (Swift: date.timeIntervalSince1970)."""
    if when_utc.tzinfo is None:
        raise ValueError("when_utc must be timezone-aware")
    return when_utc.timestamp() / 86400.0 + 2440587.5


def _refraction_deg(elevation):
    """Atmospheric refraction to add to the true elevation (NOAA's piecewise fit)."""
    if elevation > 85.0:
        return 0.0
    if elevation > 5.0:
        t = _tan(elevation)
        arcsec = 58.1 / t - 0.07 / t ** 3 + 0.000086 / t ** 5
    elif elevation > -0.575:
        e = elevation
        arcsec = 1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    else:
        arcsec = -20.772 / _tan(elevation)
    return arcsec / 3600.0


def solar_position(lat, lon, when_utc):
    """Where the sun is, seen from (lat, lon) at when_utc.

    zenith_deg and elevation_deg include atmospheric refraction (where the sun appears, what a camera sees);
    true_zenith_deg leaves it out. azimuth_deg runs clockwise from north. NOAA quotes about one arc-minute of
    accuracy for 1800–2100 while the sun is clear of the horizon.
    """
    jc = (julian_day(when_utc) - 2451545.0) / 36525.0  # Julian centuries since J2000
    minutes_utc = (when_utc.timestamp() % 86400.0) / 60.0  # Swift: use a remainder that stays positive

    mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    centre = (_sin(mean_anom) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + _sin(2 * mean_anom) * (0.019993 - 0.000101 * jc)
              + _sin(3 * mean_anom) * 0.000289)
    omega = 125.04 - 1934.136 * jc
    app_long = mean_long + centre - 0.00569 - 0.00478 * _sin(omega)
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * _cos(omega)
    decl = math.degrees(math.asin(_clamp(_sin(obliq) * _sin(app_long))))

    y = _tan(obliq / 2.0) ** 2
    eq_time_min = 4.0 * math.degrees(
        y * _sin(2 * mean_long) - 2 * ecc * _sin(mean_anom) + 4 * ecc * y * _sin(mean_anom) * _cos(2 * mean_long)
        - 0.5 * y * y * _sin(4 * mean_long) - 1.25 * ecc * ecc * _sin(2 * mean_anom))
    true_solar_min = (minutes_utc + eq_time_min + 4.0 * lon) % 1440.0  # in [0, 1440)
    hour_angle = true_solar_min / 4.0 - 180.0

    cos_zen = _clamp(_sin(lat) * _sin(decl) + _cos(lat) * _cos(decl) * _cos(hour_angle))
    true_zenith = math.degrees(math.acos(cos_zen))
    elevation = 90.0 - true_zenith + _refraction_deg(90.0 - true_zenith)

    denom = _cos(lat) * _sin(true_zenith)
    if abs(denom) < 1e-12:
        azimuth = 180.0  # sun straight overhead (or observer at a pole): azimuth is undefined
    else:
        a = math.degrees(math.acos(_clamp((_sin(lat) * cos_zen - _sin(decl)) / denom)))
        azimuth = (a + 180.0) % 360.0 if hour_angle > 0 else (540.0 - a) % 360.0
    return {
        "zenith_deg": 90.0 - elevation,
        "elevation_deg": elevation,
        "true_zenith_deg": true_zenith,
        "azimuth_deg": azimuth,
    }


def day_of_year(day):
    """1 for 1 January … 365 or 366 (FAO-56's J)."""
    return day.timetuple().tm_yday


def extraterrestrial_daily(lat, doy):
    """Ra, MJ m-2 day-1: the day's sunlight on a level surface at the top of the atmosphere (FAO-56 eqs. 21–25)."""
    phi = math.radians(lat)
    b = 2.0 * math.pi * doy / 365.0
    dr = 1.0 + 0.033 * math.cos(b)                 # inverse relative Earth–Sun distance (eq. 23)
    decl = 0.409 * math.sin(b - 1.39)              # solar declination, radians (eq. 24)
    ws = math.acos(_clamp(-math.tan(phi) * math.tan(decl)))  # sunset hour angle (eq. 25); clamped for polar day/night
    return (24.0 * 60.0 / math.pi * SOLAR_CONSTANT_MJ_MIN * dr
            * (ws * math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.sin(ws)))


def clear_sky_fao56(ra, elevation_m=0.0):
    """Rso, MJ m-2 day-1: clear-sky sunlight at the surface from Ra and the site's elevation (FAO-56 eq. 37)."""
    return (0.75 + 2e-5 * elevation_m) * ra


def lowest_sun(lat, lon, start, end):
    """The lowest sun elevation (degrees) at the start, middle and end of the flight."""
    mid = start + (end - start) / 2
    return round(min(solar_position(lat, lon, t)["elevation_deg"] for t in (start, mid, end)), 1)

# ===== Notebook code cell 19 =====
# Copied from weather/nasa_power_job.py, weather/krops_rules.py on feat/weather-vari-confidence (6175bca); keep in sync.
def _snap(value, step, offset=0.0):
    return math.floor((value + offset) / step + 0.5) * step - offset  # round half up, the same in Swift


def _place_id(lat, lon):
    return f"{'n' if lat >= 0 else 's'}{abs(lat):.3f}_{'e' if lon >= 0 else 'w'}{abs(lon):.4f}"


def grid_cell(lat, lon):
    """The rain grid cell holding a point: its centre (where we query NASA POWER) and a stable id."""
    clat, clon = _snap(lat, GRID_LAT), _snap(lon, GRID_LON, 180.0)
    return {"id": _place_id(clat, clon), "lat": round(clat, 4), "lon": round(clon, 4)}


def weatherkit_area(lat, lon):
    """The ~11 km box (0.1°) whose centre we query WeatherKit at for P30; plots in one box share it."""
    clat, clon = _snap(lat, WK_AREA), _snap(lon, WK_AREA)
    return {"id": _place_id(clat, clon), "lat": round(clat, 4), "lon": round(clon, 4)}


def _spans(days):
    """Runs of consecutive dates: [d1, d2, d3, d7] → [(d1, d3), (d7, d7)]."""
    spans = []
    for d in sorted(days):
        if spans and d == spans[-1][1] + timedelta(days=1):
            spans[-1] = (spans[-1][0], d)
        else:
            spans.append((d, d))
    return spans


def fetch_nasa_power(lat, lon, start, end, params, session, retries=3):
    """Daily values {param: {date: value or None}}, units and sources. Retries on timeouts, 429 and 5xx."""
    query = {"parameters": ",".join(params), "community": "AG", "latitude": lat, "longitude": lon,
             "start": f"{start:%Y%m%d}", "end": f"{end:%Y%m%d}", "format": "JSON"}
    for attempt in range(retries):
        try:
            r = session.get(POWER_URL, params=query, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"NASA POWER answered {r.status_code}", response=r)
            r.raise_for_status()
            payload = r.json()
            break
        except (requests.RequestException, ValueError) as error:
            if attempt == retries - 1:
                raise
            wait = 15 * 2 ** attempt
            log.warning("NASA POWER request failed (%s); retrying in %ss", error, wait)
            time.sleep(wait)

    fill = payload["header"]["fill_value"]
    values = {param: {datetime.strptime(day, "%Y%m%d").date(): (None if v == fill else float(v))
                      for day, v in series.items()}
              for param, series in payload["properties"]["parameter"].items()}
    units = {k: v["units"] for k, v in payload["parameters"].items()}
    if RAIN in params and units.get(RAIN) != "mm/day":  # the cloud fallback asks for sunlight alone
        raise ValueError(f"unexpected rain unit from NASA POWER: {units.get(RAIN)}")
    return values, units, payload["header"].get("sources", [])


def daily_from_pairs(pairs):
    """{date: mm or None} from raw (date, mm) pairs, one entry per date.

    A date given more than once keeps its value when every value agrees, and becomes missing (None) when they
    disagree: there is no telling which one is right. A None next to a value isn't a disagreement.
    """
    values = {}
    for d, v in pairs:
        values.setdefault(d, set())
        if v is not None:
            values[d].add(float(v))
    return {d: (next(iter(v)) if len(v) == 1 else None) for d, v in sorted(values.items())}


class WeatherKit:
    """Minimal WeatherKit REST client: daily values only. Server-side only, the .p8 key never ships in the app."""

    BASE = "https://weatherkit.apple.com"
    MAX_DAYS = 10  # days per daily request (Apple's limit)
    retries = 0    # re-requests for short replies this run

    def __init__(self, team_id, key_id, service_id, key_path, session, tz):
        self.team_id, self.key_id, self.service_id = team_id, key_id, service_id
        self.key = Path(key_path).expanduser().read_text()
        self.session, self.tz = session, tz
        self.calls = 0

    @classmethod
    def from_env(cls, session, tz):
        names = ["WK_TEAM_ID", "WK_KEY_ID", "WK_SERVICE_ID", "WK_KEY_PATH"]
        values = [os.getenv(n) for n in names]
        if not all(values) or not Path(values[3]).expanduser().is_file():
            return None
        return cls(*values, session=session, tz=tz)

    def _token(self):
        import jwt  # only needed when WeatherKit is configured

        now = int(time.time())
        return jwt.encode({"iss": self.team_id, "iat": now, "exp": now + 3600, "sub": self.service_id}, self.key,
                          algorithm="ES256", headers={"kid": self.key_id, "id": f"{self.team_id}.{self.service_id}"})

    def _utc(self, day):
        return datetime.combine(day, datetime.min.time(), self.tz).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _get(self, lat, lon, params):
        r = self.session.get(f"{self.BASE}/api/v1/weather/en/{lat}/{lon}", params=params,
                             headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        self.calls += 1
        r.raise_for_status()
        return r.json()

    def daily(self, lat, lon, start=None, end=None):
        params = {"dataSets": "forecastDaily", "timezone": str(self.tz)}
        if start:
            params |= {"dailyStart": self._utc(start), "dailyEnd": self._utc(end + timedelta(days=1))}
        return self._get(lat, lon, params)

    def hourly(self, lat, lon, start, end):
        """Past hourly values for the hours starting in [start, end) (aware datetimes, one request of at most 240 hours:
        Apple answers 400 to a longer range).

        Returns {hour start (UTC): {"cloud_cover", "rain_mm", "daylight", "condition"}}. An hour returned twice keeps a
        value only if the copies agree (daily_from_pairs), as daily rain does.
        """
        stamp = "%Y-%m-%dT%H:%M:%SZ"
        reply = self._get(lat, lon, {"dataSets": "forecastHourly", "timezone": str(self.tz),
                                     "hourlyStart": start.astimezone(timezone.utc).strftime(stamp),
                                     "hourlyEnd": end.astimezone(timezone.utc).strftime(stamp)})
        hours = []
        for h in reply.get("forecastHourly", {}).get("hours", []):
            t = datetime.fromisoformat(h["forecastStart"].replace("Z", "+00:00"))
            if start <= t < end:
                hours.append((t, h))
        cloud = daily_from_pairs((t, h.get("cloudCover")) for t, h in hours)
        rain = daily_from_pairs((t, h.get("precipitationAmount")) for t, h in hours)
        first = {}
        for t, h in hours:
            first.setdefault(t, h)
        return {t: {"cloud_cover": cloud[t], "rain_mm": rain[t], "daylight": first[t].get("daylight"),
                    "condition": first[t].get("conditionCode")} for t in sorted(first)}

    def _days(self, lat, lon, first, last, start, end):
        """(local date, mm) for the days one request for first–last returns, kept when they fall inside start–end."""
        got = []
        for d in self.daily(lat, lon, first, last).get("forecastDaily", {}).get("days", []):
            local = datetime.fromisoformat(d["forecastStart"].replace("Z", "+00:00")).astimezone(self.tz).date()
            if start <= local <= end:
                got.append((local, d.get("precipitationAmount")))
        return got

    def _retry(self, lat, lon, first, last, start, end, before):
        """One more request for the missing days first–last, with the gap in the middle of the range.

        Apple has dropped the last day of a request (1 Jun 2023, every time) yet returns it from a range where it isn't
        the last day. Returns what the retry gave inside start–end; `before` (the replies so far) is only for the log.
        """
        spare = self.MAX_DAYS - ((last - first).days + 1)
        ask_first, ask_last = first - timedelta(days=spare // 2), last + timedelta(days=spare - spare // 2)
        self.retries += 1
        try:
            got = self._days(lat, lon, ask_first, ask_last, start, end)
        except Exception:
            log.warning("WeatherKit retry %s to %s failed; %s to %s stay missing", ask_first, ask_last, first, last,
                        exc_info=True)
            return []
        wanted = {first + timedelta(days=i) for i in range((last - first).days + 1)}
        recovered = sorted({d for d, v in got if d in wanted and v is not None})
        earlier = dict(before)
        disagree = sorted({d for d, v in got if d in earlier and earlier[d] != v})
        log.warning("WeatherKit retry %s to %s for %s to %s: recovered %d of %d day(s)%s%s", ask_first, ask_last, first,
                    last, len(recovered), len(wanted), f" ({', '.join(d.isoformat() for d in recovered)})" if recovered else "",
                    f"; {len(disagree)} day(s) disagree with an earlier reply and count as missing" if disagree else "")
        return got

    def history(self, lat, lon, start, end):
        """{date: mm or None} from past daily values, 10 days per request (Apple's limit).

        A request that comes back short is logged and retried once per gap (see _retry). Everything WeatherKit returns
        goes through daily_from_pairs: a date given twice counts once if the values agree, and as missing if not.
        """
        pairs, day = [], start
        while day <= end:
            stop = min(day + timedelta(days=self.MAX_DAYS - 1), end)
            got = self._days(lat, lon, day, stop, start, end)
            asked = [day + timedelta(days=i) for i in range((stop - day).days + 1)]
            absent = sorted(set(asked) - {local for local, _ in got})
            if absent:
                log.warning("WeatherKit gave %d of %d days for %s, %s (%s to %s); missing %s", len(asked) - len(absent),
                            len(asked), lat, lon, day, stop, ", ".join(d.isoformat() for d in absent))
            for first, last in _spans(absent):
                got += self._retry(lat, lon, first, last, start, end, before=got)
            pairs += got
            day = stop + timedelta(days=1)
        return daily_from_pairs(pairs)

# ===== Notebook code cell 20 =====
# Copied from weather/l6_vari_confidence.py on feat/weather-vari-confidence (6175bca); keep in sync.
def flight_cloud(hours, start, end):
    """Mean cloud cover over the hours the flight overlaps; None if none of them has a value."""
    covers = [h["cloud_cover"] for t, h in hours.items()
              if t < end and t + HOUR > start and h["cloud_cover"] is not None]
    return round(sum(covers) / len(covers), 3) if covers else None


def rain_before(hours, start, span_hours):
    """Rain in the span_hours before take-off, over every hour that overlaps that span; None if any hour is missing."""
    t = (start - timedelta(hours=span_hours)).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    total = 0.0
    while t < start:
        h = hours.get(t)
        if h is None or h["rain_mm"] is None:
            return None
        total += h["rain_mm"]
        t += HOUR
    return round(total, 2)


def dayparts(reply, flight_day, tz=TZ):
    """(cloud cover 07:00–19:00 on the flight day, rain from 07:00 the day before to 07:00 the flight day).

    From WeatherKit's daily day parts; either is None when the reply lacks it.
    """
    days = {}
    for d in reply.get("forecastDaily", {}).get("days", []):
        days[datetime.fromisoformat(d["forecastStart"].replace("Z", "+00:00")).astimezone(tz).date()] = d
    today, before = days.get(flight_day) or {}, days.get(flight_day - timedelta(days=1)) or {}
    cover = (today.get("daytimeForecast") or {}).get("cloudCover")
    parts = [(before.get("daytimeForecast") or {}).get("precipitationAmount"),
             (before.get("overnightForecast") or {}).get("precipitationAmount")]
    return cover, None if any(p is None for p in parts) else round(sum(parts), 2)


def nasa_clearness(allsky, lat, day):
    """A day's ALLSKY_SFC_SW_DWN over the FAO-56 clear-sky sunlight for that latitude and day; None if unknown."""
    if allsky is None:
        return None
    clear = clear_sky_fao56(extraterrestrial_daily(lat, day_of_year(day)))
    return round(allsky / clear, 3) if clear > 0 else None


def flight_conditions(lat, lon, start, end, *, weatherkit=None, nasa_fetch=fetch_nasa_power, session=None, tz=TZ):
    """Cloud at the flight and rain before it, one source per input.

    WeatherKit hourly first; then WeatherKit's daily day parts; for cloud only, NASA POWER's clearness for the day.
    """
    cloud = {"cover": None, "class": None, "source": None}
    rain = {"last_6h_mm": None, "last_24h_mm": None, "source": None}
    day = start.astimezone(tz).date()
    if weatherkit:
        try:
            first = (start - timedelta(hours=24)).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            hours = weatherkit.hourly(lat, lon, first, end + HOUR)  # from a whole hour: rain_before needs that hour
        except Exception:
            log.warning("WeatherKit hourly unavailable for %s; trying the daily day parts", day, exc_info=True)
            hours = {}
        if hours:
            cover = flight_cloud(hours, start, end)
            if cover is not None:
                cloud = {"cover": cover, "class": cloud_class(cover), "source": "weatherkit-hourly"}
            last_6h, last_24h = rain_before(hours, start, WET_HOURS), rain_before(hours, start, 24)
            if last_6h is not None or last_24h is not None:
                rain = {"last_6h_mm": last_6h, "last_24h_mm": last_24h, "source": "weatherkit-hourly"}
        if cloud["source"] is None or rain["source"] is None:
            try:
                cover, rain_24h = dayparts(weatherkit.daily(lat, lon, day - timedelta(days=1), day), day, tz)
            except Exception:
                log.warning("WeatherKit daily unavailable for %s", day, exc_info=True)
                cover, rain_24h = None, None
            if cloud["source"] is None and cover is not None:
                cloud = {"cover": cover, "class": cloud_class(cover), "source": "weatherkit-daytime"}
            if rain["source"] is None and rain_24h is not None:
                rain = {"last_6h_mm": None, "last_24h_mm": rain_24h, "source": "weatherkit-dayparts"}
    if cloud["source"] is None and nasa_fetch:
        cell = grid_cell(lat, lon)
        try:
            values, _, _ = nasa_fetch(cell["lat"], cell["lon"], day, day, (SUN,), session)
            clearness = nasa_clearness(values.get(SUN, {}).get(day), cell["lat"], day)
        except Exception:
            log.warning("NASA POWER unavailable for %s", day, exc_info=True)
            clearness = None
        if clearness is not None:
            cloud = {"cover": None, "class": clearness_class(clearness), "source": "nasa-power", "clearness": clearness}
    return cloud, rain

# ===== Notebook code cell 21 =====
# Copied from weather/krops_rules.py, weather/nasa_power_job.py, weather/l2_scan_rain.py, weather/l6_vari_confidence.py on feat/weather-vari-confidence (6175bca); keep in sync.
def _present(daily):
    return {d: float(v) for d, v in (daily or {}).items() if v is not None}


def missing_in_p30_window(daily):
    """Days missing inside the 30-day window ending on the latest day with a value (all 30 if no day has one).

    Missing days after that latest day aren't counted: they move the window back, and choose_p30's age limit
    decides whether it is still recent enough.
    """
    rain = _present(daily)
    if not rain:
        return P30_WINDOW_DAYS
    end = max(rain)
    return sum(1 for i in range(P30_WINDOW_DAYS) if end - timedelta(days=i) not in rain)


def compute_p30(daily, source):
    """P30 from one source: the 30 calendar days ending on the latest day with a value.

    All 30 days must be present; otherwise there is no P30 (None), rather than a sum of fewer days.
    """
    rain = _present(daily)
    if not rain:
        return None
    end = max(rain)
    window = [end - timedelta(days=i) for i in range(P30_WINDOW_DAYS - 1, -1, -1)]
    if any(d not in rain for d in window):
        return None
    p30_mm = round(sum(rain[d] for d in window), 1)
    return {
        "p30_mm": p30_mm,
        "p30_start_date": window[0],
        "p30_end_date": end,
        "p30_source": source,
        "water_stress": p30_mm < P30_THRESHOLD_MM,
    }


def choose_p30(sources, today, order=P30_SOURCE_ORDER, max_age_days=P30_MAX_AGE_DAYS):
    """The L5 data-contract record: the first source in `order` with a full window ending ≤ max_age_days ago.

    `sources` maps a source name to its daily rain (NASA POWER already filtered by `suspect_months`).
    p30_quality is "ok" for the first source in the order, "fallback" for a later one, "none" if none qualifies;
    with "none", p30_mm and water_stress are None (unknown), never a silent False. p30_missing_days gives, for every
    source in the order, the days missing inside its latest 30-day window (missing_in_p30_window), so a gap that
    sent P30 to the fallback shows in the record. A window is never completed from another source.
    """
    missing = {name: missing_in_p30_window(sources.get(name)) for name in order}
    for rank, name in enumerate(order):
        record = compute_p30(sources.get(name), name)
        if record and (today - record["p30_end_date"]).days <= max_age_days:
            return record | {"p30_quality": "ok" if rank == 0 else "fallback", "p30_missing_days": missing}
    return {"p30_mm": None, "p30_start_date": None, "p30_end_date": None,
            "p30_source": None, "water_stress": None, "p30_quality": "none", "p30_missing_days": missing}


def monthly_totals(daily):
    """{(year, month): total mm} over the days present."""
    totals = {}
    for d, v in _present(daily).items():
        totals[(d.year, d.month)] = totals.get((d.year, d.month), 0.0) + v
    return totals


def monthly_normals(daily, first=NORMAL_PERIOD[0], last=NORMAL_PERIOD[1]):
    """{calendar month: median monthly total} over the years first–last (default the 1991–2020 normal)."""
    by_month = {}
    for (y, m), total in monthly_totals(daily).items():
        if first <= y <= last:
            by_month.setdefault(m, []).append(total)
    return {m: median(v) for m, v in sorted(by_month.items())}


def suspect_months(nasa, normals, weatherkit=None, ratio=SUSPECT_RATIO):
    """NASA POWER months to drop: too wet against the normal AND against WeatherKit, by either of two tests.

    - Whole month (as before): the total over the days present is over ratio × the full-month normal, and over
      ratio × WeatherKit's total for the month (a month WeatherKit has no data for is judged on the normal alone).
    - Pro-rated, for a month still filling in: the same total against ratio × the normal scaled to the days NASA POWER
      has, and against ratio × WeatherKit's total on those same days (none of them → the normal alone).
    For a whole month the second test is the first one or stricter, so it only ever adds months: no window dropped
    before is kept now. Requiring WeatherKit too keeps a genuinely wet month, which WeatherKit also sees.
    """
    nasa_rain, wk_rain = _present(nasa), _present(weatherkit)
    wk_totals = monthly_totals(wk_rain)
    out = set()
    for (y, m), total in monthly_totals(nasa_rain).items():
        if total > ratio * normals[m] and ((y, m) not in wk_totals or total > ratio * wk_totals[(y, m)]):
            out.add((y, m))
            continue
        days = [d for d in nasa_rain if (d.year, d.month) == (y, m)]
        share = len(days) / calendar.monthrange(y, m)[1]
        wk_same = [wk_rain[d] for d in days if d in wk_rain]
        if total > ratio * normals[m] * share and (not wk_same or total > ratio * sum(wk_same)):
            out.add((y, m))
    return out


def unchecked_months(nasa, min_days=SUSPECT_MIN_DAYS):
    """NASA POWER months with too few days to judge: like suspect months, they can't serve as the fallback."""
    counts = {}
    for d in _present(nasa):
        counts[(d.year, d.month)] = counts.get((d.year, d.month), 0) + 1
    return {month for month, n in counts.items() if n < min_days}


def drop_months(daily, months):
    return {d: v for d, v in daily.items() if (d.year, d.month) not in months}


def _iso_record(record):
    """A P30 record with its dates as ISO strings (None stays None)."""
    if record is None:
        return None
    return record | {k: (record[k].isoformat() if record[k] else None) for k in ("p30_start_date", "p30_end_date")}


def build_recent(rain, normals_mm, weatherkit_rain=None):
    """Checked recent NASA POWER rain: suspect and unchecked months dropped, plus the NASA-only P30 from what is left.

    An unchecked month (fewer than SUSPECT_MIN_DAYS days) can't serve as the fallback until it can be judged.
    """
    normals = {int(m): v for m, v in normals_mm.items()}
    present = {d: v for d, v in rain.items() if v is not None}
    suspect = suspect_months(present, normals, weatherkit_rain)
    unchecked = unchecked_months(present) - suspect
    checked = drop_months(present, suspect | unchecked)
    return {
        "last_day": max(present).isoformat() if present else None,
        "checked_against": "normals and weatherkit" if weatherkit_rain else "normals",
        "suspect_months": [f"{y}-{m:02d}" for y, m in sorted(suspect)],
        "unchecked_months": [f"{y}-{m:02d}" for y, m in sorted(unchecked)],
        "rain_mm": {d.isoformat(): v for d, v in sorted(checked.items())},
        "nasa_p30": _iso_record(compute_p30(checked, "nasa-power")),
    }


def p30_at(sources, scan_date, order=P30_SOURCE_ORDER):
    """The L5 P30 record as it stood on scan_date: each source cut to the days before it, then choose_p30.

    The scan day itself is left out, as the job leaves out today: its rain isn't complete when the drone flies.
    """
    before = {name: {d: v for d, v in (daily or {}).items() if d < scan_date} for name, daily in sources.items()}
    return choose_p30(before, scan_date, order=order)


def scan_p30(lat, lon, scan_day, normals_mm, *, weatherkit=None, nasa_fetch=fetch_nasa_power, session=None):
    """The L5 P30 record as it stood on the scan day: WeatherKit for the plot's area first, checked NASA POWER next.

    The decided rule, unchanged (l2_scan_rain.p30_at → choose_p30). NASA POWER goes through the job's monthly
    check first, here against WeatherKit at the plot's area (the job uses the cell centre).
    """
    first = (scan_day - timedelta(days=RECENT_LOOKBACK_DAYS)).replace(day=1)
    last = scan_day - timedelta(days=1)
    area, cell = weatherkit_area(lat, lon), grid_cell(lat, lon)
    wk_rain = None
    if weatherkit:
        try:
            wk_rain = weatherkit.history(area["lat"], area["lon"], first, last)
        except Exception:
            log.warning("WeatherKit history unavailable before %s", scan_day, exc_info=True)
    checked = None
    if nasa_fetch:
        try:
            values, _, _ = nasa_fetch(cell["lat"], cell["lon"], first, last, (RAIN,), session)
            recent = build_recent(values[RAIN], normals_mm, wk_rain)
            checked = {date.fromisoformat(d): v for d, v in recent["rain_mm"].items()}
        except Exception:
            log.warning("NASA POWER unavailable before %s", scan_day, exc_info=True)
    return _iso_record(p30_at({"weatherkit-history": wk_rain, "nasa-power": checked}, scan_day))

# ===== Notebook code cell 22 =====
# Copied from weather/l6_vari_confidence.py on feat/weather-vari-confidence (6175bca); keep in sync.
def _utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def confidence_for_scan(scan, plot, normals_mm, *, weatherkit=None, nasa_fetch=fetch_nasa_power, session=None,
                        tz=TZ, computed_at=None):
    """The backend's result for one synced scan. `scan` is the manifest the phone uploads (VARI_CONFIDENCE.md);
    `plot` has the plot's lat and lon. The result holds no location."""
    start, end = _utc(scan["flight_start"]), _utc(scan["flight_end"])
    lat, lon = float(plot["lat"]), float(plot["lon"])
    cloud, rain = flight_conditions(lat, lon, start, end, weatherkit=weatherkit, nasa_fetch=nasa_fetch,
                                    session=session, tz=tz)
    p30 = scan_p30(lat, lon, start.astimezone(tz).date(), normals_mm, weatherkit=weatherkit, nasa_fetch=nasa_fetch,
                   session=session)
    result = vari_confidence(lowest_sun(lat, lon, start, end), cloud, rain, p30)
    return {"schema": SCHEMA_VERSION, "scan_id": scan["scan_id"],
            "computed_at": computed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "flight": {"start": scan["flight_start"], "end": scan["flight_end"]}} | result | {"vari": scan.get("vari")}

# ## Step 1 — Load the image and create the ExG vegetation mask


# ===== Notebook code cell 24 =====
image_path = Path(IMAGE_PATH)
model_path = Path(MODEL_PATH)
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path.resolve()}")
if not model_path.exists():
    raise FileNotFoundError(f"CoreML model not found: {model_path.resolve()}")
print("1/7: Loading image and running ExG segmentation...")
original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if original_bgr is None:
    raise ValueError(f"Could not read image: {image_path}")
height, width = original_bgr.shape[:2]
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
exg_mask, segmented_bgr = segment_vegetation_exg(original_bgr)
segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)
cv2.imwrite(OUTPUT_EXG_MASK, exg_mask)
cv2.imwrite(OUTPUT_EXG_SEGMENTED, segmented_bgr)

# Visual checkpoint: source image, binary vegetation mask, and segmented RGB result.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(original_rgb); axes[0].set_title("Input image"); axes[0].axis("off")
axes[1].imshow(exg_mask, cmap="gray", vmin=0, vmax=255); axes[1].set_title("ExG vegetation mask"); axes[1].axis("off")
axes[2].imshow(segmented_rgb); axes[2].set_title("RGB pixels retained by ExG"); axes[2].axis("off")
plt.tight_layout(); plt.close(); plt.close(fig)
print(f"Image size: {width:,} × {height:,} px; vegetation pixels: {int(np.count_nonzero(exg_mask)):,} ({np.mean(exg_mask > 0):.1%})")

# ## Step 2 — Detect and filter palm trees


# ===== Notebook code cell 26 =====
print("2/7: Loading CoreML model and detecting trees on ExG-segmented image...")
model = ct.models.MLModel(str(model_path))
normalized_boxes, scores = parse_coreml_predictions(model, segmented_rgb)
raw_trees = []
for box, score in zip(normalized_boxes, scores):
    bbox = normalized_box_to_pixels(box, width, height)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bw <= 0 or bh <= 0:
        continue
    raw_trees.append({
        "confidence": float(score),
        "bbox": bbox,
        "width": bw,
        "height": bh,
        "area": bw * bh,
        "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        "bbox_normalized": [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height],
    })
# Reference full-tree size from interior trees (not clipped by image borders)
interior_trees = [
    t for t in raw_trees
    if t["bbox"][0] > BORDER_MARGIN_PX
    and t["bbox"][1] > BORDER_MARGIN_PX
    and t["bbox"][2] < (width - BORDER_MARGIN_PX)
    and t["bbox"][3] < (height - BORDER_MARGIN_PX)
]
ref_pool = interior_trees if interior_trees else raw_trees
if ref_pool:
    median_w = float(np.median([t["width"] for t in ref_pool]))
    median_h = float(np.median([t["height"] for t in ref_pool]))
    median_area = float(np.median([t["area"] for t in ref_pool]))
else:
    median_w = median_h = median_area = 0.0
# Keep only trees that are at least 70% of typical width, height, and area
trees = []
skipped_partial = 0
skipped_oversized = 0
for t in raw_trees:
    w_ratio = t["width"] / median_w if median_w > 0 else 1.0
    h_ratio = t["height"] / median_h if median_h > 0 else 1.0
    area_ratio = t["area"] / median_area if median_area > 0 else 1.0
    if (
        w_ratio < MIN_TREE_COMPLETENESS
        or h_ratio < MIN_TREE_COMPLETENESS
        or area_ratio < MIN_TREE_COMPLETENESS
    ):
        skipped_partial += 1
        continue
    if (
        w_ratio > MAX_TREE_DIMENSION_MULTIPLIER
        or h_ratio > MAX_TREE_DIMENSION_MULTIPLIER
        or area_ratio > MAX_TREE_AREA_MULTIPLIER
    ):
        skipped_oversized += 1
        continue
    t["tree_id"] = len(trees) + 1
    trees.append(t)
print(
    f"Detected {len(trees)} valid trees; filtered out {skipped_partial} undersized and "
    f"{skipped_oversized} oversized detections."
)

# Visual checkpoint: show the detections used by downstream VARI and zoning steps.
detection_preview = Image.fromarray(segmented_rgb.copy())
draw_detection_preview = ImageDraw.Draw(detection_preview)
try:
    preview_font = ImageFont.truetype("Arial.ttf", max(12, min(22, width // 100)))
except OSError:
    preview_font = ImageFont.load_default()
for tree in trees:
    box = tree["bbox"]
    draw_detection_preview.rectangle(box, outline=(255, 45, 45), width=max(2, width // 1500))
    draw_detection_preview.text((box[0], max(0, box[1] - 18)), f"T{tree['tree_id']} {tree['confidence']:.2f}", fill="white", stroke_width=2, stroke_fill=(150, 0, 0), font=preview_font)
display(detection_preview)
show_preview_table("Detection filtering summary", ["Measure", "Value"], [
    ["Raw model detections after confidence/NMS", len(raw_trees)],
    ["Kept trees", len(trees)],
    ["Filtered as undersized/partial", skipped_partial],
    ["Filtered as oversized", skipped_oversized],
    ["Confidence threshold", CONF_THRESHOLD],
])

# ## Steps 3–4 — Calculate tree VARI, group zones, and calculate confidence
#
# Step 4 shows the confidence formula, every zone’s intermediate factors, and the tree-count-weighted field confidence. The same shared formula is used later when the final zone recommendations are built.


# ===== Notebook code cell 28 =====
# Score the current photo's flight when the real flight times and location are configured.
WEATHER_FLIGHT_CONFIDENCE = None
FLIGHT_CONFIDENCE = None
FLIGHT_CONFIDENCE_SOURCE = "unassessed fallback"
weather_confidence_record = None
_weather_inputs_ready = all(v is not None for v in (FLIGHT_START_WIB, FLIGHT_END_WIB, FARM_LATITUDE, FARM_LONGITUDE))
if _weather_inputs_ready:
    if FLIGHT_START_WIB.tzinfo is None or FLIGHT_END_WIB.tzinfo is None:
        raise ValueError("FLIGHT_START_WIB and FLIGHT_END_WIB must include a timezone.")
    if FLIGHT_END_WIB <= FLIGHT_START_WIB:
        raise ValueError("Flight end time must be later than flight start time.")
    for _env_path in WEATHER_CONFIDENCE_ENV_PATHS:
        if _env_path.is_file():
            load_dotenv(_env_path)
            break
    TZ = WIB
    LAT, LON = float(FARM_LATITUDE), float(FARM_LONGITUDE)
    session = requests.Session()
    weatherkit = WeatherKit.from_env(session, TZ)
    print("WeatherKit:", "on" if weatherkit else "off (NASA POWER fallback; some factors may be unknown)")
    NORMALS_FILE = OUTPUT_DIR / "climate_normals.json"
    _cell = grid_cell(LAT, LON)
    _cached = json.loads(NORMALS_FILE.read_text()) if NORMALS_FILE.is_file() else {}
    if _cached.get("cell") == _cell["id"]:
        normals_mm = _cached["normals_mm"]
    else:
        _first, _last = NORMAL_PERIOD
        _values, _, _ = fetch_nasa_power(_cell["lat"], _cell["lon"], date(_first, 1, 1), date(_last, 12, 31), (RAIN,), session)
        normals_mm = {str(m): round(v, 1) for m, v in monthly_normals(_values[RAIN], _first, _last).items()}
        if len(normals_mm) < 12:
            raise ValueError(f"NASA POWER rain normals are incomplete: {sorted(normals_mm)}")
        NORMALS_FILE.write_text(json.dumps({"cell": _cell["id"], "period": [_first, _last], "normals_mm": normals_mm}, indent=2))
    _scan = {
        "scan_id": Path(IMAGE_PATH).stem,
        "flight_start": FLIGHT_START_WIB.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flight_end": FLIGHT_END_WIB.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    weather_confidence_record = confidence_for_scan(
        _scan, {"lat": LAT, "lon": LON}, normals_mm, weatherkit=weatherkit, session=session
    )
    WEATHER_FLIGHT_CONFIDENCE = float(weather_confidence_record["confidence"])
    if FLIGHT_CONFIDENCE_OVERRIDE is None:
        FLIGHT_CONFIDENCE = WEATHER_FLIGHT_CONFIDENCE
        FLIGHT_CONFIDENCE_SOURCE = "weather-based L6"
    else:
        FLIGHT_CONFIDENCE = float(FLIGHT_CONFIDENCE_OVERRIDE)
        if not 0.0 <= FLIGHT_CONFIDENCE <= 1.0:
            raise ValueError("FLIGHT_CONFIDENCE_OVERRIDE must be from 0 to 1.")
        FLIGHT_CONFIDENCE_SOURCE = "manual override"
    print(f"Weather flight confidence: {WEATHER_FLIGHT_CONFIDENCE:.3f} ({weather_confidence_record['label']}); "
          f"{'final' if weather_confidence_record['final'] else 'NOT final: one or more factors are unknown'}")
    _factor_rows = []
    for _name, _factor in weather_confidence_record["factors"].items():
        _details = {k: v for k, v in _factor.items() if k not in ("value", "class")}
        _factor_rows.append({"Factor": _name, "Score": _factor["value"], "Class": _factor["class"], "Input/source": str(_details)})
    display(pd.DataFrame(_factor_rows).set_index("Factor"))
    if weather_confidence_record["reasons"]:
        print("Reasons lowering confidence:")
        for _reason_item in weather_confidence_record["reasons"]:
            print(f"  - {_reason_item['text']} (factor {_reason_item['factor']})")
    WEATHER_CONFIDENCE_OUTPUT.write_text(json.dumps(weather_confidence_record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved factor detail: {WEATHER_CONFIDENCE_OUTPUT}")
elif FLIGHT_CONFIDENCE_OVERRIDE is not None:
    FLIGHT_CONFIDENCE = float(FLIGHT_CONFIDENCE_OVERRIDE)
    if not 0.0 <= FLIGHT_CONFIDENCE <= 1.0:
        raise ValueError("FLIGHT_CONFIDENCE_OVERRIDE must be from 0 to 1.")
    FLIGHT_CONFIDENCE_SOURCE = "manual override"
    print(f"Using manual flight confidence override: {FLIGHT_CONFIDENCE:.3f}")
else:
    print("Weather flight confidence skipped: set actual flight times and farm latitude/longitude in Configuration.")
    print(f"Zone confidence will use the configured fallback: {DEFAULT_ZONE_CONFIDENCE:.2f}.")

# ===== Notebook code cell 29 =====
print("3/7: Calculating VARI map and tree-level statistics...")
b, g, r = cv2.split(original_bgr)
b = b.astype(np.float32)
g = g.astype(np.float32)
r = r.astype(np.float32)
denominator = g + r - b
vari_map = np.full((height, width), np.nan, dtype=np.float32)
valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
np.divide(g - r, denominator, out=vari_map, where=valid)
for tree in trees:
    tree["vari"] = stats_for_box(vari_map, exg_mask, tree["bbox"])
    tree["bbox_annotation"] = pixel_bbox_record(tree["bbox"], width, height)
print(f"4/7: Grouping trees into {ZONE_GRID_ROWS}x{ZONE_GRID_COLS} zones (up to {TREES_PER_ZONE} trees/zone)...")
zones = group_trees_into_zones(
    trees,
    grid_rows=ZONE_GRID_ROWS,
    grid_cols=ZONE_GRID_COLS,
    row_tolerance_multiplier=ROW_TOLERANCE_MULTIPLIER,
    max_zone_distance_multiplier=MAX_ZONE_DISTANCE_MULTIPLIER,
    zone_padding_multiplier=ZONE_PADDING_MULTIPLIER,
    image_width=width,
    image_height=height,
)
pre_merge_zone_count = len(zones)
zones = merge_small_adjacent_zones(zones)
if len(zones) < pre_merge_zone_count:
    print(f"Merged {pre_merge_zone_count - len(zones)} undersized zone(s) into adjacent zones.")

# Step 4 confidence preview. The leaf-pixel count below uses the same union of tree boxes
# intersected with finite VARI pixels and the ExG mask as the final zone-statistics step.
all_tree_leaf_counts = [
    int(tree.get("vari", {}).get("vegetation_pixel_count") or 0)
    for zone in zones for tree in zone.get("trees", [])
    if int(tree.get("vari", {}).get("vegetation_pixel_count") or 0) > 0
]
step4_median_leaf_pixels_per_tree = float(np.median(all_tree_leaf_counts)) if all_tree_leaf_counts else 0.0
step4_flight_confidence = (DEFAULT_ZONE_CONFIDENCE if FLIGHT_CONFIDENCE is None
                           else float(FLIGHT_CONFIDENCE))
step4_flight_source = ("DEFAULT_ZONE_CONFIDENCE stand-in" if FLIGHT_CONFIDENCE is None
                       else FLIGHT_CONFIDENCE_SOURCE)
step4_confidence_rows = []
for zone in zones:
    zx1, zy1, zx2, zy2 = zone["bbox"]
    zone_tree_union = np.zeros((zy2 - zy1, zx2 - zx1), dtype=bool)
    for tree in zone["trees"]:
        tx1, ty1, tx2, ty2 = tree["bbox"]
        lx1, ly1 = max(0, tx1 - zx1), max(0, ty1 - zy1)
        lx2, ly2 = min(zx2 - zx1, tx2 - zx1), min(zy2 - zy1, ty2 - zy1)
        if lx2 > lx1 and ly2 > ly1:
            zone_tree_union[ly1:ly2, lx1:lx2] = True
    zone_vari_crop = vari_map[zy1:zy2, zx1:zx2]
    zone_exg_crop = exg_mask[zy1:zy2, zx1:zx2] > 0
    leaf_pixel_count = int(np.count_nonzero(zone_tree_union & zone_exg_crop & np.isfinite(zone_vari_crop)))
    zone_id = zone["zone_id"]
    override = ZONE_CONFIDENCE_BY_ID.get(str(zone_id), ZONE_CONFIDENCE_BY_ID.get(zone_id))
    parts = calculate_zone_confidence_components(
        len(zone["trees"]), leaf_pixel_count, step4_median_leaf_pixels_per_tree,
        step4_flight_confidence, override,
    )
    zone["_step4_confidence_preview"] = parts
    step4_confidence_rows.append((zone, parts))

show_preview_table(
    "Step 4 — Zone confidence calculation inputs and results",
    ["Zone", "Trees", "Zone ExG/VARI pixels", "Median pixels/tree", "Tree factor", "Coverage ratio", "Leaf factor", "Zone factor", "Flight factor", "Raw → clamped", "Override", "Local confidence"],
    [[f"Z{zone['zone_id']}", len(zone["trees"]), parts["leaf_pixel_count"],
      f"{step4_median_leaf_pixels_per_tree:.2f}", f"{parts['tree_factor']:.4f}",
      f"{parts['leaf_pixel_ratio']:.4f}", f"{parts['leaf_pixel_factor']:.4f}",
      f"{parts['zone_factor']:.4f}", f"{step4_flight_confidence:.4f}",
      f"{parts['raw_confidence']:.4f} → {parts['calculated_confidence']:.4f}",
      "none" if parts["manual_override"] is None else f"{float(parts['manual_override']):.4f}",
      f"{parts['confidence']:.4f}"]
     for zone, parts in step4_confidence_rows],
)
print("Step 4 confidence formula:")
print(f"  Tree factor = min(1, sqrt(tree count ÷ {ZONE_CONFIDENCE_FULL_TREES:g} full-tree reference))")
print(f"  Leaf coverage ratio = zone ExG/VARI pixels ÷ (tree count × {step4_median_leaf_pixels_per_tree:.2f} median pixels/tree)")
print(f"  Leaf factor = min(1, leaf coverage ratio ÷ {ZONE_LEAF_PIXEL_FULL_RATIO:g} full-coverage reference)")
print("  Zone factor = tree factor × leaf factor")
print(f"  Flight factor = {step4_flight_confidence:.4f} ({step4_flight_source})")
print(f"  Calculated local confidence = clamp(flight factor × zone factor, {ZONE_CONFIDENCE_FLOOR:.2f}, 1.00)")
print("  A configured per-zone confidence override replaces the calculated local confidence.")
for zone, parts in step4_confidence_rows:
    n = len(zone["trees"])
    print(f"  Z{zone['zone_id']}: tree factor=min(1,sqrt({n}/{ZONE_CONFIDENCE_FULL_TREES:g}))={parts['tree_factor']:.6f}; "
          f"leaf ratio={parts['leaf_pixel_count']}/({n}×{step4_median_leaf_pixels_per_tree:.6f})={parts['leaf_pixel_ratio']:.6f}; "
          f"leaf factor=min(1,{parts['leaf_pixel_ratio']:.6f}/{ZONE_LEAF_PIXEL_FULL_RATIO:g})={parts['leaf_pixel_factor']:.6f}; "
          f"zone factor={parts['tree_factor']:.6f}×{parts['leaf_pixel_factor']:.6f}={parts['zone_factor']:.6f}; "
          f"raw confidence={step4_flight_confidence:.6f}×{parts['zone_factor']:.6f}={parts['raw_confidence']:.6f}; "
          f"clamped={parts['calculated_confidence']:.6f}; final local={parts['confidence']:.6f}")
step4_total_trees = sum(len(zone["trees"]) for zone in zones)
step4_field_confidence_raw = (
    sum(parts["confidence"] * len(zone["trees"]) for zone, parts in step4_confidence_rows) / step4_total_trees
    if step4_total_trees else 0.0
)
step4_field_confidence = min(1.0, max(ZONE_CONFIDENCE_FLOOR, step4_field_confidence_raw))
step4_weighted_terms = " + ".join(
    f"({parts['confidence']:.6f}×{len(zone['trees'])})" for zone, parts in step4_confidence_rows
)
print("  Overall field confidence = tree-count-weighted mean of local zone confidence:")
print(f"    ({step4_weighted_terms}) ÷ {step4_total_trees} trees = {step4_field_confidence_raw:.6f}")
print(f"    Clamp to [{ZONE_CONFIDENCE_FLOOR:.2f}, 1.00] = {step4_field_confidence:.6f}")

# Data checkpoint: tree VARI records and the spatial groups formed from detected trees.
show_preview_table("Tree-level VARI sample (first 12 trees)", ["Tree", "Zone", "Confidence", "Mean VARI", "Median VARI", "Vegetation pixels"], [
    [f"T{t['tree_id']}", next((f"Z{z['zone_id']}" for z in zones if t in z["trees"]), "—"),
     f"{t['confidence']:.3f}",
     f"{t['vari']['mean_vari']:.4f}" if t["vari"]["mean_vari"] is not None else "N/A",
     f"{t['vari']['median_vari']:.4f}" if t["vari"]["median_vari"] is not None else "N/A",
     t["vari"]["vegetation_pixel_count"]]
    for t in trees[:12]
])
show_preview_table("Zone grouping checkpoint", ["Zone", "Trees", "Approx. row", "Bounding box (px)"], [
    [f"Z{z['zone_id']}", len(z["trees"]), z.get("approx_row"), str(z["bbox"])] for z in zones
])

# ## Step 5 — Calculate zone VARI and apply the fertilizer boost


# ===== Notebook code cell 31 =====
print("5/7: Coloring ExG tree masks by zone and calculating zone statistics...")
try:
    font = ImageFont.truetype("Arial.ttf", max(12, min(24, width // 100)))
except OSError:
    font = ImageFont.load_default()
tree_line_width = max(2, width // 1500)
MASK_OVERLAY_ALPHA = 0.60
# Image 3: CoreML object detections drawn on the ExG-segmented RGB image
coreml_on_exg_img = Image.fromarray(segmented_rgb.copy())
draw_coreml_exg = ImageDraw.Draw(coreml_on_exg_img)
for tree in trees:
    tb = tree["bbox"]
    draw_coreml_exg.rectangle(tb, outline=(255, 40, 40), width=tree_line_width)
    tree_label = f"T{tree['tree_id']} {tree['confidence']:.2f}"
    tx, ty = tb[0], max(0, tb[1] - 18)
    draw_coreml_exg.text(
        (tx, ty), tree_label,
        fill=(255, 255, 255), stroke_width=2, stroke_fill=(150, 0, 0), font=font
    )
coreml_on_exg_img.save(OUTPUT_COREML_ANNOTATED, quality=95)
# Image 4a: Black canvas where ONLY detected tree ExG mask pixels are colored by Zone ID
zone_tree_mask_rgb = np.zeros_like(original_rgb)
# Image 4b: Original WebODM image where detected tree ExG mask pixels are blended by Zone ID
zone_tree_overlay_rgb = original_rgb.copy()
cmap_zones = plt.colormaps["tab20"]
zone_label_positions = []
zone_json = []
zone_exg_seed_masks = []
for zone in zones:
    zb = zone["bbox"]
    zx1, zy1, zx2, zy2 = zb
    tree_records = []
# Pick a unique RGB color for this zone
    rgba = cmap_zones((zone["zone_id"] - 1) % 20 / 20.0)
    zone_color_u8 = np.array([int(round(c * 255)) for c in rgba[:3]], dtype=np.uint8)
    zone_color = tuple(int(v) for v in zone_color_u8)
    zone_exg_seed_mask = np.zeros((height, width), dtype=np.uint8)
# Union mask of member tree boxes so Zone VARI only uses member tree ExG pixels
    tree_box_union = np.zeros((zy2 - zy1, zx2 - zx1), dtype=bool)
    for tree in zone["trees"]:
        tx1, ty1, tx2, ty2 = tree["bbox"]
        tree_records.append({
            "tree_id": tree["tree_id"],
            "confidence": tree["confidence"],
            "bbox": tree["bbox_annotation"],
            "vari": tree["vari"],
        })
# Get ExG mask pixels inside this tree's detection area
        tree_exg_mask = exg_mask[ty1:ty2, tx1:tx2] > 0
        zone_exg_seed_mask[ty1:ty2, tx1:tx2][tree_exg_mask] = 255
# 1) Color the tree's ExG mask pixels on the black canvas
        zone_tree_mask_rgb[ty1:ty2, tx1:tx2][tree_exg_mask] = zone_color_u8
# 2) Alpha-blend the zone color onto the tree's ExG mask pixels on the original image
        orig_crop = zone_tree_overlay_rgb[ty1:ty2, tx1:tx2]
        orig_crop[tree_exg_mask] = (
            orig_crop[tree_exg_mask].astype(np.float32) * (1.0 - MASK_OVERLAY_ALPHA)
            + zone_color_u8.astype(np.float32) * MASK_OVERLAY_ALPHA
        ).astype(np.uint8)
# Mark this tree's region inside the zone union mask
        lx1, ly1 = max(0, tx1 - zx1), max(0, ty1 - zy1)
        lx2, ly2 = min(zx2 - zx1, tx2 - zx1), min(zy2 - zy1, ty2 - zy1)
        if lx2 > lx1 and ly2 > ly1:
            tree_box_union[ly1:ly2, lx1:lx2] = True
    zone_exg_seed_masks.append(zone_exg_seed_mask)
# Calculate Zone VARI strictly from the colored ExG tree pixels in this zone
    vari_crop = vari_map[zy1:zy2, zx1:zx2]
    mask_crop = (exg_mask[zy1:zy2, zx1:zx2] > 0) & tree_box_union
    valid_vals = vari_crop[mask_crop & np.isfinite(vari_crop)]
    if valid_vals.size == 0:
        zone_stats = {
            "vegetation_pixel_count": 0,
            "mean_vari": None, "median_vari": None,
            "std_vari": None, "mad_vari": None,
            "min_vari": None, "max_vari": None,
        }
    else:
        med = float(np.median(valid_vals))
        zone_stats = {
            "vegetation_pixel_count": int(valid_vals.size),
            "mean_vari": float(np.mean(valid_vals)),
            "median_vari": med,
            "std_vari": float(np.std(valid_vals)),
            "mad_vari": float(np.median(np.abs(valid_vals - med))),
            "min_vari": float(np.min(valid_vals)),
            "max_vari": float(np.max(valid_vals)),
        }
    zone_label = f"Z{zone['zone_id']} (n={len(tree_records)})"
    zx, zy = zx1, max(0, zy1 - 28)
    zone_label_positions.append((zx, zy, zone_label, zone_color))
    zone_json.append({
        "zone_id": zone["zone_id"],
        "approx_row": zone["approx_row"],
        "tree_count": len(tree_records),
        "bbox": pixel_bbox_record(zb, width, height),
        "vari": zone_stats,
        "trees": tree_records,
    })
reference_mean = add_zone_boost_and_fertilizer(zone_json, image_size=(width, height))
zones_only_img = Image.fromarray(zone_tree_mask_rgb)
draw_zones_only = ImageDraw.Draw(zones_only_img)
combined_annotated_img = Image.fromarray(zone_tree_overlay_rgb)
draw_combined = ImageDraw.Draw(combined_annotated_img)

# Data checkpoint: zone VARI, confidence, boost mode, and applied nutrient supplements.
show_preview_table("Zone statistics and boost preview", ["Zone", "Trees", "Mean VARI", "Median", "Min", "Max", "Field confidence", "Boost mode", "Mg kg/zone/year", "N kg/zone/year"], [
    [f"Z{z['zone_id']}", z["tree_count"],
     *[f"{z.get('vari', {}).get(k):.4f}" if z.get("vari", {}).get(k) is not None else "N/A" for k in ("mean_vari", "median_vari", "min_vari", "max_vari")],
     f"{z.get('recommendation', {}).get('overall_field_confidence', 0):.3f}",
     ("Blanket P25" if z.get("recommendation", {}).get("boost_mode") == "blanket_p25" else "Zone-specific"),
     f"{z.get('recommendation', {}).get('fertilizers', {}).get('kieserite_Mg_expensive', {}).get('annual_total_kg', 0):.4f}",
     f"{z.get('recommendation', {}).get('fertilizers', {}).get('urea_N_expensive', {}).get('annual_total_kg', 0):.4f}"]
    for z in zone_json
])

# ## Step 6 — View per-tree VARI heatmaps
#
# Each tile shows the original RGB crop dimmed, with bold quartile colors over a exact ExG mask inside each tree bbox derived from ExG. The exact ExG-selected pixels inside each bounding box are colored, matching the zone overlay mask. All valid ExG-masked VARI pixels from the detected trees are pooled to find the 25th, 50th, and 75th percentile cutoffs. Those shared cutoffs divide values into four color bands (quartiles), so colors differ clearly while remaining comparable between trees. Black pixels are outside the ExG mask.


# ===== Notebook code cell 33 =====
# Build a four-color VARI heatmap for each detected tree, using shared pixel quartiles.
from matplotlib.colors import ListedColormap, BoundaryNorm

# Assign one quartile color to each tree using its mean VARI value.
valid_tree_means = np.array([
    float(tree["vari"]["mean_vari"]) for tree in trees
    if tree.get("vari", {}).get("mean_vari") is not None
    and np.isfinite(tree["vari"]["mean_vari"])
], dtype=np.float32)
if valid_tree_means.size:
    tree_vari_q25, tree_vari_q50, tree_vari_q75 = np.quantile(valid_tree_means, [0.25, 0.50, 0.75])
else:
    tree_vari_q25 = tree_vari_q50 = tree_vari_q75 = 0.0
tree_vari_quartile_cutoffs = [float(tree_vari_q25), float(tree_vari_q50), float(tree_vari_q75)]
print("Tree-mean VARI quartile cutoffs:")
print(f"  Q1/Q2 boundary (25th percentile): {tree_vari_q25:.4f}")
print(f"  Q2/Q3 boundary (50th percentile): {tree_vari_q50:.4f}")
print(f"  Q3/Q4 boundary (75th percentile): {tree_vari_q75:.4f}")

# One solid color per tree: low VARI is red; high VARI is green.
tree_vari_colors = ["#d73027", "#fc8d59", "#fee08b", "#1a9850"]
tree_vari_cmap = ListedColormap(tree_vari_colors, name="TreeMeanVARI_Quartiles")
tree_vari_norm = BoundaryNorm(np.arange(5), tree_vari_cmap.N)
TREE_VARI_HEATMAP_ALPHA = 0.65  # Slightly bolder quartile color while keeping the tree visible.
tree_heatmap_data = []
for tree in trees:
    x1, y1, x2, y2 = map(int, tree["bbox"])
    crop_vari = vari_map[y1:y2, x1:x2]
    # Use precisely the ExG pixels inside this detected tree bbox, matching the zone overlay.
    exg_seed_mask = exg_mask[y1:y2, x1:x2] > 0
    overlay_mask = exg_seed_mask & np.isfinite(crop_vari)
    tree_mean = tree.get("vari", {}).get("mean_vari")
    if tree_mean is not None and np.isfinite(tree_mean):
        quartile_index = int(np.digitize(float(tree_mean), tree_vari_quartile_cutoffs, right=True))
    else:
        quartile_index = None
    tree_heatmap_data.append((tree, x1, y1, x2, y2, overlay_mask, quartile_index))

tree_heatmap_dir = Path(OUTPUT_TREE_VARI_HEATMAP_DIR)
tree_heatmap_dir.mkdir(parents=True, exist_ok=True)
tree_heatmap_paths = []

for tree, x1, y1, x2, y2, overlay_mask, quartile_index in tree_heatmap_data:
    # Digitize each valid pixel into Q1–Q4 using the shared 25/50/75% cutoffs.
    if quartile_index is None:
        quartile_rgb = np.full((*overlay_mask.shape, 3), 128, dtype=np.uint8)
        quartile_label = "N/A"
    else:
        solid_color = np.array([int(tree_vari_colors[quartile_index][j:j+2], 16) for j in (1, 3, 5)], dtype=np.uint8)
        quartile_rgb = np.broadcast_to(solid_color, (*overlay_mask.shape, 3))
        quartile_label = f"Q{quartile_index + 1}"
    # Dim the original crop so the quartile colors stand out while image detail remains visible.
    original_crop = original_rgb[y1:y2, x1:x2].copy()
    original_crop = (original_crop.astype(np.float32) * 0.42).astype(np.uint8)
    # Apply bold quartile colors across the filled canopy mask on the dimmed original crop.
    original_crop[overlay_mask] = (
        original_crop[overlay_mask].astype(np.float32) * (1.0 - TREE_VARI_HEATMAP_ALPHA)
        + quartile_rgb[overlay_mask].astype(np.float32) * TREE_VARI_HEATMAP_ALPHA
    ).astype(np.uint8)
    heatmap_path = tree_heatmap_dir / f"tree_{int(tree['tree_id']):03d}_vari.png"
    Image.fromarray(original_crop).save(heatmap_path)
    tree["vari_heatmap"] = str(heatmap_path.resolve())
    tree_heatmap_paths.append((tree, heatmap_path))

if tree_heatmap_paths:
    columns = min(5, len(tree_heatmap_paths))
    rows = int(math.ceil(len(tree_heatmap_paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.25), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (tree, heatmap_path) in zip(axes.flat, tree_heatmap_paths):
        ax.imshow(plt.imread(heatmap_path), interpolation="nearest")
        mean_vari = tree.get("vari", {}).get("mean_vari")
        mean_label = f"{mean_vari:.3f}" if mean_vari is not None else "N/A"
        ax.set_title(f"T{tree['tree_id']} | mean {mean_label}", fontsize=9)
    quartile_mappable = cm.ScalarMappable(cmap=tree_vari_cmap, norm=tree_vari_norm)
    quartile_mappable.set_array([])
    cbar = fig.colorbar(
        quartile_mappable, ax=axes.ravel().tolist(), orientation="horizontal",
        fraction=0.025, pad=0.025, boundaries=np.arange(5), ticks=[0.5, 1.5, 2.5, 3.5],
    )
    cbar.ax.set_xticklabels(["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    cbar.set_label(
        f"Tree mean VARI quartiles | boundaries: {tree_vari_q25:.3f}, {tree_vari_q50:.3f}, {tree_vari_q75:.3f}"
    )
    fig.suptitle("One color per tree — assigned by tree mean VARI quartile", fontsize=15, fontweight="bold")
    plt.savefig(OUTPUT_TREE_VARI_HEATMAP_MONTAGE, dpi=160, bbox_inches="tight")
    plt.close()
    plt.close(fig)
else:
    print("No tree detections available for per-tree VARI heatmaps.")

print(f"Saved {len(tree_heatmap_paths)} individual heatmaps in: {tree_heatmap_dir.resolve()}")
print(f"Montage: {Path(OUTPUT_TREE_VARI_HEATMAP_MONTAGE).resolve()}")

# ## Step 7 — Apply tree VARI quartile colors across the field
#
# This view places each detected tree's single quartile color back at its position in the full original image. It uses the same exact ExG mask inside each tree bbox and 50% color opacity as the per-tree crops.


# ===== Notebook code cell 35 =====
# Paint each tree's assigned VARI quartile onto its location in the complete field image.
field_quartile_rgb = original_rgb.copy()
field_tree_color_overlay = field_quartile_rgb.copy()
field_tree_claimed = np.zeros((height, width), dtype=bool)
for tree, x1, y1, x2, y2, overlay_mask, quartile_index in tree_heatmap_data:
    if quartile_index is None:
        continue
    crop_h, crop_w = overlay_mask.shape
    field_mask = field_tree_claimed[y1:y2, x1:x2]
    # Match the zoning mask exactly: ExG-selected pixels within the tree bbox.
    # If detections overlap, the first tree keeps shared pixels.
    paint_mask = overlay_mask & ~field_mask
    color_hex = tree_vari_colors[quartile_index].lstrip("#")
    tree_color_rgb = np.array([int(color_hex[j:j + 2], 16) for j in (0, 2, 4)], dtype=np.uint8)
    overlay_crop = field_tree_color_overlay[y1:y2, x1:x2]
    overlay_crop[paint_mask] = (
        overlay_crop[paint_mask].astype(np.float32) * (1.0 - TREE_VARI_HEATMAP_ALPHA)
        + tree_color_rgb.astype(np.float32) * TREE_VARI_HEATMAP_ALPHA
    ).astype(np.uint8)
    field_mask[paint_mask] = True

field_tree_quartile_img = Image.fromarray(field_tree_color_overlay)
field_tree_quartile_img.save(OUTPUT_FIELD_TREE_VARI_QUARTILE, quality=95)
plt.figure(figsize=(14, 10))
plt.imshow(field_tree_quartile_img)
plt.title("Full field — one VARI quartile color per tree", fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.close()
print(f"Full-field tree quartile overlay saved to: {Path(OUTPUT_FIELD_TREE_VARI_QUARTILE).resolve()}")

# ## Step 8 — Create zone annotations, white separators, and zone centers


# ===== Notebook code cell 37 =====
# Draw only the Zone ID text labels (no bounding boxes)
for zx, zy, zone_label, zone_color in zone_label_positions:
    for canvas in (draw_zones_only, draw_combined):
        canvas.text(
            (zx, zy), zone_label,
            fill=zone_color, stroke_width=2, stroke_fill=(0, 0, 0), font=font
        )
# Snapshot a separate plain version with only the small zone ID/tree-count labels.
plain_zone_overlay_img = combined_annotated_img.copy()
plain_zone_overlay_img.save(OUTPUT_PLAIN_ZONE_ANNOTATED, quality=95)

# Keep the existing oil_palm_zones_annotated.jpg output with centered Zone and Boost labels.
boost_font_size = max(16, min(36, width // 60))
try:
    boost_font = ImageFont.truetype("Arial.ttf", boost_font_size)
except OSError:
    boost_font = ImageFont.load_default()
for zone, zone_color in zip(zone_json, [
    tuple(int(round(c * 255)) for c in cmap_zones((z["zone_id"] - 1) % 20 / 20.0)[:3])
    for z in zones
]):
    rec = zone.get("recommendation", {})
    boost = rec.get("boost_fraction")
    if rec.get("boost_mode") == "blanket_p25":
        boost_text = f"Blanket P25: {boost * 100:.1f}%" if boost is not None else "Blanket P25: N/A"
    else:
        boost_text = f"Boost: {boost * 100:.1f}%" if boost is not None else "Boost: N/A"
    box = zone["bbox"]
    center = ((box["x1"] + box["x2"]) // 2, (box["y1"] + box["y2"]) // 2)
    pad = max(5, boost_font_size // 5)
    zone_center = (center[0], center[1] - int(boost_font_size * 0.65))
    boost_center = (center[0], center[1] + int(boost_font_size * 0.65))
    for label, label_center in ((f"Zone {zone['zone_id']}", zone_center), (boost_text, boost_center)):
        text_box = draw_combined.textbbox(label_center, label, font=boost_font, anchor="mm", stroke_width=2)
        draw_combined.rounded_rectangle(
            (text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad),
            radius=max(4, pad), fill=zone_color, outline=(255, 255, 255), width=2,
        )
        draw_combined.text(
            label_center, label, fill=(255, 255, 255), font=boost_font, anchor="mm",
            stroke_width=2, stroke_fill=(0, 0, 0),
        )
white_separator_img = Image.fromarray(original_rgb.copy())
zone_centers = []
for zone, source_zone in zip(zone_json, zones):
    centers = [tree["center"] for tree in source_zone["trees"]]
    if centers:
        zone_centers.append((float(np.mean([p[0] for p in centers])),
                             float(np.mean([p[1] for p in centers]))))
    else:
        box = zone["bbox"]
        zone_centers.append(((box["x1"] + box["x2"]) / 2,
                             (box["y1"] + box["y2"]) / 2))
# Assign each image pixel to the nearest zone's actual ExG tree-mask seed.
# This follows the same colored vegetation pixels as the original zoning
# output, while producing one shared boundary between neighboring zones.
yy, xx = np.ogrid[:height, :width]
nearest_zone = np.zeros((height, width), dtype=np.int16)
nearest_distance = np.full((height, width), np.inf, dtype=np.float32)
for zone_index, (zone_seed_mask, (center_x, center_y)) in enumerate(zip(zone_exg_seed_masks, zone_centers)):
    if np.any(zone_seed_mask):
        non_seed_pixels = np.where(zone_seed_mask > 0, 0, 255).astype(np.uint8)
        distance = cv2.distanceTransform(non_seed_pixels, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    else:
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).astype(np.float32)
    closer = distance < nearest_distance
    nearest_zone[closer] = zone_index
    nearest_distance[closer] = distance[closer]
separator_mask = np.zeros((height, width), dtype=bool)
separator_mask[:, 1:] |= nearest_zone[:, 1:] != nearest_zone[:, :-1]
separator_mask[1:, :] |= nearest_zone[1:, :] != nearest_zone[:-1, :]
separator_width = max(1, int(WHITE_ZONE_BOUNDARY_WIDTH_PX))
if separator_width % 2 == 0:
    separator_width += 1
separator_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (separator_width, separator_width)
)
separator_mask = cv2.dilate(separator_mask.astype(np.uint8), separator_kernel) > 0
white_separator_rgb = original_rgb.copy()
white_separator_rgb[separator_mask] = (255, 255, 255)
white_separator_img = Image.fromarray(white_separator_rgb)
# Start from the zone-colored original-image overlay, then add the small zone IDs.
plain_with_separator_rgb = zone_tree_overlay_rgb.copy()
plain_with_separator_img = Image.fromarray(plain_with_separator_rgb)
plain_with_separator_draw = ImageDraw.Draw(plain_with_separator_img)
for zx, zy, zone_label, zone_color in zone_label_positions:
    plain_with_separator_draw.text(
        (zx, zy), zone_label, fill=zone_color, stroke_width=2, stroke_fill=(0, 0, 0), font=font
    )
plain_with_separator_rgb = np.array(plain_with_separator_img)
plain_with_separator_rgb[separator_mask] = (255, 255, 255)
plain_zone_with_separator_img = Image.fromarray(plain_with_separator_rgb)
plain_zone_with_separator_img.save(OUTPUT_PLAIN_ZONE_WITH_SEPARATOR, quality=95)
zones_only_img.save(OUTPUT_ZONE_ANNOTATED, quality=95)
combined_annotated_img.save(OUTPUT_ANNOTATED_IMAGE, quality=95)
white_separator_img.save(OUTPUT_WHITE_ZONE_ANNOTATED, quality=95)
# Calculate each zone's center point based on the white separator zoning segmentation image.
grid_y, grid_x = np.indices((height, width))
is_white_bg = np.all(original_rgb >= 240, axis=-1)
for zone_index, zone in enumerate(zone_json):
    zone_partition_mask = (nearest_zone == zone_index) & (~separator_mask)
    zone_visible_mask = zone_partition_mask & (~is_white_bg)
    active_mask = (
        zone_visible_mask
        if np.any(zone_visible_mask)
        else (zone_partition_mask if np.any(zone_partition_mask) else (nearest_zone == zone_index))
    )
    cx = float(np.mean(grid_x[active_mask]))
    cy = float(np.mean(grid_y[active_mask]))
    part_mask = (nearest_zone == zone_index)
    cx_part = float(np.mean(grid_x[part_mask]))
    cy_part = float(np.mean(grid_y[part_mask]))
    zone["center_point"] = {
        "x": round(cx, 2),
        "y": round(cy, 2),
        "normalized_x": round(cx / width, 6),
        "normalized_y": round(cy / height, 6),
    }
    zone["center"] = {
        "x": round(cx, 2),
        "y": round(cy, 2),
    }
    zone["partition_center_point"] = {
        "x": round(cx_part, 2),
        "y": round(cy_part, 2),
        "normalized_x": round(cx_part / width, 6),
        "normalized_y": round(cy_part / height, 6),
    }

# Save a second white-separator image with zone IDs placed at their calculated centers.
white_separator_numbers_img = Image.fromarray(white_separator_rgb.copy())
number_draw = ImageDraw.Draw(white_separator_numbers_img)
number_font_size = 3 * max(18, min(42, width // 50))  # 200% larger than the previous zone-ID font.
try:
    number_font = ImageFont.truetype("Arial.ttf", number_font_size)
except OSError:
    number_font = ImageFont.load_default()
for zone in zone_json:
    center = zone["center_point"]
    number_draw.text(
        (int(center["x"]), int(center["y"])), f"Z{zone['zone_id']}",
        fill=(0, 0, 0), stroke_width=max(2, number_font_size // 9),
        stroke_fill=(255, 255, 255), font=number_font, anchor="mm",
    )
white_separator_numbers_img.save(OUTPUT_WHITE_ZONE_NUMBERS, quality=95)

# Visual checkpoint: compare the clean white-boundary image with the zone overlay and annotations.
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
axes[0].imshow(zones_only_img); axes[0].set_title("Zone pixels only"); axes[0].axis("off")
axes[1].imshow(combined_annotated_img); axes[1].set_title("Zones + tree/boost annotations"); axes[1].axis("off")
axes[2].imshow(white_separator_img); axes[2].set_title("White zone boundaries"); axes[2].axis("off")
plt.tight_layout(); plt.close(); plt.close(fig)
show_preview_table("Zone center points", ["Zone", "Center x (px)", "Center y (px)", "Normalized x", "Normalized y"], [
    [f"Z{z['zone_id']}", z["center_point"]["x"], z["center_point"]["y"], z["center_point"]["normalized_x"], z["center_point"]["normalized_y"]]
    for z in zone_json
])

# ## Step 9 — Save per-zone image crops


# ===== Notebook code cell 39 =====
zone_base_crops_dir = Path(OUTPUT_ZONE_BASE_CROPS_DIR)
zone_exg_crops_dir = Path(OUTPUT_ZONE_EXG_CROPS_DIR)
zone_base_crops_dir.mkdir(parents=True, exist_ok=True)
zone_exg_crops_dir.mkdir(parents=True, exist_ok=True)
for zone in zone_json:
    box = zone["bbox"]
    crop_bounds = (
        max(0, int(box["x1"])), max(0, int(box["y1"])),
        min(width, int(box["x2"])), min(height, int(box["y2"])),
    )
    if crop_bounds[2] <= crop_bounds[0] or crop_bounds[3] <= crop_bounds[1]:
        zone["base_image_crop"] = None
        zone["exg_image_crop"] = None
        continue
    zone_name = f"zone_{int(zone['zone_id']):02d}.jpg"
    base_crop_path = zone_base_crops_dir / zone_name
    exg_crop_path = zone_exg_crops_dir / zone_name
    Image.fromarray(original_rgb).crop(crop_bounds).save(base_crop_path, quality=95)
    Image.fromarray(segmented_rgb).crop(crop_bounds).save(exg_crop_path, quality=95)
    zone["base_image_crop"] = str(base_crop_path.resolve())
    zone["exg_image_crop"] = str(exg_crop_path.resolve())

# Visual checkpoint: show base-image and ExG crops side by side for each zone.
if zone_json:
    fig, axes = plt.subplots(len(zone_json), 2, figsize=(12, max(4, 3.5 * len(zone_json))), squeeze=False)
    for row, zone in enumerate(zone_json):
        base_path = zone.get("base_image_crop")
        exg_path = zone.get("exg_image_crop")
        if base_path and Path(base_path).exists():
            axes[row, 0].imshow(plt.imread(base_path))
        axes[row, 0].set_title(f"Z{zone['zone_id']} — base crop"); axes[row, 0].axis("off")
        if exg_path and Path(exg_path).exists():
            axes[row, 1].imshow(plt.imread(exg_path))
        axes[row, 1].set_title(f"Z{zone['zone_id']} — ExG crop"); axes[row, 1].axis("off")
    plt.tight_layout(); plt.close(); plt.close(fig)

# ## Step 10 — Write tables, CSV files, plots, and JSON output
# The next cell prints **Separate VARI Supplement Product-Mass Conversion (annual)**. Dolomite and Kieserite each default to a base of 1.0 kg/palm/year; a dose entered for that product replaces its default using dose per tree × applications per year. The **Base product** and **Base+supplement** columns show that base separately from any zone VARI boost. The two Mg products are alternatives; choose one rather than adding both rows together. The output prints each base-rate source. The same base and combined amounts are saved in `oil_palm_zone_fertilizer_product_mass.csv` and, in each zone record and the separate product bag estimate array in the JSON. Run the notebook from the beginning to refresh these outputs.
#
# A separate **Zone Fertilizer Bag Purchase Estimate** table reports per-zone bag equivalents based on the combined base + VARI supplement product quantity. If that combined zone total is unavailable, it falls back to supplement kg/palm/year × tree count + base product kg/zone/year. Bag sizes are configurable; the field total is rounded up after summing zones.


# ===== Notebook code cell 41 =====
# Save CSV and print summary table
save_zone_vari_csv(zone_json, OUTPUT_ZONE_CSV)
print_zone_vari_table(zone_json)
product_mass_rows = build_and_print_product_mass_table(zone_json)
product_bag_rows = build_and_print_zone_bag_purchase_table(product_mass_rows)
save_product_mass_csv(product_mass_rows, OUTPUT_PRODUCT_MASS_CSV)
save_vari_distribution_plot(zone_json, OUTPUT_VARI_DISTRIBUTION)
# Save JSON
result = {
    "metadata": {
        "source_image": str(image_path),
        "image_width_px": width,
        "image_height_px": height,
        "exg_method": "ExG = 2G - R - B; min-max normalized; Otsu threshold; 3x3 opening/closing",
        "vari_formula": "(G - R) / (G + R - B)",
        "vari_distribution_plot": str(Path(OUTPUT_VARI_DISTRIBUTION).resolve()),
        "white_separator_zone_annotation_image": str(Path(OUTPUT_WHITE_ZONE_ANNOTATED).resolve()),
        "white_separator_zone_numbers_image": str(Path(OUTPUT_WHITE_ZONE_NUMBERS).resolve()),
        "plain_zone_overlay_with_separator": str(Path(OUTPUT_PLAIN_ZONE_WITH_SEPARATOR).resolve()),
        "field_tree_vari_quartile_overlay": str(Path(OUTPUT_FIELD_TREE_VARI_QUARTILE).resolve()),
        "per_tree_vari_heatmaps": {"directory": str(Path(OUTPUT_TREE_VARI_HEATMAP_DIR).resolve()), "montage": str(Path(OUTPUT_TREE_VARI_HEATMAP_MONTAGE).resolve()), "mask": "finite VARI pixels intersected with the current ExG mask across each full tree bounding-box crop; quartile colors overlay the ExG-segmented RGB crop", "color_scale": "four shared quartile bins using detected-tree mean VARI values; cutoff values are the pooled 25th, 50th, and 75th percentiles; colors blend over RGB at 0.72 alpha"},
        "zone_boost_formula": "Bi=max(0, 1-V_input/Vref)*zone_confidence; Vref/P25 use valid reference-zone means (zone factor >= configured threshold), falling back to all valid zones if none qualify; if P95-P25 gap <= configured cutoff, V_input=P25 for every zone, otherwise V_input=each zone mean",
        "zone_specific_boost_formula": "B_zone=max(0, 1-zone_mean_vari/Vref)*zone_confidence; shown for comparison even when P25 blanket mode is applied",
        "p25_zone_mean_vari": zone_json[0]["recommendation"].get("p25_zone_mean_vari") if zone_json else None,
        "p95_p25_vari_gap": zone_json[0]["recommendation"].get("p95_p25_vari_gap") if zone_json else None,
        "minimum_p95_p25_gap_for_blanket_boost": MIN_P95_P25_VARI_GAP_FOR_BLANKET_BOOST,
        "zone_boost_mode": zone_json[0]["recommendation"].get("boost_mode") if zone_json else "zone_specific",
        "zone_mean_vari_percentile_rank_definition": "100 × number of valid zone means less than or equal to this zone mean ÷ number of valid zones; empirical rank within this image only",
        "fertilizer_boost_formula": "Additional nutrient supplement: Mg_i=Bi*Mg_max; N_i=Bi*N_max; annual zone supplement kg=annual supplement kg/ha*zone area ha",
        "reference_zone_mean_vari": reference_mean,
        "lowest_zone_min_vari_value": (
            zone_json[0].get("vari", {}).get("lowest_zone_min_vari_value")
            if zone_json else None
        ),
        "lowest_zone_min_vari_flag_definition": (
            "Each zone tied for the lowest finite min_vari is flagged for review; this is an image-screening cue, not a diagnosis."
        ),
        "zone_confidence_rule": (
            "zone_factor=min(1,sqrt(tree_count/full_tree_count))*min(1,leaf_pixel_ratio/full_leaf_ratio); "
            "zone_local_confidence=clamp(flight_confidence*zone_factor,floor,1); "
            "optional ZONE_CONFIDENCE_BY_ID values override zone_local_confidence; "
            "overall_field_confidence=clamp(tree_count_weighted_mean(zone_local_confidence),floor,1); "
            "all zone boosts use the single overall_field_confidence"
        ),
        "overall_field_confidence": (
            zone_json[0]["recommendation"].get("overall_field_confidence") if zone_json else None
        ),
        "overall_field_confidence_raw": (
            zone_json[0]["confidence"].get("overall_field_confidence_raw") if zone_json else None
        ),
        "overall_field_confidence_aggregation": "sum(zone_local_confidence × tree_count) ÷ sum(tree_count), clamped to [0.10, 1.00]",
        "zone_confidence_parameters": {
            "full_tree_count": ZONE_CONFIDENCE_FULL_TREES,
            "full_leaf_pixel_ratio": ZONE_LEAF_PIXEL_FULL_RATIO,
            "reference_min_zone_factor": ZONE_REFERENCE_MIN_FACTOR,
            "confidence_floor": ZONE_CONFIDENCE_FLOOR,
            "field_confidence_weighting": "tree_count_weighted_mean",
            "suggested_min_reference_zones": MIN_REFERENCE_ZONES_SUGGESTED,
        },
        "flight_confidence": FLIGHT_CONFIDENCE,
        "flight_confidence_source": FLIGHT_CONFIDENCE_SOURCE,
        "weather_flight_confidence": weather_confidence_record,
        "flight_confidence_used": DEFAULT_ZONE_CONFIDENCE if FLIGHT_CONFIDENCE is None else FLIGHT_CONFIDENCE,
        "flight_confidence_stand_in": DEFAULT_ZONE_CONFIDENCE if FLIGHT_CONFIDENCE is None else None,
        "reference_zone_ids": zone_json[0]["recommendation"].get("reference_zone_ids", []) if zone_json else [],
        "reference_fallback_all_zones": zone_json[0]["recommendation"].get("reference_fallback_all_zones") if zone_json else False,
        "zone_confidence_run_flags": zone_json[0]["recommendation"].get("zone_confidence_run_flags", []) if zone_json else [],
        "max_nutrient_targets_kg_ha_per_year": MAX_NUTRIENT_KG_HA_PER_YEAR,
        "fertilizer_nutrient_by_product": FERTILIZER_NUTRIENT_BY_PRODUCT,
        "fertilizer_product_nutrient_fractions": FERTILIZER_PRODUCT_NUTRIENT_FRACTIONS,
        "fertilizer_product_bag_size_kg": FERTILIZER_BAG_SIZE_KG,
        "base_product_rates_kg_per_palm_per_year": FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR,
        "fertilizer_output_scope": "VARI-scaled additional elemental nutrient supplement and its product-mass equivalent; full maintenance fertilizer schedule is not included unless a separate approved product base rate is configured",
        "fertilizer_prices_per_kg": FERTILIZER_PRICE_PER_KG,
        "fertilizer_prices_per_bag_rp": FERTILIZER_PRICE_PER_BAG_RP,
        "configured_total_farm_area_ha": TOTAL_FARM_AREA_HA,
        "effective_total_zone_area_ha": sum(
            zone.get("recommendation", {}).get("zone_area_ha") or 0.0 for zone in zone_json
        ),
        "max_nutrient_totals_for_effective_farm_area_kg_per_year": {
            nutrient: float(rate) * sum(
                zone.get("recommendation", {}).get("zone_area_ha") or 0.0
                for zone in zone_json
            )
            for nutrient, rate in MAX_NUTRIENT_KG_HA_PER_YEAR.items()
            if rate is not None
        },
        "ground_area_source": (
            "user-supplied per-zone area mapping" if load_zone_areas()
            else "configured uniform area per zone" if UNIFORM_ZONE_AREA_HA is not None
            else "detected tree count × configured spacing² (active default)" if USE_SPACING_BASED_ZONE_AREA_DEFAULT
            else "configured total farm area split by detected tree counts" if TOTAL_FARM_AREA_HA is not None
            else "not supplied"
        ),
        "zone_area_spacing_m": DEFAULT_TREE_SPACING_M if USE_SPACING_BASED_ZONE_AREA_DEFAULT else None,
        "zone_area_formula": "zone_area_ha = detected_tree_count × spacing_m² ÷ 10,000; explicit per-zone or uniform areas override this estimate",
        "vari_pixel_filter": "Only finite VARI pixels with non-zero denominator inside ExG mask are included",
        "model_path": str(model_path),
        "confidence_threshold": CONF_THRESHOLD,
        "maximum_tree_dimension_multiplier": MAX_TREE_DIMENSION_MULTIPLIER,
        "maximum_tree_area_multiplier": MAX_TREE_AREA_MULTIPLIER,
        "nms_iou_threshold": IOU_THRESHOLD,
        "trees_per_zone_max": TREES_PER_ZONE,
        "minimum_trees_to_keep_zone": MIN_TREES_TO_KEEP_ZONE,
        "zone_grouping": "Approximate image-row clustering; columns use fixed spatial bins based on median within-row tree spacing so missing detections do not shift subsequent bins; zone bbox encloses detected member boxes only",
        "row_tolerance_multiplier": ROW_TOLERANCE_MULTIPLIER,
        "detected_tree_count": len(trees),
        "zone_count": len(zone_json),
        "coordinate_system": "Pixel coordinates use origin (0,0) at top-left; bbox x1,y1 inclusive-ish and x2,y2 exclusive crop boundary; center_point and center provide (x, y) coordinates for each zone based on the white-separator zoning image. normalized_xyxy and normalized_x/y values are divided by original image width/height.",
        "limitations": [
            "ExG is a heuristic vegetation mask and may include weeds or omit shaded palm leaves.",
            "Zone confidence is a heuristic from tree count and ExG leaf-pixel coverage; notebook parameters marked GUESS require calibration against field observations.",
            "Zone rectangles are generated from spatial grouping of detected trees; they are not guaranteed to match actual plantation blocks or planting rows.",
            "Zone VARI summarizes ExG-selected pixels in the union of member tree detection boxes; pixels in overlapping boxes are counted once.",
            "The optional P25 blanket-boost mode is a configurable spatial rule, not an agronomic diagnosis; validate its VARI gap cutoff for the imagery and management plan.",
            "VARI is an RGB vegetation index, not a direct diagnosis of palm health or fertilizer requirement.",
            "Fertilizer rates and prices are illustrative/configurable and are not agronomic recommendations.",
            "Reported fertilizer boosts and totals are elemental Mg or N quantities; they are not converted to product mass.",
            "Zone area is estimated from detected tree count × assumed spacing² (currently 9 m × 9 m per tree), unless measured per-zone or uniform areas are supplied. Missed detections affect the estimate; it is not a geospatial area measurement.",
        ],
    },
    "zones": zone_json,
    "fertilizer_product_mass_conversions": product_mass_rows,
    "fertilizer_product_bag_estimates": product_bag_rows,
}
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)

# Output checkpoint: preview the distribution graphic and list the principal files written.
display(Image.open(OUTPUT_VARI_DISTRIBUTION))
show_preview_table("Files created in this step", ["Output", "Path"], [
    ["Zone summary CSV", str(Path(OUTPUT_ZONE_CSV).resolve())],
    ["Fertilizer product-mass CSV", str(Path(OUTPUT_PRODUCT_MASS_CSV).resolve())],
    ["VARI distribution plot", str(Path(OUTPUT_VARI_DISTRIBUTION).resolve())],
    ["Per-tree VARI heatmap montage", str(Path(OUTPUT_TREE_VARI_HEATMAP_MONTAGE).resolve())],
    ["Full-field tree VARI quartile overlay", str(Path(OUTPUT_FIELD_TREE_VARI_QUARTILE).resolve())],
    ["Individual per-tree heatmaps", str(Path(OUTPUT_TREE_VARI_HEATMAP_DIR).resolve())],
    ["JSON result", str(Path(OUTPUT_JSON).resolve())],
])

# ## Step 11 — Build and display the pipeline preview


# ===== Notebook code cell 43 =====
# Build and save the 3x3 visualization figure with per-tree heatmaps, VARI table, distribution, and boundaries
fig, axes = plt.subplots(3, 3, figsize=(26, 22))
axes[0][0].imshow(original_rgb)
axes[0][0].set_title("1. Original WebODM Image", fontsize=12, fontweight="bold")
axes[0][0].axis("off")
axes[0][1].imshow(segmented_rgb)
axes[0][1].set_title("2. ExG-Segmented RGB Image", fontsize=12, fontweight="bold")
axes[0][1].axis("off")
axes[0][2].imshow(coreml_on_exg_img)
axes[0][2].set_title(f"3. Object Detection on ExG ({len(trees)} trees)", fontsize=12, fontweight="bold")
axes[0][2].axis("off")
axes[1][0].imshow(zones_only_img)
axes[1][0].set_title(f"4. Zoning Only ({len(zones)} zones)", fontsize=12, fontweight="bold")
axes[1][0].axis("off")
axes[1][1].imshow(combined_annotated_img)
axes[1][1].set_title("5. Zones + Tree/Boost Annotations", fontsize=12, fontweight="bold")
axes[1][1].axis("off")
# 6th Panel: per-tree VARI heatmap montage
axes[1][2].imshow(plt.imread(OUTPUT_TREE_VARI_HEATMAP_MONTAGE))
axes[1][2].set_title("6. Per-tree VARI Quartile Colors (one color/tree)", fontsize=12, fontweight="bold")
axes[1][2].axis("off")
# 7th Panel: Zone VARI Heatmap Table (-1.0 to +1.0)
ax_table = axes[2][0]
ax_table.axis("off")
ax_table.set_title("7. Zone VARI Heatmap Table (-1.0 to +1.0)", fontsize=12, fontweight="bold")
if zone_json:
    cmap_vari = plt.colormaps["RdYlGn"]
    norm_vari = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    col_labels = ["Zone", "Trees", "Mean", "Median", "Min", "Max"]
    heatmap_cols = {2, 3, 4, 5}
    cell_text, cell_colors, text_colors = [], [], []
    for r_idx, z in enumerate(zone_json):
        st = z.get("vari", {})
        vals = [st.get("mean_vari"), st.get("median_vari"), st.get("min_vari"), st.get("max_vari")]
        row_str = [
            f"Z{z['zone_id']}",
            str(z["tree_count"]),
            *[f"{v:.3f}" if (v is not None and np.isfinite(v)) else "N/A" for v in vals],
        ]
        raw_row = [None, None, *vals]
        base_bg = (0.95, 0.96, 0.98, 1.0) if r_idx % 2 == 0 else (1.0, 1.0, 1.0, 1.0)
        r_bg, r_fg = [], []
        for c_idx, v in enumerate(raw_row):
            if c_idx == 0:
                z_rgba = cmap_zones((z["zone_id"] - 1) % 20 / 20.0)
                r_bg.append(z_rgba)
                lum = 0.299 * z_rgba[0] + 0.587 * z_rgba[1] + 0.114 * z_rgba[2]
                r_fg.append("#111111" if lum > 0.52 else "#ffffff")
            elif c_idx in heatmap_cols and v is not None and np.isfinite(v):
                rgba = cmap_vari(norm_vari(float(np.clip(v, -1.0, 1.0))))
                r_bg.append(rgba)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                r_fg.append("#111111" if lum > 0.52 else "#ffffff")
            else:
                r_bg.append(base_bg)
                r_fg.append("#111111")
        cell_text.append(row_str)
        cell_colors.append(r_bg)
        text_colors.append(r_fg)
    tbl = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#263238")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.get_text().set_color(text_colors[r - 1][c])
            if c in heatmap_cols or c == 0:
                cell.get_text().set_weight("bold")
    sm = cm.ScalarMappable(cmap=cmap_vari, norm=norm_vari)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_table, orientation="horizontal", fraction=0.05, pad=0.04)
    cbar.set_label("VARI Scale (-1.0 to +1.0)", fontsize=9)
    cbar.set_ticks(np.linspace(-1.0, 1.0, 9))
axes[2][1].imshow(plt.imread(OUTPUT_VARI_DISTRIBUTION))
axes[2][1].set_title("8. Zone VARI Distribution (P25–P95)", fontsize=12, fontweight="bold")
axes[2][1].axis("off")
white_zone_ax = axes[2][2]
white_zone_ax.imshow(plt.imread(OUTPUT_WHITE_ZONE_ANNOTATED))
white_zone_ax.set_title("9. Zone White-separated", fontsize=12, fontweight="bold")
# Keep the exported white-separator image clean; add zone IDs only to this
# interactive pipeline preview.
for zone in zone_json:
    center = zone.get("center_point", {})
    if center.get("x") is None or center.get("y") is None:
        continue
    white_zone_ax.text(
        center["x"], center["y"], f"Z{zone['zone_id']}",
        color="white", fontsize=11, fontweight="bold",
        ha="center", va="center",
        path_effects=[path_effects.withStroke(linewidth=3, foreground="black")],
    )
white_zone_ax.axis("off")
axes[2][2].axis("off")
plt.tight_layout()
fig.savefig(OUTPUT_PIPELINE_PLOT, dpi=200, bbox_inches="tight")
print("\n11/11: Outputs written:")
print(f"  1. ExG mask:                 {Path(OUTPUT_EXG_MASK).resolve()}")
print(f"  2. ExG-segmented RGB:        {Path(OUTPUT_EXG_SEGMENTED).resolve()}")
print(f"  3. Detections on ExG image:  {Path(OUTPUT_COREML_ANNOTATED).resolve()}")
print(f"  4. Zones only (on original): {Path(OUTPUT_ZONE_ANNOTATED).resolve()}")
print(f"  5. Zones + trees (original): {Path(OUTPUT_ANNOTATED_IMAGE).resolve()}")
print(f"  6. Plain zone image:          {Path(OUTPUT_PLAIN_ZONE_ANNOTATED).resolve()}")
print(f"  7. Plain zones + separators:  {Path(OUTPUT_PLAIN_ZONE_WITH_SEPARATOR).resolve()}")
print(f"  6. White-separator zoning:   {Path(OUTPUT_WHITE_ZONE_ANNOTATED).resolve()}")
print(f"  7. White separators + IDs:   {Path(OUTPUT_WHITE_ZONE_NUMBERS).resolve()}")
print(f"  7. Base image zone crops:    {zone_base_crops_dir.resolve()} ({len(zone_json)} images)")
print(f"  8. ExG image zone crops:     {zone_exg_crops_dir.resolve()} ({len(zone_json)} images)")
print(f"  9. Pipeline comparison plot:  {Path(OUTPUT_PIPELINE_PLOT).resolve()}")
print(f" 10. JSON output:              {Path(OUTPUT_JSON).resolve()}")
print(f" 11. Zone VARI CSV:            {Path(OUTPUT_ZONE_CSV).resolve()}")
print(f" 12. Product mass CSV:         {Path(OUTPUT_PRODUCT_MASS_CSV).resolve()}")
print(f" 13. VARI distribution plot:   {Path(OUTPUT_VARI_DISTRIBUTION).resolve()}")
print(f" 14. Per-tree VARI heatmaps:    {Path(OUTPUT_TREE_VARI_HEATMAP_DIR).resolve()} ({len(trees)} images)")
print(f" 15. Tree heatmap montage:     {Path(OUTPUT_TREE_VARI_HEATMAP_MONTAGE).resolve()}")
print(" 15. Zone Center Points (x, y) from white separator zoning:")
display(Image.open(OUTPUT_PLAIN_ZONE_ANNOTATED))
display(Image.open(OUTPUT_PLAIN_ZONE_WITH_SEPARATOR))
display(Image.open(OUTPUT_WHITE_ZONE_NUMBERS))
for z in zone_json:
    cp = z["center_point"]
    print(f"      Zone {z['zone_id']}: x = {cp['x']}, y = {cp['y']} (normalized: x = {cp['normalized_x']:.4f}, y = {cp['normalized_y']:.4f})")
if SHOW_PLOTS:
    plt.close()
else:
    plt.close(fig)

# # Part 2 — Annual NPK/NPKMg comparison
#
# This section adds the standalone `compare_npk_recommendation.py` workflow. It calculates annual fertilizer use from each product’s dose per tree and fertilizer frequency, then compares total nutrient supply per hectare with the reference baseline, converts oxide-label values to elemental nutrients, and estimates separate N, P, K, and Mg supplements when needed. It runs independently of the image-zoning steps above.
#
# Edit the input values in the configuration cell, then run the remaining cells in this section from top to bottom. The final report is shown in a horizontally scrollable, aligned output block.


# ## A. Configure the user fertilizer and reference values
#
# The notebook reads fertilizer composition, dose per tree, and frequency from the submission JSON. For every recognized fertilizer: annual product kg = dosePerTree × fertilizerHistory.freq × detected tree count; divide by farm areaHa for kg/ha/year. Bag equivalents are derived from dose and bag size; they are not user inputs. Configure fallback label/rate, comparison tolerance, baseline, and standalone product label guarantees here.
#
# Standalone P/K options use **SP-36 (36% P₂O₅, 50 kg bag)** and **KCl/MOP (60% K₂O, 50 kg bag)**. Petrokimia Gresik documents the SP-36 label and 50 kg pack ([product specification](https://petrokimia-gresik.com/product/pupuk-sp-36)); PT PPI documents KCl/MOP at 60% K₂O ([product information](https://www.ptppi.co.id/product/detail/50)). Product labels are oxide-based, so the calculation converts the elemental P or K gap to product mass using the same oxide-to-element factors defined above. The bag sizes and label fractions are configurable in the code. KCl contains chloride, so verify it is suitable for the field and use the actual bag guarantee.
#
#
# An age-based NPK product target is also shown separately. It first solves the blend quantity needed to meet all baseline N/P/K targets, converts that annual blend quantity to a per-tree dose, expresses it in 2 kg base-dose units per application, and then applies the Commonize multiplier for the planting-age band. The image does not specify a multiplier for trees younger than 4 years, so the notebook reports that target as unavailable for that age instead of guessing.
#
#
# The Commonize age multiplier is treated as a target. When an annual fertilizer budget is entered, the notebook can reduce each blend option’s multiplier as needed to keep the estimated whole-bag cost (NPK plus the selected zone Mg product and Urea) within the budget plus the configured margin. It never reduces the NPK blend below 1×, which is the amount calculated to meet the baseline; if that minimum exceeds the cap, the notebook reports the conflict. Set `ENABLE_NPK_AGE_BUDGET_CAP = False` to disable budget-based capping and use the full Commonize age multiplier. Set it to `True` to enable the cap. Configure `MAX_NPK_COST_OVER_BUDGET_PERCENT` to change the allowed margin.


# ===== Notebook code cell 46 =====
# Imports for this standalone comparison section (also works if this section is run by itself).
import math
import html as _npk_html
from contextlib import redirect_stdout as _npk_redirect_stdout
from io import StringIO as _NPKStringIO
_NPKHTML = HTML
_npk_display = display
from datetime import date as _npk_date

# Fallback values are used if the submission contains no usable NPK dose.
USER_FERTILIZER_ANALYSIS = "12-12-12"  # Fallback label: N-P2O5-K2O[-MgO], in percent.
USER_FERTILIZER_PRODUCT_NAME = None
PULSE_CHECK_PRODUCT_PRICE_PER_BAG_RP = FERTILIZER_PRICE_PER_BAG_RP

USER_BAGS_PER_HA_PER_YEAR = 0.0  # Derived from dosePerTree; not entered by the farmer.
BAG_SIZE_KG = 50.0
USER_DOSE_PER_TREE_KG_PER_APPLICATION = None
USER_FERTILIZER_APPLICATIONS_PER_YEAR = None
USER_DETECTED_TREE_COUNT = sum(int(zone.get("tree_count", 0)) for zone in zone_json)
USER_FARM_AREA_HA = float(UI_FARM_INPUT.get("areaHa") or TOTAL_FARM_AREA_HA or 1.0)
USER_NPK_PRODUCT_KG_PER_HA_PER_YEAR = 0.0

# Read NPK composition and dose from the submitted history. The form's P/K/Mg
# fields are the standard N-P2O5-K2O-MgO label guarantees. Dose is per tree,
# per application; annual use = dose × frequency × detected palms.
_ui_fertilizer_entries = (UI_SUBMISSION_DATA.get("fertilizerHistory") or {}).get("fertilizers", [])
_ui_npk_entry = next((item for item in _ui_fertilizer_entries
                      if str(item.get("name", "")).strip().upper() == "NPK"), None)
_ui_fertilizer_frequency = (UI_SUBMISSION_DATA.get("fertilizerHistory") or {}).get("freq")
if _ui_npk_entry is not None:
    _ui_comp = _ui_npk_entry.get("composition") or {}
    if all(key in _ui_comp for key in ("N", "P", "K", "Mg")):
        USER_FERTILIZER_ANALYSIS = "-".join(str(_ui_comp[key]) for key in ("N", "P", "K", "Mg"))
    USER_FERTILIZER_PRODUCT_NAME = _ui_npk_entry.get("name", "NPK")
    BAG_SIZE_KG = float(_ui_npk_entry.get("bagKg", BAG_SIZE_KG))
    if _ui_npk_entry.get("dosePerTree") is not None:
        if _ui_fertilizer_frequency is None:
            raise ValueError("fertilizerHistory.freq is required to annualize dosePerTree.")
        if USER_FARM_AREA_HA <= 0 or BAG_SIZE_KG <= 0:
            raise ValueError("Farm area and fertilizer bag size must be positive.")
        USER_DOSE_PER_TREE_KG_PER_APPLICATION = float(_ui_npk_entry["dosePerTree"])
        USER_FERTILIZER_APPLICATIONS_PER_YEAR = float(_ui_fertilizer_frequency)
        _npk_annual_product_kg_farm = (
            USER_DOSE_PER_TREE_KG_PER_APPLICATION
            * USER_FERTILIZER_APPLICATIONS_PER_YEAR
            * USER_DETECTED_TREE_COUNT
        )
        USER_NPK_PRODUCT_KG_PER_HA_PER_YEAR = _npk_annual_product_kg_farm / USER_FARM_AREA_HA
        USER_BAGS_PER_HA_PER_YEAR = USER_NPK_PRODUCT_KG_PER_HA_PER_YEAR / BAG_SIZE_KG

OUTPUT_DECIMALS = 6  # Consistent decimal places for all displayed mass and rate values.
NUTRIENT_MATCH_TOLERANCE_PERCENT = 5.0  # Differences within ±5% of baseline are labeled close to target.
NPK_GAP_ROUTE_THRESHOLD_PERCENT = 30.0  # Route cutoff only: larger N/P/K gaps suggest reviewing blend quantity; tune for your policy.
ALTERNATIVE_NPK_PRODUCTS = {
    "Reference-style 12-6-22-3": "12-6-22-3",  # Optional candidate blends, in N-P2O5-K2O-MgO order.
    "Reference-style 13-6-27-4": "13-6-27-4",
    "Reference-style 13-8-27-4": "13-8-27-4",
    "Reference-style 16-16-16": "16-16-16",
}

def npk_label_price_per_bag_rp(label_percent):
    """Look up bag price from the canonical N-P2O5-K2O-MgO label."""
    key = "-".join(f"{float(label_percent[name]):g}" for name in ("N", "P2O5", "K2O", "MgO"))
    return PULSE_CHECK_PRODUCT_PRICE_PER_BAG_RP.get(key)

# Age-based NPK product target from the supplied Commonize column.
# A 2 kg/tree/application dose is one base unit. First solve the blend quantity
# that meets the elemental baseline; express it in these base units, then scale it
# by the age multiplier from the supplied Commonize column.
NPK_BASE_DOSE_KG_PER_TREE_PER_APPLICATION = 2.0
ENABLE_NPK_AGE_MULTIPLIER = True  # Set False to use the baseline blend amount without age scaling.
NPK_AGE_COMMONIZE_MULTIPLIERS = {
    (4, 8): 1.25,
    (9, 13): 2.00,
    (14, 20): 1.75,
    (21, None): 1.50,  # 21+ years (the image labels this as >20 years).
}
ENABLE_NPK_AGE_BUDGET_CAP = True  # Constrain age multipliers using the submitted annual fertilizer budget.
MAX_NPK_COST_OVER_BUDGET_PERCENT = 15.0  # Configurable maximum spend above the submitted annual budget.
NPK_AGE_REFERENCE_DATE = _npk_date.today()  # Used with farm planting year/month to select an age band.

# Current project baseline. Values are elemental nutrient rates in kg/ha/year.
BASELINE_ELEMENTAL_KG_HA_YEAR = {
    "N": 44.00,
    "P": 5.00,
    "K": 54.00,
    "Mg": 6.00,
}

# Standalone Mg products, using the same label-basis assumptions as
# oil_palm_zones_vari_fertilizer.py. Enter the guaranteed MgO fraction from
# the product label; the calculation converts MgO to elemental Mg in two steps.
STANDALONE_MG_PRODUCTS = {
    "Kieserite": {"mgo_fraction": 0.255, "bag_size_kg": 50.0},
    "Dolomite": {"mgo_fraction": 0.22, "bag_size_kg": 50.0},
}

# Standalone phosphorus products. SP-36 guarantees P2O5, not elemental P.
STANDALONE_P_PRODUCTS = {
    "SP-36": {"oxide_fraction": 0.36, "bag_size_kg": 50.0},
}

# Standalone potassium products. KCl/MOP guarantees K2O, not elemental K.
# Confirm chloride suitability and actual guaranteed analysis for the field/product.
STANDALONE_K_PRODUCTS = {
    "KCl / MOP": {"oxide_fraction": 0.60, "bag_size_kg": 50.0},
}

# Standalone nitrogen options, using the guaranteed elemental N fraction on the bag.
STANDALONE_N_PRODUCTS = {
    "ZA": {"n_fraction": 0.21, "bag_size_kg": 50.0},
    "Urea": {"n_fraction": 0.46, "bag_size_kg": 50.0},
}

# Atomic masses, used to convert fertilizer oxide label amounts to elemental masses.
ATOMIC_MASS_G_MOL = {"P": 30.973761998, "K": 39.0983, "Mg": 24.305, "O": 15.999}
OXIDE_TO_ELEMENT = {
    "P2O5": {
        "element": "P",
        "element_mass_g_mol": 2 * ATOMIC_MASS_G_MOL["P"],
        "oxide_mass_g_mol": 2 * ATOMIC_MASS_G_MOL["P"] + 5 * ATOMIC_MASS_G_MOL["O"],
    },
    "K2O": {
        "element": "K",
        "element_mass_g_mol": 2 * ATOMIC_MASS_G_MOL["K"],
        "oxide_mass_g_mol": 2 * ATOMIC_MASS_G_MOL["K"] + ATOMIC_MASS_G_MOL["O"],
    },
    "MgO": {
        "element": "Mg",
        "element_mass_g_mol": ATOMIC_MASS_G_MOL["Mg"],
        "oxide_mass_g_mol": ATOMIC_MASS_G_MOL["Mg"] + ATOMIC_MASS_G_MOL["O"],
    },
}
for oxide_data in OXIDE_TO_ELEMENT.values():
    oxide_data["factor"] = oxide_data["element_mass_g_mol"] / oxide_data["oxide_mass_g_mol"]

# ## B. Parse labels and convert to elemental nutrients
#
# These helpers interpret the bag label and calculate elemental N, P, K, and Mg supplied per hectare per year.


# ===== Notebook code cell 48 =====
def parse_analysis(label):
    """Read N-P2O5-K2O[-MgO] percentages from a string such as '12-6-22-3'."""
    try:
        values = [float(value.strip()) for value in label.replace(",", "-").split("-")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("Set USER_FERTILIZER_ANALYSIS as e.g. '15-15-15' or '12-6-22-3'.") from exc
    if len(values) == 3:
        values.append(0.0)
    if len(values) != 4 or any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError("Enter three or four label percentages, each from 0 to 100.")
    return dict(zip(("N", "P2O5", "K2O", "MgO"), values))


def npk_fmt(value):
    """Format all displayed masses/rates to the configured common precision."""
    return f"{value:.{OUTPUT_DECIMALS}f}"


def build_annual_nutrients(label_percent, bags_per_ha_year, bag_size_kg=None):
    """Return elemental supply per hectare/year, optionally using an explicit bag size."""
    effective_bag_size_kg = BAG_SIZE_KG if bag_size_kg is None else float(bag_size_kg)
    if not math.isfinite(effective_bag_size_kg) or effective_bag_size_kg <= 0:
        raise ValueError("bag_size_kg must be greater than zero.")
    product_kg_ha_year = bags_per_ha_year * effective_bag_size_kg
    oxide_or_element_kg_ha_year = {
        nutrient: product_kg_ha_year * percent / 100.0
        for nutrient, percent in label_percent.items()
    }
    return {
        "N": oxide_or_element_kg_ha_year["N"],
        "P": oxide_or_element_kg_ha_year["P2O5"] * OXIDE_TO_ELEMENT["P2O5"]["factor"],
        "K": oxide_or_element_kg_ha_year["K2O"] * OXIDE_TO_ELEMENT["K2O"]["factor"],
        "Mg": oxide_or_element_kg_ha_year["MgO"] * OXIDE_TO_ELEMENT["MgO"]["factor"],
    }, oxide_or_element_kg_ha_year, product_kg_ha_year


def build_recorded_fertilizer_history_supply():
    """Calculate elemental nutrient supply from entered product doses, per ha/year."""
    history = UI_SUBMISSION_DATA.get("fertilizerHistory") or {}
    entries = history.get("fertilizers", [])
    if not entries:
        return None
    frequency = history.get("freq")
    if frequency is None:
        raise ValueError("fertilizerHistory.freq is required to annualize dosePerTree.")
    frequency = float(frequency)
    tree_count = int(USER_DETECTED_TREE_COUNT)
    area_ha = float(USER_FARM_AREA_HA)
    if area_ha <= 0:
        raise ValueError("Farm area must be positive to calculate kg/ha/year.")

    totals = {nutrient: 0.0 for nutrient in ("N", "P", "K", "Mg")}
    breakdown, ignored = [], []
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
        label_basis_kg_ha = {}

        if normalized == "npk":
            comp = entry.get("composition") or {}
            if not all(key in comp for key in ("N", "P", "K", "Mg")):
                ignored.append({"product": name, "reason": "NPK composition is missing"})
                continue
            labels = {"N": float(comp["N"]), "P2O5": float(comp["P"]),
                      "K2O": float(comp["K"]), "MgO": float(comp["Mg"])}
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
    return {"elemental_supply_kg_ha_year": totals, "products": breakdown, "ignored_products": ignored}

# ## C. Calculate standalone N, P, K, and Mg options
#
# For a positive Mg gap, this prints kieserite and dolomite as alternatives. Label MgO is converted to elemental Mg before calculating the product mass and bag count.


# ===== Notebook code cell 50 =====
def print_standalone_n_recommendation(elemental_year):
    """Show separate ZA/urea options for the elemental N shortfall."""
    n_target = BASELINE_ELEMENTAL_KG_HA_YEAR["N"]
    n_from_blend = elemental_year["N"]
    n_gap = max(0.0, n_target - n_from_blend)
    print("\nStandalone N supplement options (per hectare/year):")
    print(f"N gap = max(0, baseline {n_target:.{OUTPUT_DECIMALS}f} − N from recorded fertilizers "
          f"{n_from_blend:.{OUTPUT_DECIMALS}f}) = {n_gap:.{OUTPUT_DECIMALS}f} kg elemental N/ha/year")
    if n_gap <= 0:
        print("No standalone N is calculated from this baseline comparison.")
        return
    headers = ["Product", "Label", "Needed kg/ha/year", "Bag kg", "Exact bags", "Buy full bags"]
    rows = []
    for name, config in STANDALONE_N_PRODUCTS.items():
        fraction = float(config["n_fraction"])
        bag_size = float(config["bag_size_kg"])
        if not 0 < fraction <= 1 or bag_size <= 0:
            raise ValueError(f"Invalid N label fraction or bag size configured for {name}.")
        product_needed = n_gap / fraction
        exact_bags = product_needed / bag_size
        full_bags = math.ceil(exact_bags - 1e-12)
        rows.append([name, f"{fraction * 100:.{OUTPUT_DECIMALS}f}% N",
                     npk_fmt(product_needed), npk_fmt(bag_size),
                     npk_fmt(exact_bags), str(full_bags)])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    print("Calculation: product kg = N gap (kg elemental N) ÷ label fraction (kg N/kg product).")
    print("ZA and urea are alternatives; do not add both rows together.")


def print_standalone_oxide_recommendation(elemental_year, nutrient, oxide_basis, products):
    """Calculate standalone P or K options from oxide-label guarantees."""
    target = BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient]
    supplied = elemental_year[nutrient]
    gap = max(0.0, target - supplied)
    oxide_factor = OXIDE_TO_ELEMENT[oxide_basis]["factor"]
    print(f"\nStandalone {nutrient} supplement options (per hectare/year):")
    print(f"Elemental {nutrient} gap = max(0, baseline {npk_fmt(target)} − recorded fertilizer supply "
          f"{npk_fmt(supplied)}) = {npk_fmt(gap)} kg {nutrient}/ha/year")
    if gap <= 0:
        print(f"No standalone {nutrient} is calculated from this baseline comparison.")
        return

    headers = ["Product", "Label guarantee", f"Elemental {nutrient}/kg product",
               "Needed kg/ha/year", "Bag kg", "Exact bags", "Buy full bags"]
    rows, details = [], []
    for name, config in products.items():
        oxide_fraction = float(config["oxide_fraction"])
        bag_size = float(config["bag_size_kg"])
        if not 0 < oxide_fraction <= 1 or bag_size <= 0:
            raise ValueError(f"Invalid oxide label fraction or bag size configured for {name}.")
        elemental_per_kg = oxide_fraction * oxide_factor
        product_needed = gap / elemental_per_kg
        exact_bags = product_needed / bag_size
        full_bags = math.ceil(exact_bags - 1e-12)
        rows.append([name, f"{oxide_fraction * 100:.{OUTPUT_DECIMALS}f}% {oxide_basis}",
                     f"{npk_fmt(elemental_per_kg)} kg {nutrient}/kg",
                     npk_fmt(product_needed), npk_fmt(bag_size),
                     npk_fmt(exact_bags), str(full_bags)])
        details.append((name, oxide_fraction, bag_size, elemental_per_kg,
                        product_needed, exact_bags, full_bags))

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    oxide = OXIDE_TO_ELEMENT[oxide_basis]
    print(f"\nHow the standalone {nutrient} product amount is calculated:")
    print(f"  1. Elemental shortfall = {npk_fmt(target)} kg baseline − {npk_fmt(supplied)} kg from recorded fertilizers "
          f"= {npk_fmt(gap)} kg {nutrient}/ha/year.")
    print(f"  2. {oxide_basis} → elemental {nutrient} factor = {oxide['element_mass_g_mol']:.9f} ÷ "
          f"{oxide['oxide_mass_g_mol']:.9f} = {oxide_factor:.9f}.")
    for name, oxide_fraction, bag_size, elemental_per_kg, product_needed, exact_bags, full_bags in details:
        oxide_per_bag = bag_size * oxide_fraction
        nutrient_per_bag = oxide_per_bag * oxide_factor
        print(f"  {name}:")
        print(f"    Label: {oxide_fraction * 100:.{OUTPUT_DECIMALS}f}% {oxide_basis} ÷ 100 "
              f"= {npk_fmt(oxide_fraction)} kg {oxide_basis}/kg product.")
        print(f"    One bag: {npk_fmt(bag_size)} kg × {npk_fmt(oxide_fraction)} "
              f"= {npk_fmt(oxide_per_bag)} kg {oxide_basis}; × {oxide_factor:.9f} "
              f"= {npk_fmt(nutrient_per_bag)} kg elemental {nutrient} per bag.")
        print(f"    Elemental {nutrient} per kg product: {npk_fmt(oxide_fraction)} × {oxide_factor:.9f} "
              f"= {npk_fmt(elemental_per_kg)} kg {nutrient}/kg product.")
        print(f"    Product needed: {npk_fmt(gap)} kg {nutrient} ÷ {npk_fmt(elemental_per_kg)} "
              f"kg {nutrient}/kg product = {npk_fmt(product_needed)} kg product/ha/year.")
        print(f"    Bags: {npk_fmt(product_needed)} kg ÷ {npk_fmt(bag_size)} kg/bag "
              f"= {npk_fmt(exact_bags)}; buy {full_bags} full bag(s).")


def print_standalone_p_recommendation(elemental_year):
    print_standalone_oxide_recommendation(elemental_year, "P", "P2O5", STANDALONE_P_PRODUCTS)


def print_standalone_k_recommendation(elemental_year):
    print_standalone_oxide_recommendation(elemental_year, "K", "K2O", STANDALONE_K_PRODUCTS)


def print_standalone_mg_recommendation(elemental_year):
    """Show standalone Mg product options for any shortfall against baseline."""
    mg_target = BASELINE_ELEMENTAL_KG_HA_YEAR["Mg"]
    mg_from_blend = elemental_year["Mg"]
    mg_gap = max(0.0, mg_target - mg_from_blend)
    print("\nStandalone Mg supplement options (per hectare/year):")
    print("These are alternatives; select one Mg source rather than adding the rows together.")
    print(f"Elemental Mg gap = max(0, baseline {npk_fmt(mg_target)} − recorded fertilizer supply {npk_fmt(mg_from_blend)}) "
          f"= {npk_fmt(mg_gap)} kg Mg/ha/year")
    if mg_gap <= 0:
        print("No standalone Mg is calculated from this baseline comparison.")
        return

    headers = ["Product", "Label guarantee", "Elemental Mg/kg product", "Needed kg/ha/year",
               "Bag kg", "Exact bags", "Buy full bags"]
    rows = []
    calculations = []
    mg_factor = OXIDE_TO_ELEMENT["MgO"]["factor"]
    for name, config in STANDALONE_MG_PRODUCTS.items():
        mgo_fraction = float(config["mgo_fraction"])
        bag_size = float(config["bag_size_kg"])
        if not 0 <= mgo_fraction <= 1 or bag_size <= 0:
            raise ValueError(f"Invalid Mg label fraction or bag size configured for {name}.")
        elemental_mg_per_kg_product = mgo_fraction * mg_factor
        product_needed = mg_gap / elemental_mg_per_kg_product if elemental_mg_per_kg_product else math.inf
        exact_bags = product_needed / bag_size
        full_bags = math.ceil(exact_bags - 1e-12)
        rows.append([
            name,
            f"{mgo_fraction * 100:.{OUTPUT_DECIMALS}f}% MgO",
            f"{npk_fmt(elemental_mg_per_kg_product)} kg Mg/kg",
            npk_fmt(product_needed),
            npk_fmt(bag_size),
            npk_fmt(exact_bags),
            str(full_bags),
        ])
        calculations.append((name, mgo_fraction, bag_size, elemental_mg_per_kg_product,
                             product_needed, exact_bags, full_bags))

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    print("\nHow the standalone Mg amounts are calculated:")
    print(f"  1. Mg shortfall = {npk_fmt(mg_target)} kg baseline − {npk_fmt(mg_from_blend)} kg from NPK blend "
          f"= {npk_fmt(mg_gap)} kg elemental Mg/ha/year.")
    print(f"  2. MgO → elemental Mg factor = {OXIDE_TO_ELEMENT['MgO']['element_mass_g_mol']:.9f} ÷ "
          f"{OXIDE_TO_ELEMENT['MgO']['oxide_mass_g_mol']:.9f} = {mg_factor:.9f}.")
    for name, mgo_fraction, bag_size, elemental_per_kg, product_needed, exact_bags, full_bags in calculations:
        print(f"  {name}:")
        print(f"    Label: {mgo_fraction * 100:.{OUTPUT_DECIMALS}f}% MgO ÷ 100 "
              f"= {npk_fmt(mgo_fraction)} kg MgO/kg product.")
        print(f"    Elemental Mg per kg product = {npk_fmt(mgo_fraction)} × {mg_factor:.9f} "
              f"= {npk_fmt(elemental_per_kg)} kg Mg/kg product.")
        print(f"    Product needed = {npk_fmt(mg_gap)} kg Mg ÷ {npk_fmt(elemental_per_kg)} kg Mg/kg product "
              f"= {npk_fmt(product_needed)} kg product/ha/year.")
        print(f"    Exact bags = {npk_fmt(product_needed)} kg ÷ {npk_fmt(bag_size)} kg/bag "
              f"= {npk_fmt(exact_bags)}; buy {full_bags} full bag(s).")

# ## D. Build the comparison report
#
# The report shows per-bag nutrient content, annual supply versus baseline, the calculation steps, standalone N and Mg options, and a field-wide blend decision aid. The decision aid is separate from the zone-specific boost calculations above.


# ===== Notebook code cell 52 =====


def _selected_zone_boost_cost_for_budget():
    """Whole-bag annual cost for the chosen zone Mg product plus Urea, or None if unknown."""
    try:
        bag_rows = product_bag_rows
    except NameError:
        return None
    history_products = {
        str(item.get("name", "")).strip().casefold()
        for item in ((UI_SUBMISSION_DATA.get("fertilizerHistory") or {}).get("fertilizers", []))
    }
    if history_products.intersection({"dolomite", "dolomit"}):
        mg_product = "dolomite_Mg_cheaper"
    elif history_products.intersection({"kieserite", "kieserit"}):
        mg_product = "kieserite_Mg_expensive"
    else:
        mg_product = "dolomite_Mg_cheaper"
    required_products = {mg_product, "urea_N_expensive"}
    totals = {row.get("product"): row for row in bag_rows
              if row.get("zone_id") == "FIELD TOTAL" and row.get("product") in required_products}
    if set(totals) != required_products:
        return None
    costs = [totals[product].get("full_bag_purchase_cost_rp") for product in required_products]
    if any(cost is None for cost in costs):
        return None
    return sum(float(cost) for cost in costs)


def _budget_cap_age_multiplier(configured_multiplier, baseline_exact_bags_per_ha, price_per_bag_rp):
    """Cap age scaling by annual whole-bag cost while preserving the 1× baseline."""
    configured_multiplier = float(configured_multiplier)
    result = {
        "configured_multiplier": configured_multiplier,
        "applied_multiplier": configured_multiplier,
        "budget_cap_enabled": bool(ENABLE_NPK_AGE_BUDGET_CAP),
        "budget_rp": UI_SUBMISSION_DATA.get("fertilizerBudgetRp"),
        "maximum_over_budget_percent": float(MAX_NPK_COST_OVER_BUDGET_PERCENT),
        "allowed_spend_rp": None,
        "fixed_zone_boost_cost_rp": None,
        "estimated_npk_price_per_bag_rp": price_per_bag_rp,
        "calculation_steps": [],
        "status": "budget cap disabled" if not ENABLE_NPK_AGE_BUDGET_CAP else "budget not entered",
    }
    steps = result["calculation_steps"]
    if not ENABLE_NPK_AGE_BUDGET_CAP:
        steps.append("Budget cap is disabled; keep the configured Commonize multiplier.")
        return result
    budget = UI_SUBMISSION_DATA.get("fertilizerBudgetRp")
    if budget is None:
        steps.append("fertilizerBudgetRp is empty; no budget cap is applied, so keep the configured Commonize multiplier.")
        return result
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        result["status"] = "invalid budget input"
        steps.append("Budget input is not a valid number; no budget cap is applied.")
        return result
    if not math.isfinite(budget) or budget < 0:
        result["status"] = "invalid budget input"
        steps.append("Budget must be a finite nonnegative Rupiah amount; no budget cap is applied.")
        return result
    result["budget_rp"] = budget
    margin = float(MAX_NPK_COST_OVER_BUDGET_PERCENT)
    margin_fraction = margin / 100.0
    budget_factor = 1.0 + margin_fraction
    allowed_spend = budget * budget_factor
    result["allowed_spend_rp"] = allowed_spend
    steps.append(
        f"Margin factor = 1 + ({margin:.2f}% ÷ 100) = 1 + {margin_fraction:.4f} = {budget_factor:.4f}; "
        "1 represents the original 100% budget, and the added fraction is the allowed margin."
    )
    steps.append(
        f"Allowed annual spend = Rp{budget:,.0f} × {budget_factor:.4f} = Rp{allowed_spend:,.0f}."
    )
    zone_cost = _selected_zone_boost_cost_for_budget()
    result["fixed_zone_boost_cost_rp"] = zone_cost
    if zone_cost is None:
        result["status"] = "cannot cap: selected zone Mg/Urea cost is unavailable"
        steps.append("Zone Mg/Urea whole-bag cost is unavailable, so the total budget cannot be split; keep configured multiplier.")
        return result
    steps.append(f"Reserve selected zone Mg + Urea whole-bag cost = Rp{zone_cost:,.0f}.")
    if price_per_bag_rp is None or float(price_per_bag_rp) <= 0:
        result["status"] = "cannot cap: blend bag price is not configured"
        steps.append("This NPK blend has no configured bag price, so its affordable bag count cannot be calculated.")
        return result
    steps.append(f"NPK budget remaining = Rp{allowed_spend:,.0f} − Rp{zone_cost:,.0f} = Rp{allowed_spend - zone_cost:,.0f}.")
    try:
        farm_area = float(USER_FARM_AREA_HA)
    except (TypeError, ValueError, NameError):
        result["status"] = "cannot cap: farm area is unavailable"
        steps.append("Farm area is unavailable, so baseline bags for the whole farm cannot be calculated.")
        return result
    baseline_farm_bags = float(baseline_exact_bags_per_ha) * farm_area
    result["baseline_exact_bags_for_farm_year"] = baseline_farm_bags
    if baseline_farm_bags <= 0:
        result["status"] = "cannot cap: baseline blend quantity is unavailable"
        steps.append("Baseline exact bag quantity is unavailable; keep configured multiplier.")
        return result
    baseline_full_bags = math.ceil(baseline_farm_bags - 1e-12)
    baseline_cost = baseline_full_bags * float(price_per_bag_rp) + zone_cost
    result["baseline_full_bags_for_farm_year"] = baseline_full_bags
    result["baseline_combined_whole_bag_cost_rp"] = baseline_cost
    steps.append(
        f"Baseline blend = {baseline_exact_bags_per_ha:.6f} bags/ha × {farm_area:.6f} ha = "
        f"{baseline_farm_bags:.6f} exact farm bags; buy ceil(...) = {baseline_full_bags} bag(s)."
    )
    max_npk_spend = allowed_spend - zone_cost
    if max_npk_spend < 0:
        result["applied_multiplier"] = 1.0
        result["status"] = "baseline and zone products exceed budget cap; baseline protected"
        steps.append(f"Zone products alone exceed the cap (remaining NPK budget = Rp{max_npk_spend:,.0f}); retain 1.00× baseline.")
        steps.append(f"Baseline plus zone products cost Rp{baseline_cost:,.0f}; the 1.00× baseline is protected even though it exceeds the cap.")
        return result
    affordable_full_bags = math.floor((max_npk_spend / float(price_per_bag_rp)) + 1e-12)
    minimum_baseline_full_bags = math.ceil(baseline_farm_bags - 1e-12)
    result["maximum_affordable_npk_full_bags"] = affordable_full_bags
    steps.append(
        f"Affordable NPK bags = floor(Rp{max_npk_spend:,.0f} ÷ Rp{float(price_per_bag_rp):,.0f}/bag) "
        f"= {affordable_full_bags} full bag(s)."
    )
    if affordable_full_bags < minimum_baseline_full_bags:
        result["applied_multiplier"] = 1.0
        result["status"] = "baseline blend alone exceeds budget cap; baseline protected"
        steps.append(
            f"The baseline needs {minimum_baseline_full_bags} full bag(s), above the {affordable_full_bags} affordable; "
            "keep 1.00× because the nutrient baseline cannot be reduced."
        )
        steps.append(f"Baseline plus zone products cost Rp{baseline_cost:,.0f}, exceeding the allowed Rp{allowed_spend:,.0f}.")
        return result
    extra_affordable_bags = max(0.0, affordable_full_bags - baseline_farm_bags)
    affordable_increase_fraction = extra_affordable_bags / baseline_farm_bags
    max_affordable_multiplier = 1.0 + affordable_increase_fraction
    applied = min(configured_multiplier, max(1.0, max_affordable_multiplier))
    result["maximum_multiplier_allowed_by_budget"] = max_affordable_multiplier
    result["applied_multiplier"] = applied
    result["estimated_npk_full_bags_for_farm"] = math.ceil(baseline_farm_bags * applied - 1e-12)
    result["estimated_combined_whole_bag_cost_rp"] = (
        result["estimated_npk_full_bags_for_farm"] * float(price_per_bag_rp) + zone_cost
    )
    steps.append(
        f"Extra bag capacity above baseline = {affordable_full_bags} affordable bags − "
        f"{baseline_farm_bags:.6f} exact baseline bags = {extra_affordable_bags:.6f} bag(s)."
    )
    steps.append(
        f"Affordable increase fraction = {extra_affordable_bags:.6f} extra bags ÷ "
        f"{baseline_farm_bags:.6f} baseline bags = {affordable_increase_fraction:.6f} "
        f"({affordable_increase_fraction * 100:.2f}% above baseline)."
    )
    steps.append(
        f"Maximum affordable multiplier = 1.00 baseline + {affordable_increase_fraction:.6f} increase "
        f"= 1 + ({affordable_full_bags} ÷ {baseline_farm_bags:.6f} − 1) "
        f"= {max_affordable_multiplier:.6f}×. At this multiplier, baseline {baseline_farm_bags:.6f} × "
        f"{max_affordable_multiplier:.6f} = {baseline_farm_bags * max_affordable_multiplier:.6f} bags, "
        f"which equals the {affordable_full_bags} affordable whole-bag limit."
    )
    steps.append(
        f"Applied multiplier = min(Commonize {configured_multiplier:.6f}×, max(1.00×, {max_affordable_multiplier:.6f}×)) "
        f"= {applied:.6f}×."
    )
    steps.append(
        f"At {applied:.6f}×: ceil({baseline_farm_bags:.6f} × {applied:.6f}) = "
        f"{result['estimated_npk_full_bags_for_farm']} NPK bag(s); combined whole-bag cost = "
        f"Rp{result['estimated_combined_whole_bag_cost_rp']:,.0f}."
    )
    result["status"] = (
        "age multiplier reduced to fit budget cap"
        if applied < configured_multiplier - 1e-12
        else "configured age multiplier fits budget cap"
    )
    return result



def build_age_based_npk_target(label_percent):
    """Meet the blend nutrient baseline first, then scale its per-tree dose by age."""
    planting_year = UI_FARM_INPUT.get("year")
    planting_month = UI_FARM_INPUT.get("plantingMonth")
    reference_date = NPK_AGE_REFERENCE_DATE
    age_years, age_band, configured_multiplier = None, None, None
    if ENABLE_NPK_AGE_MULTIPLIER:
        if planting_year is None:
            return {"available": False, "reason": "farm planting year is needed when age multiplier mode is enabled"}
        age_years = reference_date.year - int(planting_year)
        if planting_month is not None and reference_date.month < int(planting_month):
            age_years -= 1
        if age_years < 0:
            return {"available": False, "reason": "planting date is later than the age reference date"}
        for (age_min, age_max), age_factor in NPK_AGE_COMMONIZE_MULTIPLIERS.items():
            if age_years >= age_min and (age_max is None or age_years <= age_max):
                configured_multiplier = float(age_factor)
                age_band = f"{age_min}+ years" if age_max is None else f"{age_min}-{age_max} years"
                break
        if configured_multiplier is None:
            return {"available": False,
                    "reason": f"no Commonize multiplier is specified for {age_years} years of age",
                    "tree_age_years": age_years, "age_reference_date": reference_date.isoformat()}
        multiplier = configured_multiplier
    else:
        # With age mode off, keep the baseline-satisfying blend at 1×.
        multiplier = 1.0
        age_band = "age multiplier disabled"

    frequency = (UI_SUBMISSION_DATA.get("fertilizerHistory") or {}).get("freq")
    if frequency is None:
        return {"available": False, "reason": "fertilizerHistory.freq is missing",
                "tree_age_years": age_years, "age_band": age_band,
                "commonize_multiplier": multiplier}
    frequency = float(frequency)
    tree_count, area_ha = int(USER_DETECTED_TREE_COUNT), float(USER_FARM_AREA_HA)
    if frequency <= 0 or tree_count <= 0 or area_ha <= 0:
        return {"available": False, "reason": "frequency, detected tree count, and farm area must be positive"}

    # First calculate the blend quantity that satisfies every elemental baseline.
    one_bag_supply, _, _ = build_annual_nutrients(label_percent, 1.0)
    bags_required_each = {
        n: float(BASELINE_ELEMENTAL_KG_HA_YEAR[n]) / one_bag_supply[n]
        if one_bag_supply[n] > 0 else math.inf
        for n in ("N", "P", "K")
    }
    if not all(math.isfinite(v) for v in bags_required_each.values()):
        return {"available": False, "reason": "selected blend cannot supply all baseline N/P/K nutrients"}
    baseline_bags_ha = max(bags_required_each.values())
    limiting_nutrient = max(bags_required_each, key=bags_required_each.get)
    configured_age_multiplier = multiplier
    budget_cap = _budget_cap_age_multiplier(
        configured_age_multiplier, baseline_bags_ha, npk_label_price_per_bag_rp(label_percent)
    )
    multiplier = budget_cap["applied_multiplier"]
    baseline_kg_ha = baseline_bags_ha * BAG_SIZE_KG
    baseline_kg_farm = baseline_kg_ha * area_ha
    baseline_kg_tree_year = baseline_kg_farm / tree_count

    # Convert baseline kg/tree/year to 2 kg base-dose units per application;
    # then multiply those units by the tree-age Commonize factor.
    base_units_per_tree_application = baseline_kg_tree_year / (NPK_BASE_DOSE_KG_PER_TREE_PER_APPLICATION * frequency)
    adjusted_units_per_tree_application = base_units_per_tree_application * multiplier
    target_per_application = adjusted_units_per_tree_application * NPK_BASE_DOSE_KG_PER_TREE_PER_APPLICATION
    target_per_tree_year = target_per_application * frequency
    target_farm_year = target_per_tree_year * tree_count
    target_ha_year = target_farm_year / area_ha

    npk_entry = next((item for item in _ui_fertilizer_entries
                      if str(item.get("name", "")).strip().upper() == "NPK"), None)
    current_dose = float(npk_entry["dosePerTree"]) if npk_entry and npk_entry.get("dosePerTree") is not None else 0.0
    current_per_tree_year = current_dose * frequency
    return {
        "available": True,
        "tree_age_years": age_years,
        "age_reference_date": reference_date.isoformat() if ENABLE_NPK_AGE_MULTIPLIER else None,
        "age_band": age_band,
        "age_multiplier_enabled": bool(ENABLE_NPK_AGE_MULTIPLIER),
        "configured_age_multiplier": configured_multiplier,
        "commonize_multiplier": multiplier,
        "budget_adjusted_age_multiplier": multiplier,
        "budget_cap": budget_cap,
        "limiting_nutrient_for_baseline_blend": limiting_nutrient,
        "baseline_blend_bags_ha_year": baseline_bags_ha,
        "baseline_blend_product_kg_ha_year": baseline_kg_ha,
        "baseline_blend_product_kg_farm_year": baseline_kg_farm,
        "baseline_blend_product_kg_tree_year": baseline_kg_tree_year,
        "base_dose_unit_kg_per_tree_per_application": float(NPK_BASE_DOSE_KG_PER_TREE_PER_APPLICATION),
        "fertilizer_applications_per_year": frequency,
        "base_units_per_tree_application_before_age_multiplier": base_units_per_tree_application,
        "age_adjusted_base_units_per_tree_application": adjusted_units_per_tree_application,
        "target_dose_kg_per_tree_per_application": target_per_application,
        "target_kg_per_tree_year": target_per_tree_year,
        "current_npk_dose_kg_per_tree_per_application": current_dose,
        "current_npk_kg_per_tree_year": current_per_tree_year,
        "difference_current_minus_target_kg_per_tree_year": current_per_tree_year - target_per_tree_year,
        "current_as_percent_of_target": current_per_tree_year / target_per_tree_year * 100.0 if target_per_tree_year else None,
        "target_npk_product_kg_farm_year": target_farm_year,
        "detected_tree_count": tree_count,
        "farm_area_ha": area_ha,
        "target_npk_product_kg_ha_year": target_ha_year,
        "formula": "baseline blend bags/ha = max(elemental N/P/K baseline ÷ elemental N/P/K per bag); baseline blend kg/tree/year = baseline blend kg/ha/year × farm ha ÷ detected trees; base-dose units/tree/application = baseline blend kg/tree/year ÷ (2 kg × applications/year); applied multiplier = Commonize age factor when enabled, otherwise 1.00; adjusted dose = base-dose units × applied multiplier × 2 kg",
    }


def build_field_npk_json_record(label_percent):
    """Return machine-readable field-wide blend comparison and standalone product options."""
    npk_elemental_year, label_basis_year, product_kg = build_annual_nutrients(
        label_percent, USER_BAGS_PER_HA_PER_YEAR
    )
    history_supply = build_recorded_fertilizer_history_supply()
    age_based_npk_target = build_age_based_npk_target(label_percent)
    elemental_year = (history_supply["elemental_supply_kg_ha_year"]
                      if history_supply is not None else npk_elemental_year)
    comparisons = {}
    gaps = {}
    for nutrient, target in BASELINE_ELEMENTAL_KG_HA_YEAR.items():
        supplied = float(elemental_year[nutrient])
        difference = supplied - float(target)
        gap = max(0.0, float(target) - supplied)
        gaps[nutrient] = gap
        comparisons[nutrient] = {
            "baseline_elemental_kg_ha_year": float(target),
            "recorded_fertilizer_supply_elemental_kg_ha_year": supplied,
            "npk_blend_only_elemental_kg_ha_year": float(npk_elemental_year[nutrient]),
            "difference_elemental_kg_ha_year": difference,
            "shortfall_elemental_kg_ha_year": gap,
            "shortfall_percent_of_baseline": gap / float(target) * 100.0 if target else 0.0,
            "surplus_percent_of_baseline": max(0.0, difference) / float(target) * 100.0 if target else 0.0,
        }

    product_groups = {
        "N": [(name, "N", float(cfg["n_fraction"]), 1.0, float(cfg["bag_size_kg"]))
              for name, cfg in STANDALONE_N_PRODUCTS.items()],
        "P": [(name, "P2O5", float(cfg["oxide_fraction"]), OXIDE_TO_ELEMENT["P2O5"]["factor"], float(cfg["bag_size_kg"]))
              for name, cfg in STANDALONE_P_PRODUCTS.items()],
        "K": [(name, "K2O", float(cfg["oxide_fraction"]), OXIDE_TO_ELEMENT["K2O"]["factor"], float(cfg["bag_size_kg"]))
              for name, cfg in STANDALONE_K_PRODUCTS.items()],
        "Mg": [(name, "MgO", float(cfg["mgo_fraction"]), OXIDE_TO_ELEMENT["MgO"]["factor"], float(cfg["bag_size_kg"]))
               for name, cfg in STANDALONE_MG_PRODUCTS.items()],
    }
    standalone = {}
    for nutrient, products in product_groups.items():
        gap = gaps[nutrient]
        options = []
        for name, label_basis, label_fraction, oxide_factor, bag_size in products:
            elemental_fraction = label_fraction * oxide_factor
            product_needed = gap / elemental_fraction if gap > 0 and elemental_fraction > 0 else 0.0
            exact_bags = product_needed / bag_size if bag_size > 0 else 0.0
            _standalone_price = PULSE_CHECK_PRODUCT_PRICE_PER_BAG_RP.get(name)
            options.append({
                "product": name,
                "price_per_bag_rp": _standalone_price,
                "estimated_cost_exact_bags_per_ha_rp": exact_bags * _standalone_price if _standalone_price is not None else None,
                "estimated_cost_full_bags_per_ha_rp": (math.ceil(exact_bags - 1e-12) * _standalone_price
                    if _standalone_price is not None and exact_bags > 0 else 0 if _standalone_price is not None else None),
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
        (gaps[n] / float(BASELINE_ELEMENTAL_KG_HA_YEAR[n]) * 100.0 if BASELINE_ELEMENTAL_KG_HA_YEAR[n] else 0.0)
        for n in ("N", "P", "K")
    )
    all_met = all(gaps[n] == 0 for n in BASELINE_ELEMENTAL_KG_HA_YEAR)
    if all_met:
        route = "no_field_wide_top_up_indicated"
    elif largest_npk_gap <= NPK_GAP_ROUTE_THRESHOLD_PERCENT:
        route = "standalone_top_up_route_with_zone_boosts_kept_separate"
    else:
        route = "review_blend_quantity_or_alternative_npk_before_top_up"

    # Costed blend choices use the supplied bag prices and whole bags to buy.
    blend_price_estimates = []
    current_blend_price = npk_label_price_per_bag_rp(label_percent)
    blend_choices = [("Current blend", label_percent, USER_BAGS_PER_HA_PER_YEAR)]
    for choice_name, choice_label in ALTERNATIVE_NPK_PRODUCTS.items():
        choice_analysis = parse_analysis(choice_label)
        if choice_analysis != label_percent:
            blend_choices.append((choice_name, choice_analysis, 0.0))
    for choice_name, choice_analysis, used_bags in blend_choices:
        one_bag, _, _ = build_annual_nutrients(choice_analysis, 1.0)
        if any(one_bag[nutrient] <= 0 for nutrient in ("N", "P", "K")):
            continue
        needed_by_nutrient = {
            nutrient: float(BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient]) / one_bag[nutrient]
            for nutrient in ("N", "P", "K")
        }
        exact_bags = max(needed_by_nutrient.values())
        full_bags = math.ceil(exact_bags - 1e-12)
        additional_bags = max(0, math.ceil(exact_bags - float(used_bags) - 1e-12))
        price = npk_label_price_per_bag_rp(choice_analysis)
        choice_age_target = build_age_based_npk_target(choice_analysis)
        age_multiplier_value = choice_age_target.get("commonize_multiplier") if choice_age_target.get("available") else None
        age_exact_bags = exact_bags * age_multiplier_value if age_multiplier_value is not None else None
        age_full_bags = math.ceil(age_exact_bags - 1e-12) if age_exact_bags is not None else None
        blend_price_estimates.append({
            "blend_option": choice_name,
            "label_percent_n_p2o5_k2o_mgo": {key: float(value) for key, value in choice_analysis.items()},
            "price_per_bag_rp": price,
            "baseline_exact_bags_per_ha_year": exact_bags,
            "baseline_full_bags_per_ha_year": full_bags,
            "baseline_full_bag_cost_rp_per_ha_year": full_bags * price if price is not None else None,
            "additional_bags_to_baseline_per_ha_year": additional_bags,
            "additional_bag_cost_rp_per_ha_year": additional_bags * price if price is not None else None,
            "limiting_nutrient": max(needed_by_nutrient, key=needed_by_nutrient.get),
            "age_multiplier": age_multiplier_value,
            "configured_age_multiplier": choice_age_target.get("configured_age_multiplier"),
            "age_budget_cap": choice_age_target.get("budget_cap"),
            "age_adjusted_exact_bags_per_ha_year": age_exact_bags,
            "age_adjusted_full_bags_per_ha_year": age_full_bags,
            "age_adjusted_full_bag_cost_rp_per_ha_year": age_full_bags * price if price is not None and age_full_bags is not None else None,
        })

    # Reproduce the final overview's closest-to-budget comparison in the JSON.
    _budget_value = UI_SUBMISSION_DATA.get("fertilizerBudgetRp")
    try:
        _budget_value = float(_budget_value) if _budget_value is not None else None
        if _budget_value is not None and (not math.isfinite(_budget_value) or _budget_value < 0):
            _budget_value = None
    except (TypeError, ValueError):
        _budget_value = None
    _zone_cost_for_ranking = _selected_zone_boost_cost_for_budget()
    _budget_rank_rows = []
    for _blend in blend_price_estimates:
        _applied_mult = _blend.get("age_multiplier")
        _unit_bags_ha = _blend.get("baseline_exact_bags_per_ha_year")
        _bag_price = _blend.get("price_per_bag_rp")
        _farm_area = float(USER_FARM_AREA_HA)
        _exact_farm_bags = (
            float(_unit_bags_ha) * float(_applied_mult) * _farm_area
            if _unit_bags_ha is not None and _applied_mult is not None else None
        )
        _full_farm_bags = math.ceil(_exact_farm_bags - 1e-12) if _exact_farm_bags is not None else None
        _npk_farm_cost = (
            _full_farm_bags * float(_bag_price)
            if _full_farm_bags is not None and _bag_price is not None else None
        )
        _combined_farm_cost = (
            _npk_farm_cost + float(_zone_cost_for_ranking)
            if _npk_farm_cost is not None and _zone_cost_for_ranking is not None else None
        )
        _difference = _combined_farm_cost - _budget_value if _combined_farm_cost is not None and _budget_value is not None else None
        _budget_rank_rows.append({
            "blend_option": _blend["blend_option"],
            "composition_label_percent_n_p2o5_k2o_mgo": _blend["label_percent_n_p2o5_k2o_mgo"],
            "age_multiplier_applied": _applied_mult,
            "baseline_exact_bags_per_ha_year": _unit_bags_ha,
            "age_adjusted_exact_bags_farm_year": _exact_farm_bags,
            "age_adjusted_full_bags_farm_year": _full_farm_bags,
            "npk_price_per_bag_rp": _bag_price,
            "npk_full_bag_cost_farm_year_rp": _npk_farm_cost,
            "selected_zone_mg_urea_whole_bag_cost_farm_year_rp": _zone_cost_for_ranking,
            "combined_estimated_cost_farm_year_rp": _combined_farm_cost,
            "budget_difference_rp": _difference,
            "absolute_budget_difference_rp": abs(_difference) if _difference is not None else None,
        })
    _rankable_budget_rows = [row for row in _budget_rank_rows if row["absolute_budget_difference_rp"] is not None]
    _rankable_budget_rows.sort(key=lambda row: row["absolute_budget_difference_rp"])
    for _rank, _row in enumerate(_rankable_budget_rows, start=1):
        _row["budget_fit_rank"] = _rank
        _row["closest_to_budget"] = _rank == 1
    for _row in _budget_rank_rows:
        _row.setdefault("budget_fit_rank", None)
        _row.setdefault("closest_to_budget", False)
    _budget_status = (
        "ranked_by_absolute_difference_to_budget"
        if _budget_value is not None and _rankable_budget_rows
        else "budget_not_entered" if _budget_value is None
        else "cannot_rank_cost_unavailable"
    )
    _budget_comparison = {
        "annual_budget_rp": _budget_value,
        "budget_cap_enabled": bool(ENABLE_NPK_AGE_BUDGET_CAP),
        "comparison_basis": "whole-farm annual whole-bag NPK cost plus the selected zone Mg and Urea whole-bag costs; closest means the smallest absolute difference from the entered budget",
        "selected_zone_mg_urea_whole_bag_cost_farm_year_rp": _zone_cost_for_ranking,
        "status": _budget_status,
        "closest_blend_option": _rankable_budget_rows[0]["blend_option"] if _rankable_budget_rows else None,
        "closest_combined_estimated_cost_farm_year_rp": _rankable_budget_rows[0]["combined_estimated_cost_farm_year_rp"] if _rankable_budget_rows else None,
        "closest_absolute_budget_difference_rp": _rankable_budget_rows[0]["absolute_budget_difference_rp"] if _rankable_budget_rows else None,
        "ranked_options": _budget_rank_rows,
    }

    return {
        "scope": "all recognized recorded fertilizer-history products compared with the configured elemental baseline; dose-based annual use",
        "user_blend": {
            "label_order": "N-P2O5-K2O-MgO",
            "label_percent": {key: float(value) for key, value in label_percent.items()},
            "bags_per_ha_year": float(USER_BAGS_PER_HA_PER_YEAR),
            "bag_size_kg": float(BAG_SIZE_KG),
            "price_per_bag_rp": npk_label_price_per_bag_rp(label_percent),
            "estimated_cost_exact_bags_per_ha_rp": (
                float(USER_BAGS_PER_HA_PER_YEAR) * npk_label_price_per_bag_rp(label_percent)
                if npk_label_price_per_bag_rp(label_percent) is not None else None
            ),
            "estimated_cost_full_bags_per_ha_rp": (
                math.ceil(float(USER_BAGS_PER_HA_PER_YEAR) - 1e-12) * npk_label_price_per_bag_rp(label_percent)
                if npk_label_price_per_bag_rp(label_percent) is not None else None
            ),
            "estimated_cost_exact_bags_per_ha_rp": (
                float(USER_BAGS_PER_HA_PER_YEAR) * npk_label_price_per_bag_rp(label_percent)
                if npk_label_price_per_bag_rp(label_percent) is not None else None
            ),
            "estimated_cost_full_bags_per_ha_rp": (
                math.ceil(float(USER_BAGS_PER_HA_PER_YEAR) - 1e-12) * npk_label_price_per_bag_rp(label_percent)
                if npk_label_price_per_bag_rp(label_percent) is not None else None
            ),
            "product_kg_ha_year": float(product_kg),
            "label_basis_kg_ha_year": {key: float(value) for key, value in label_basis_year.items()},
            "elemental_supply_kg_ha_year": {key: float(value) for key, value in npk_elemental_year.items()},
            "recorded_history_total_elemental_supply_kg_ha_year": {key: float(value) for key, value in elemental_year.items()},
            "dose_per_tree_kg_per_application": USER_DOSE_PER_TREE_KG_PER_APPLICATION,
            "fertilizer_applications_per_year": USER_FERTILIZER_APPLICATIONS_PER_YEAR,
            "detected_tree_count_used": USER_DETECTED_TREE_COUNT,
            "farm_area_ha_used": USER_FARM_AREA_HA,
            "npk_product_kg_farm_per_year": (
                USER_DOSE_PER_TREE_KG_PER_APPLICATION * USER_FERTILIZER_APPLICATIONS_PER_YEAR * USER_DETECTED_TREE_COUNT
                if USER_DOSE_PER_TREE_KG_PER_APPLICATION is not None else None
            ),
        },
        "recorded_fertilizer_history_supply": history_supply,
        "age_based_npk_product_target": age_based_npk_target,
        "baseline_elemental_kg_ha_year": {key: float(value) for key, value in BASELINE_ELEMENTAL_KG_HA_YEAR.items()},
        "nutrient_comparison": comparisons,
        "standalone_top_up_options": standalone,
        "blend_purchase_price_estimates": blend_price_estimates,
        "closest_to_budget_ranking": _budget_comparison,
        "routing": {
            "largest_npk_shortfall_percent": largest_npk_gap,
            "small_gap_threshold_percent": float(NPK_GAP_ROUTE_THRESHOLD_PERCENT),
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


def build_zone_boost_json_records(zone_records):
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
            "overall_field_confidence": rec.get("overall_field_confidence"),
            "fertilizer_boosts": rec.get("fertilizers", {}),
        })
    return records


def print_npk_comparison(label_percent):
    if not math.isfinite(USER_BAGS_PER_HA_PER_YEAR) or USER_BAGS_PER_HA_PER_YEAR < 0:
        raise ValueError("USER_BAGS_PER_HA_PER_YEAR must be zero or a positive number.")
    if not math.isfinite(BAG_SIZE_KG) or BAG_SIZE_KG <= 0:
        raise ValueError("BAG_SIZE_KG must be greater than zero.")
    if not math.isfinite(NPK_GAP_ROUTE_THRESHOLD_PERCENT) or not 0 <= NPK_GAP_ROUTE_THRESHOLD_PERCENT <= 100:
        raise ValueError("NPK_GAP_ROUTE_THRESHOLD_PERCENT must be between 0 and 100.")
    if not math.isfinite(NUTRIENT_MATCH_TOLERANCE_PERCENT) or not 0 <= NUTRIENT_MATCH_TOLERANCE_PERCENT <= 100:
        raise ValueError("NUTRIENT_MATCH_TOLERANCE_PERCENT must be between 0 and 100.")

    npk_elemental_year, label_basis_year, npk_product_kg_ha_year = build_annual_nutrients(
        label_percent, USER_BAGS_PER_HA_PER_YEAR
    )
    history_supply = build_recorded_fertilizer_history_supply()
    elemental_year = (history_supply["elemental_supply_kg_ha_year"]
                      if history_supply is not None else npk_elemental_year)
    print("Annual recorded-fertilizer nutrient comparison with the reference baseline")
    print(f"User product label (N-P2O5-K2O-MgO): {label_percent['N']:g}-"
          f"{label_percent['P2O5']:g}-{label_percent['K2O']:g}-{label_percent['MgO']:g}")
    if USER_DOSE_PER_TREE_KG_PER_APPLICATION is not None:
        _npk_farm_kg = (USER_DOSE_PER_TREE_KG_PER_APPLICATION
                        * USER_FERTILIZER_APPLICATIONS_PER_YEAR * USER_DETECTED_TREE_COUNT)
        print("Dose-based annual use (all entered products):")
        for _product_row in history_supply["products"]:
            print(f"  {_product_row['product']}: {_product_row['dose_kg_per_tree_per_application']:g} kg/tree/application × "
                  f"{_product_row['applications_per_year']:g} applications/year × "
                  f"{_product_row['detected_tree_count']} detected trees = "
                  f"{npk_fmt(_product_row['annual_product_kg_per_farm'])} kg product/farm/year; ÷ "
                  f"{npk_fmt(_product_row['farm_area_ha'])} ha = "
                  f"{npk_fmt(_product_row['annual_product_kg_ha'])} kg product/ha/year")
        for _ignored in history_supply["ignored_products"]:
            print(f"  {_ignored['product']} not included in nutrient totals: {_ignored['reason']}.")
        print(f"  NPK equivalent: {npk_fmt(USER_BAGS_PER_HA_PER_YEAR)} bags/ha/year × "
              f"{npk_fmt(BAG_SIZE_KG)} kg/bag = {npk_fmt(npk_product_kg_ha_year)} kg NPK/ha/year")
    else:
        print(f"Fallback NPK application: {npk_fmt(USER_BAGS_PER_HA_PER_YEAR)} bags/ha/year × "
              f"{npk_fmt(BAG_SIZE_KG)} kg/bag = {npk_fmt(npk_product_kg_ha_year)} kg NPK/ha/year")
    print("Baseline values are interpreted as elemental nutrient kg/ha/year, as labeled in the image.")
    print(f"Comparison tolerance: within ±{NUTRIENT_MATCH_TOLERANCE_PERCENT:.2f}% of a baseline is labeled close to target; "
          "this does not change calculated nutrient amounts.")

    age_target = build_age_based_npk_target(label_percent)
    print("\nAge-adjusted NPK target (baseline blend first, then age multiplier):")
    if age_target.get("available"):
        if age_target.get("age_multiplier_enabled"):
            print(f"  Age {age_target['tree_age_years']} years ({age_target['age_band']}); age multiplier enabled "
                  f"({age_target['commonize_multiplier']:.2f}×).")
        else:
            print("  Age multiplier mode disabled; applying 1.00× to the baseline blend.")
        _budget_info = age_target.get("budget_cap", {})
        if _budget_info.get("budget_cap_enabled") and _budget_info.get("budget_rp") is not None:
            print(f"  Budget cap = Rp{_budget_info.get('allowed_spend_rp'):,.0f} "
                  f"(budget + {MAX_NPK_COST_OVER_BUDGET_PERCENT:.1f}%); "
                  f"multiplier status: {_budget_info.get('status')}.")
        _configured_multiplier_display = age_target.get("configured_age_multiplier")
        if _configured_multiplier_display is None:
            _configured_multiplier_display = 1.0
        print(f"  Configured Commonize multiplier = {_configured_multiplier_display:.2f}×; "
              f"applied multiplier = {age_target['commonize_multiplier']:.2f}×.")
        print(f"  Baseline blend is {npk_fmt(age_target['baseline_blend_bags_ha_year'])} exact bags/ha/year "
              f"(limited by {age_target['limiting_nutrient_for_baseline_blend']}).")
        print(f"  Adjusted blend = {npk_fmt(age_target['baseline_blend_bags_ha_year'])} bags × "
              f"{age_target['commonize_multiplier']:.2f} = "
              f"{npk_fmt(age_target['baseline_blend_bags_ha_year'] * age_target['commonize_multiplier'])} exact bags/ha/year; "
              f"{npk_fmt(age_target['target_kg_per_tree_year'])} kg/tree/year for this farm's detected tree count.")
        print(f"  Current NPK input = {npk_fmt(age_target['current_npk_kg_per_tree_year'])} kg/tree/year; "
              f"{age_target['current_as_percent_of_target']:.2f}% of age-adjusted target.")
    else:
        print(f"  Not calculated: {age_target.get('reason')}")

    print("\nNPK label nutrient content (per bag):")
    print("This table converts the NPK label into elemental nutrient mass in one configured bag; annual totals below include all recognized history products.")
    per_bag_rows = []
    nutrient_label_basis = {"N": "N", "P": "P2O5", "K": "K2O", "Mg": "MgO"}
    for nutrient, label_basis in nutrient_label_basis.items():
        label_pct = label_percent[label_basis]
        label_fraction = label_pct / 100.0
        amount_label_basis = BAG_SIZE_KG * label_fraction
        factor = 1.0 if nutrient == "N" else OXIDE_TO_ELEMENT[label_basis]["factor"]
        elemental_per_bag = amount_label_basis * factor
        if nutrient == "N":
            calculation = f"{npk_fmt(BAG_SIZE_KG)} × ({label_pct:.6f}% ÷ 100) = {npk_fmt(elemental_per_bag)} kg N"
        else:
            calculation = (f"{npk_fmt(BAG_SIZE_KG)} × ({label_pct:.6f}% ÷ 100) × {factor:.9f} "
                           f"= {npk_fmt(elemental_per_bag)} kg {nutrient}")
        per_bag_rows.append([
            nutrient,
            f"{label_pct:.6f}% {label_basis}",
            calculation,
            f"{npk_fmt(elemental_per_bag)} kg {nutrient}/bag",
        ])
    per_bag_headers = ["Nutrient", "User label", "Calculation for one bag", "Element in one bag"]
    per_bag_widths = [max(len(per_bag_headers[index]), *(len(row[index]) for row in per_bag_rows))
                      for index in range(len(per_bag_headers))]
    print(" | ".join(header.ljust(per_bag_widths[index]) for index, header in enumerate(per_bag_headers)))
    print("-+-".join("-" * width for width in per_bag_widths))
    for row in per_bag_rows:
        print(" | ".join(value.ljust(per_bag_widths[index]) for index, value in enumerate(row)))

    rows = []
    for nutrient, baseline_rate in BASELINE_ELEMENTAL_KG_HA_YEAR.items():
        user_total = elemental_year[nutrient]
        user_rate = user_total
        difference = user_rate - baseline_rate
        difference_pct = difference / baseline_rate * 100.0 if baseline_rate else 0.0
        if abs(difference_pct) <= NUTRIENT_MATCH_TOLERANCE_PERCENT:
            status = f"CLOSE (within ±{NUTRIENT_MATCH_TOLERANCE_PERCENT:.1f}%)"
        elif difference > 0:
            status = f"ABOVE by {difference_pct:.2f}%"
        else:
            status = f"BELOW by {abs(difference_pct):.2f}%"
        rows.append([
            nutrient,
            npk_fmt(baseline_rate),
            npk_fmt(user_total),
            npk_fmt(difference),
            status,
        ])

    headers = [
        "Nutrient",
        "Baseline kg/ha/year",
        "User supplies kg/ha/year",
        "Difference kg/ha/year",
        "Comparison",
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print("\n" + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    print("\nCalculation details (elemental nutrient supply from recorded products):")
    for nutrient in BASELINE_ELEMENTAL_KG_HA_YEAR:
        baseline_rate = BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient]
        print(f"  {nutrient}:")
        if history_supply is not None:
            for product_row in history_supply["products"]:
                contribution = product_row["elemental_supply_kg_ha_year"][nutrient]
                if contribution == 0:
                    continue
                print(f"    {product_row['product']}: {npk_fmt(contribution)} kg {nutrient}/ha/year")
            user_rate = elemental_year[nutrient]
            print(f"    Total recorded {nutrient} = {npk_fmt(user_rate)} kg {nutrient}/ha/year")
        else:
            label_basis = {"N": "N", "P": "P2O5", "K": "K2O", "Mg": "MgO"}[nutrient]
            factor = 1.0 if nutrient == "N" else OXIDE_TO_ELEMENT[label_basis]["factor"]
            amount_in_label_basis = label_basis_year[label_basis]
            user_rate = elemental_year[nutrient]
            print(f"    Product used = {npk_fmt(npk_product_kg_ha_year)} kg/ha/year")
            print(f"    {label_basis} amount = {npk_fmt(npk_product_kg_ha_year)} × "
                  f"({label_percent[label_basis]:.6f}% ÷ 100) = {npk_fmt(amount_in_label_basis)} kg/ha/year")
            print(f"    Elemental conversion factor = {factor:.9f}; supply = {npk_fmt(user_rate)} kg/ha/year")
        print(f"    Compare with baseline: {npk_fmt(user_rate)} − {npk_fmt(baseline_rate)} "
              f"= {npk_fmt(user_rate - baseline_rate)} kg {nutrient}/ha/year")

    print_standalone_n_recommendation(elemental_year)
    print_standalone_p_recommendation(elemental_year)
    print_standalone_k_recommendation(elemental_year)
    print_standalone_mg_recommendation(elemental_year)

    print("\nField-wide NPK decision aid (separate from zone-specific boosts):")
    npk_shortfall_pct = {
        nutrient: max(0.0, BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient] - elemental_year[nutrient])
        / BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient] * 100.0
        for nutrient in ("N", "P", "K")
    }
    limiting_gap_nutrient = max(npk_shortfall_pct, key=npk_shortfall_pct.get)
    largest_gap_pct = npk_shortfall_pct[limiting_gap_nutrient]
    all_baseline_targets_met = all(
        elemental_year[nutrient] >= BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient]
        for nutrient in ("N", "P", "K", "Mg")
    )
    if all_baseline_targets_met:
        print("The current blend meets or exceeds the supplied N, P, K, and Mg baseline values.")
        print("Suggested route: no field-wide corrective fertilizer is indicated by this comparison.")
        print("If Part 1's image analysis recommends a positive VARI-based zone boost, apply that boost only; "
              "this NPK comparison does not calculate zone needs.")
    elif largest_gap_pct <= NPK_GAP_ROUTE_THRESHOLD_PERCENT:
        print(f"Largest N/P/K shortfall: {largest_gap_pct:.2f}% ({limiting_gap_nutrient}), within the "
              f"configured {NPK_GAP_ROUTE_THRESHOLD_PERCENT:.2f}% small-gap range.")
        print("Suggested route: do not add another bag of the NPK blend for this small gap. "
              "Use the configured single-nutrient top-up options for any nutrient gaps, "
              "and follow any positive VARI-based zone boost from Part 1.")
        print("Standalone product options for N, P, K, and Mg are calculated above; verify labels and field suitability.")
    else:
        print(f"Largest N/P/K shortfall: {largest_gap_pct:.2f}% ({limiting_gap_nutrient}), above "
              f"the configured {NPK_GAP_ROUTE_THRESHOLD_PERCENT:.2f}% cutoff.")
        print("Suggested route: first consider adding bags of the current blend or switching to a "
              "better-matched NPK blend. Keep this field-wide correction separate from zonal boosts.")
    print("The route cutoff only chooses the field-wide correction path. Standalone N, P, K, and Mg options are shown separately; coordinate these with any zone-specific boost.")

    print("\nBlend quantities to meet all field-wide N/P/K baseline targets (per hectare/year):")
    print("The estimate uses the most limiting of N, P, or K. Check any nutrient surplus before choosing a blend.")
    candidate_blends = [("Current blend", label_percent, USER_BAGS_PER_HA_PER_YEAR)]
    for candidate_name, candidate_label in ALTERNATIVE_NPK_PRODUCTS.items():
        candidate_analysis = parse_analysis(candidate_label)
        if candidate_analysis == label_percent:
            continue
        candidate_blends.append((candidate_name, candidate_analysis, 0.0))
    candidate_rows = []
    age_adjusted_blend_rows = []
    age_normal_blend_rows = []
    budget_multiplier_rows = []
    age_process_rows = []
    for candidate_name, candidate_analysis, already_used in candidate_blends:
        candidate_age_target = build_age_based_npk_target(candidate_analysis)
        _candidate_age_multiplier = candidate_age_target.get("commonize_multiplier") if candidate_age_target.get("available") else None
        _candidate_age_apps = candidate_age_target.get("fertilizer_applications_per_year") if candidate_age_target.get("available") else None
        for _cap_step_number, _cap_step in enumerate(
            candidate_age_target.get("budget_cap", {}).get("calculation_steps", []), start=1
        ):
            age_process_rows.append([
                candidate_name, f"Budget cap {_cap_step_number}", _cap_step
            ])
        one_bag_supply, _, _ = build_annual_nutrients(candidate_analysis, 1.0)
        missing_blend_nutrients = [nutrient for nutrient in ("N", "P", "K")
                                   if one_bag_supply[nutrient] <= 0]
        if missing_blend_nutrients:
            candidate_rows.append([
                candidate_name, "Cannot meet all N/P/K", "—", "—",
                "No " + "/".join(missing_blend_nutrients) + " in label",
                "N/A", "N/A", "N/A",
            ])
            age_adjusted_blend_rows.append([candidate_name, "—", "—", "—", "—", "N/A", "N/A", "N/A"])
            age_normal_blend_rows.append([candidate_name, "—", "—", "—", "—", "N/A", "N/A", "N/A"])
            budget_multiplier_rows.append([candidate_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "Blend cannot supply baseline N/P/K"])
            age_process_rows.append([candidate_name, "Baseline blend", "Cannot meet all baseline N/P/K targets with this blend."])
            continue
        # one_bag_supply is elemental kg of N, P, K, Mg supplied by one bag.
        # P and K are converted from the oxide form printed on fertilizer labels first.
        label_basis_by_element = {"N": "N", "P": "P2O5", "K": "K2O"}
        factor_by_element = {
            "N": 1.0,
            "P": OXIDE_TO_ELEMENT["P2O5"]["factor"],
            "K": OXIDE_TO_ELEMENT["K2O"]["factor"],
        }
        bags_required_each = {
            nutrient: BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient] / one_bag_supply[nutrient]
            for nutrient in ("N", "P", "K")
        }
        print(f"\n  {candidate_name} calculation per nutrient (one {npk_fmt(BAG_SIZE_KG)} kg bag):")
        for nutrient in ("N", "P", "K"):
            label_basis = label_basis_by_element[nutrient]
            label_percent_value = candidate_analysis[label_basis]
            label_mass_per_bag = BAG_SIZE_KG * label_percent_value / 100.0
            conversion_factor = factor_by_element[nutrient]
            elemental_per_bag = one_bag_supply[nutrient]
            target = BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient]
            needed_bags = bags_required_each[nutrient]
            if nutrient == "N":
                print(f"    {nutrient}: label is {label_percent_value:g}% N (already elemental).")
                print(f"      N per bag = {npk_fmt(BAG_SIZE_KG)} kg × ({label_percent_value:g} ÷ 100) = "
                      f"{npk_fmt(label_mass_per_bag)} kg N/bag.")
            else:
                print(f"    {nutrient}: label is {label_percent_value:g}% {label_basis}; convert oxide label mass to elemental {nutrient}.")
                print(f"      {label_basis} per bag = {npk_fmt(BAG_SIZE_KG)} kg × ({label_percent_value:g} ÷ 100) = "
                      f"{npk_fmt(label_mass_per_bag)} kg {label_basis}/bag.")
                print(f"      Elemental {nutrient} per bag = {npk_fmt(label_mass_per_bag)} kg {label_basis} × "
                      f"{conversion_factor:.9f} kg {nutrient}/kg {label_basis} = "
                      f"{npk_fmt(elemental_per_bag)} kg {nutrient}/bag.")
            print(f"      Bags to meet baseline = {npk_fmt(target)} kg {nutrient}/ha/year ÷ "
                  f"{npk_fmt(elemental_per_bag)} kg {nutrient}/bag = {npk_fmt(needed_bags)} bags/ha/year.")
        exact_total = max(bags_required_each.values())
        limiting_nutrient = max(bags_required_each, key=bags_required_each.get)
        print(f"    One blend amount must meet all three targets, so use the largest result: "
              f"max({', '.join(f'{nutrient} {npk_fmt(value)}' for nutrient, value in bags_required_each.items())}) "
              f"= {npk_fmt(exact_total)} bags/ha/year; limiting nutrient = {limiting_nutrient}.")
        whole_total = math.ceil(exact_total - 1e-12)
        additional_whole_bags = max(0, math.ceil(exact_total - already_used - 1e-12))
        price_per_bag = npk_label_price_per_bag_rp(candidate_analysis)
        baseline_full_bag_cost = whole_total * price_per_bag if price_per_bag is not None else None
        additional_full_bag_cost = additional_whole_bags * price_per_bag if price_per_bag is not None else None
        baseline_product_kg_ha = exact_total * BAG_SIZE_KG
        if _candidate_age_multiplier is not None and USER_DETECTED_TREE_COUNT > 0 and USER_FARM_AREA_HA > 0:
            age_product_kg_ha = baseline_product_kg_ha * _candidate_age_multiplier
            baseline_farm_kg = baseline_product_kg_ha * USER_FARM_AREA_HA
            baseline_tree_kg_year = baseline_farm_kg / USER_DETECTED_TREE_COUNT
            base_units_per_application = baseline_tree_kg_year / (NPK_BASE_DOSE_KG_PER_TREE_PER_APPLICATION * _candidate_age_apps)
            adjusted_units_per_application = base_units_per_application * _candidate_age_multiplier
            adjusted_dose_per_application = adjusted_units_per_application * NPK_BASE_DOSE_KG_PER_TREE_PER_APPLICATION
            adjusted_tree_kg_year = adjusted_dose_per_application * _candidate_age_apps
            age_product_kg_tree_year = age_product_kg_ha * USER_FARM_AREA_HA / USER_DETECTED_TREE_COUNT
            adjusted_exact_bags = exact_total * _candidate_age_multiplier
            adjusted_full_bags = math.ceil(adjusted_exact_bags - 1e-12)
            adjusted_full_bag_cost = adjusted_full_bags * price_per_bag if price_per_bag is not None else None
            _candidate_configured_multiplier = candidate_age_target.get("configured_age_multiplier")
            if _candidate_configured_multiplier is None:
                _candidate_configured_multiplier = 1.0
            _candidate_multiplier_label = (
                f"{_candidate_configured_multiplier:.2f}× → {_candidate_age_multiplier:.2f}×"
                if abs(_candidate_configured_multiplier - _candidate_age_multiplier) > 1e-12
                else f"{_candidate_age_multiplier:.2f}×"
            )
            _normal_exact_bags = exact_total * _candidate_configured_multiplier
            _normal_full_bags = math.ceil(_normal_exact_bags - 1e-12)
            _normal_cost = _normal_full_bags * price_per_bag if price_per_bag is not None else None
            _normal_kg_tree_year = baseline_product_kg_ha * _candidate_configured_multiplier * USER_FARM_AREA_HA / USER_DETECTED_TREE_COUNT
            age_normal_blend_rows.append([candidate_name, npk_fmt(exact_total),
                f"{_candidate_configured_multiplier:.2f}×", npk_fmt(_normal_exact_bags),
                str(_normal_full_bags), "N/A" if price_per_bag is None else f"Rp{price_per_bag:,.0f}",
                "N/A" if _normal_cost is None else f"Rp{_normal_cost:,.0f}",
                npk_fmt(_normal_kg_tree_year)])
            age_adjusted_blend_rows.append([candidate_name, npk_fmt(exact_total),
                f"{_candidate_age_multiplier:.2f}×", npk_fmt(adjusted_exact_bags),
                str(adjusted_full_bags), "N/A" if price_per_bag is None else f"Rp{price_per_bag:,.0f}",
                "N/A" if adjusted_full_bag_cost is None else f"Rp{adjusted_full_bag_cost:,.0f}",
                npk_fmt(age_product_kg_tree_year)])
            _budget_info = candidate_age_target.get("budget_cap", {})
            _baseline_farm_bags = _budget_info.get("baseline_exact_bags_for_farm_year")
            if _baseline_farm_bags is None:
                _baseline_farm_bags = exact_total * float(USER_FARM_AREA_HA)
            _budget_cap_value = _budget_info.get("maximum_multiplier_allowed_by_budget")
            _budget_cap_label = f"{_budget_cap_value:.3f}×" if _budget_cap_value is not None else (
                "No cap" if _budget_info.get("status") in {"budget not entered", "budget cap disabled"}
                else "Baseline protected / unavailable"
            )
            _combined_cost = _budget_info.get("estimated_combined_whole_bag_cost_rp")
            _budget_status = _budget_info.get("status", "budget status unavailable")
            _budget_input = _budget_info.get("budget_rp")
            _allowed_input = _budget_info.get("allowed_spend_rp")
            _affordable_bags = _budget_info.get("maximum_affordable_npk_full_bags")
            budget_multiplier_rows.append([
                candidate_name,
                "N/A" if _budget_input is None else f"Rp{_budget_input:,.0f}",
                "N/A" if _allowed_input is None else f"Rp{_allowed_input:,.0f}",
                f"{_baseline_farm_bags:.3f}",
                f"{_candidate_configured_multiplier:.2f}×",
                _budget_cap_label,
                "N/A" if _affordable_bags is None else str(_affordable_bags),
                f"{_candidate_age_multiplier:.2f}×",
                f"{_baseline_farm_bags * _candidate_age_multiplier:.3f}",
                "N/A" if _combined_cost is None else f"Rp{_combined_cost:,.0f}",
                _budget_status,
            ])
            age_process_rows.extend([
                [candidate_name, "Budget-aware age factor",
                 f"Commonize {_candidate_configured_multiplier:.2f}× → applied {_candidate_age_multiplier:.2f}×; "
                 f"{candidate_age_target.get('budget_cap', {}).get('status', 'budget status unavailable')}"],
                [candidate_name, "1. Baseline blend", f"{npk_fmt(exact_total)} bags/ha/year × {npk_fmt(BAG_SIZE_KG)} kg/bag = {npk_fmt(baseline_product_kg_ha)} kg/ha/year"],
                [candidate_name, "2. Farm total", f"{npk_fmt(baseline_product_kg_ha)} kg/ha/year × {npk_fmt(USER_FARM_AREA_HA)} ha = {npk_fmt(baseline_farm_kg)} kg/farm/year"],
                [candidate_name, "3. Per tree before age", f"{npk_fmt(baseline_farm_kg)} kg/farm/year ÷ {USER_DETECTED_TREE_COUNT} trees = {npk_fmt(baseline_tree_kg_year)} kg/tree/year"],
                [candidate_name, "4. 2 kg base units", f"{npk_fmt(baseline_tree_kg_year)} ÷ (2 kg × {npk_fmt(_candidate_age_apps)} applications/year) = {npk_fmt(base_units_per_application)} units/tree/application"],
                [candidate_name, "5. Apply age factor", f"{npk_fmt(base_units_per_application)} × {_candidate_age_multiplier:.2f} × 2 kg = {npk_fmt(adjusted_dose_per_application)} kg/tree/application"],
                [candidate_name, "6. Annual per tree", f"{npk_fmt(adjusted_dose_per_application)} kg/application × {npk_fmt(_candidate_age_apps)} applications/year = {npk_fmt(adjusted_tree_kg_year)} kg/tree/year"],
                [candidate_name, "7. Adjusted bags", f"{npk_fmt(exact_total)} baseline bags/ha × {_candidate_age_multiplier:.2f} = {npk_fmt(exact_total * _candidate_age_multiplier)} exact bags/ha/year"],
            ])
        else:
            age_adjusted_blend_rows.append([candidate_name, npk_fmt(exact_total), "N/A", "N/A", "N/A",
                "N/A" if price_per_bag is None else f"Rp{price_per_bag:,.0f}", "N/A", "N/A"])
            age_normal_blend_rows.append([candidate_name, npk_fmt(exact_total), "N/A", "N/A", "N/A",
                "N/A" if price_per_bag is None else f"Rp{price_per_bag:,.0f}", "N/A", "N/A"])
            _baseline_farm_bags = exact_total * float(USER_FARM_AREA_HA)
            _budget_info = candidate_age_target.get("budget_cap", {})
            budget_multiplier_rows.append([
                candidate_name, "N/A", "N/A", f"{_baseline_farm_bags:.3f}", "N/A", "N/A", "N/A",
                "N/A", "N/A", "N/A", candidate_age_target.get("reason", "age multiplier unavailable"),
            ])
            age_process_rows.append([candidate_name, "Age adjustment", f"Unavailable: {age_target.get('reason', 'farm area or tree count is unavailable')}."])
        candidate_rows.append([
            candidate_name,
            npk_fmt(exact_total),
            str(whole_total),
            str(additional_whole_bags),
            limiting_nutrient,
            "N/A" if price_per_bag is None else f"Rp{price_per_bag:,.0f}",
            "N/A" if baseline_full_bag_cost is None else f"Rp{baseline_full_bag_cost:,.0f}",
            "N/A" if additional_full_bag_cost is None else f"Rp{additional_full_bag_cost:,.0f}",
        ])
        print(f"  {candidate_name}: at {whole_total} full bags/ha/year, estimated elemental supply is:")
        for nutrient in ("N", "P", "K", "Mg"):
            delivered = one_bag_supply[nutrient] * whole_total
            difference = delivered - BASELINE_ELEMENTAL_KG_HA_YEAR[nutrient]
            print(f"    {nutrient}: {npk_fmt(delivered)} kg/ha/year "
                  f"({difference:+.{OUTPUT_DECIMALS}f} kg/ha/year vs baseline)")
        if already_used > 0:
            print(f"    Current use is {npk_fmt(already_used)} bags/ha/year; buy up to "
                  f"{additional_whole_bags} additional full bag(s)/ha/year to reach this estimate.")
        else:
            print("    This is a replacement-blend estimate; do not add it on top of the current blend amount.")
        if one_bag_supply["Mg"] == 0:
            print("    This blend supplies no Mg; any Mg need remains separate.")
    candidate_headers = ["Blend option", "Exact total bags/ha", "Full bags total", "Additional bags", "Limiting nutrient",
                         "Price/bag", "Full-bag total cost/ha", "Additional-bag cost/ha"]
    candidate_widths = [max(len(candidate_headers[index]), *(len(row[index]) for row in candidate_rows))
                        for index in range(len(candidate_headers))]
    print("\n" + " | ".join(header.ljust(candidate_widths[index])
                              for index, header in enumerate(candidate_headers)))
    print("-+-".join("-" * width for width in candidate_widths))
    for row in candidate_rows:
        print(" | ".join(value.ljust(candidate_widths[index]) for index, value in enumerate(row)))
    print("Costs use the supplied price per bag; full-bag total and additional-bag costs are per hectare/year. Missing prices show N/A.")

    age_headers = ["Blend option", "Baseline exact bags/ha", "Multiplier",
                   "Adjusted exact bags/ha", "Adjusted full bags/ha", "Price/bag",
                   "Full-bag cost/ha", "Adjusted kg/tree/year"]
    print("\nAge-adjusted quantities — normal age multiplier (no budget cap):")
    age_widths = [max(len(age_headers[i]), *(len(row[i]) for row in age_normal_blend_rows))
                  for i in range(len(age_headers))]
    print(" | ".join(h.ljust(age_widths[i]) for i, h in enumerate(age_headers)))
    print("-+-".join("-" * width for width in age_widths))
    for row in age_normal_blend_rows:
        print(" | ".join(value.ljust(age_widths[i]) for i, value in enumerate(row)))
    print("\nAge-adjusted quantities — budget-capped multiplier:")
    age_widths = [max(len(age_headers[i]), *(len(row[i]) for row in age_adjusted_blend_rows))
                  for i in range(len(age_headers))]
    print(" | ".join(h.ljust(age_widths[i]) for i, h in enumerate(age_headers)))
    print("-+-".join("-" * width for width in age_widths))
    for row in age_adjusted_blend_rows:
        print(" | ".join(value.ljust(age_widths[i]) for i, value in enumerate(row)))
    print("\nBaseline, age multiplier, and budget cap comparison:")
    budget_headers = ["Blend", "Budget input", "Allowed incl. margin", "Baseline exact bags/farm/year",
                      "Age target", "Budget cap", "Affordable NPK bags", "Applied multiplier",
                      "Adjusted exact bags/farm/year", "Combined cost", "Status / explanation"]
    budget_widths = [max(len(budget_headers[i]), *(len(row[i]) for row in budget_multiplier_rows))
                     for i in range(len(budget_headers))]
    print(" | ".join(h.ljust(budget_widths[i]) for i, h in enumerate(budget_headers)))
    print("-+-".join("-" * width for width in budget_widths))
    for row in budget_multiplier_rows:
        print(" | ".join(value.ljust(budget_widths[i]) for i, value in enumerate(row)))
    print("Baseline bags meet the N/P/K targets. Age target is the Commonize factor. Budget cap is the largest factor affordable within the entered budget plus margin, after reserving the selected zone-product costs.")

    print("\nStep-by-step calculation process:")
    process_headers = ["Blend option", "Step", "Calculation"]
    process_widths = [max(len(process_headers[i]), *(len(row[i]) for row in age_process_rows))
                      for i in range(len(process_headers))]
    print(" | ".join(h.ljust(process_widths[i]) for i, h in enumerate(process_headers)))
    print("-+-".join("-" * width for width in process_widths))
    for row in age_process_rows:
        print(" | ".join(value.ljust(process_widths[i]) for i, value in enumerate(row)))
    print("\nRaw budget-cap formula (calculated separately for each blend):")
    print("  B = user-entered annual fertilizer budget (Rp)")
    print("  m = allowed over-budget margin (%)")
    print("  Margin factor = 1 + (m/100); 1 means the original 100% budget, and m/100 is the extra margin fraction.")
    print("    Example: m=15% → 1 + 15/100 = 1.15, allowing the budget plus 15%.")
    print("  A = B × [1 + (m/100)]                       [maximum allowed spend, Rp]")
    print("  Z = selected zone Mg + Urea whole-bag cost (Rp)")
    print("  R = A − Z                                   [budget left for NPK, Rp]")
    print("  P = this blend's configured price per bag (Rp/bag)")
    print("  F = floor(max(0, R) ÷ P)                    [affordable whole NPK bags]")
    print("  Q = baseline exact NPK bags/ha × farm area (ha) [baseline exact bags for farm/year]")
    print("  Extra bag capacity = F − Q; affordable increase fraction = (F − Q) ÷ Q.")
    print("  C = 1 + [(F − Q) ÷ Q] = F ÷ Q              [baseline 1× plus affordable headroom]")
    print("  M_age = Commonize multiplier selected from tree age")
    print("  M_applied = min(M_age, max(1.00, C))          [retain at least the 1× nutrient baseline]")
    print("  Adjusted exact bags = Q × M_applied; whole bags to buy = ceil(adjusted exact bags).")
    print("  Estimated combined cost = whole NPK bags × P + Z.")
    print("\nIn plain words: baseline sets the nutrient amount, age sets the desired scale-up, and the budget limits the scale-up after zone-product costs are reserved.")
    if UI_SUBMISSION_DATA.get("fertilizerBudgetRp") is None:
        print("  No budget value is entered yet, so this run retains each Commonize multiplier without budget capping.")

    print("\nImportant: the table compares fertilizer label composition with the supplied baseline. "
          "Check that the label guarantees and the baseline use the same elemental/oxide basis. "
          "The cutoff and blend quantities are configurable estimates, not agronomic prescriptions. "
          "Confirm rates and baseline with an agronomist using field data; "
          "this script is not an agronomic diagnosis.")

# ## E. Run and display the report
#
# This reads `user_submission_request.json` from the notebook kernel’s current working directory, outside the generated `outputs/` folder. Farm area, profile fields, fertilizer history, harvest history, and review month flow into `ui_data`; NPK label, dose per tree, fertilizer frequency, detected tree count, and farm area determine current nutrient supply. Recognized standalone products in the fertilizer history contribute their nutrients too; bag equivalents are derived from the configured bag size. The image-derived zone analysis remains based on the image/model. This section appends calculated field-wide NPK and zone boost results to the zone JSON, including the whole-farm closest-to-budget ranking for each age-adjusted blend option. Run from the beginning.


# ===== Notebook code cell 54 =====
analysis = parse_analysis(USER_FERTILIZER_ANALYSIS)
report_buffer = _NPKStringIO()
with _npk_redirect_stdout(report_buffer):
    print_npk_comparison(analysis)
_npk_display(_NPKHTML(
    "<pre style='white-space:pre; overflow-x:auto; padding:12px; "
    "background:#f7f8fa; border:1px solid #d6d9de; border-radius:8px;'>"
    + _npk_html.escape(report_buffer.getvalue()) + "</pre>"
))

# Merge field-wide NPK/top-up calculations into the existing zone-analysis JSON.
# Per-zone VARI-based N/Mg boosts remain in their own scope and are included beside them.
combined_json_path = Path(OUTPUT_JSON)
if not combined_json_path.exists():
    raise FileNotFoundError(
        f"Zone JSON not found at {combined_json_path}; run the notebook's image and zoning steps first."
    )
with combined_json_path.open("r", encoding="utf-8") as _json_file:
    combined_result = json.load(_json_file)
combined_result["combined_fertilizer_calculations"] = {
    "field_wide_npk_baseline_comparison": build_field_npk_json_record(analysis),
    "zone_vari_boost_calculations": build_zone_boost_json_records(combined_result.get("zones", [])),
    "scope_note": (
        "Field-wide standalone top-ups close the configured NPK baseline gap. Zone VARI boosts are separate "
        "additional zone calculations already derived from confidence, VARI, zone area, and configured Mg/N rates. "
        "They are stored together here for review but are not automatically added or netted against each other; "
        "check for overlap before deciding field application quantities."
    ),
}
with combined_json_path.open("w", encoding="utf-8") as _json_file:
    json.dump(combined_result, _json_file, indent=2, ensure_ascii=False, allow_nan=False)
print(f"Combined field NPK and zone boost calculations saved to: {combined_json_path.resolve()}")

# Remove the former app-facing block from an existing JSON file as well.
combined_result.pop("ui_data", None)
with combined_json_path.open("w", encoding="utf-8") as _json_file:
    json.dump(combined_result, _json_file, indent=2, ensure_ascii=False, allow_nan=False)
print("Combined fertilizer calculations saved without the ui_data block.")

# Export submitted product use plus the applicable VARI-derived zone supplements.
# The supplement comes from product_kg_per_zone_per_year (supplement only), so a
# configured base dose from fertilizer history is added once and is never double-counted.
_submission_farm = UI_SUBMISSION_DATA.get("farm", {})
_submission_history = UI_SUBMISSION_DATA.get("fertilizerHistory") or {}
_submission_frequency = _submission_history.get("freq")
_submission_tree_count = int(USER_DETECTED_TREE_COUNT)

# Add up the VARI-only product mass across zones for each supported standalone product.
_vari_supplement_kg_by_product = {
    product: 0.0 for product in ("dolomite_Mg_cheaper", "kieserite_Mg_expensive", "urea_N_expensive")
}
_vari_supplement_known = {product: True for product in _vari_supplement_kg_by_product}
for _mass_row in product_mass_rows:
    _product_key = _mass_row.get("product")
    if _product_key not in _vari_supplement_kg_by_product:
        continue
    _zone_supplement_kg = _mass_row.get("product_kg_per_zone_per_year")
    if _zone_supplement_kg is None:
        _vari_supplement_known[_product_key] = False
    else:
        _vari_supplement_kg_by_product[_product_key] += float(_zone_supplement_kg)
for _product_key, _is_known in _vari_supplement_known.items():
    if not _is_known:
        _vari_supplement_kg_by_product[_product_key] = None

# Match the Mg supplement to the product the farmer already uses when available.
# If no Mg product is in the submission, use Dolomite as the default summary option.
_history_entries = _submission_history.get("fertilizers", [])
def _summary_product_key(_name):
    _normalized_name = str(_name or "").strip().casefold()
    if _normalized_name in ("dolomite", "dolomit"):
        return "dolomite_Mg_cheaper"
    if _normalized_name in ("kieserite", "kieserit"):
        return "kieserite_Mg_expensive"
    if _normalized_name == "urea":
        return "urea_N_expensive"
    return None

_history_mg_keys = [_summary_product_key(_entry.get("name")) for _entry in _history_entries]
if "dolomite_Mg_cheaper" in _history_mg_keys:
    _selected_mg_product = "dolomite_Mg_cheaper"
elif "kieserite_Mg_expensive" in _history_mg_keys:
    _selected_mg_product = "kieserite_Mg_expensive"
else:
    _selected_mg_product = "dolomite_Mg_cheaper"

_age_adjusted_npk_target = (
    combined_result.get("combined_fertilizer_calculations", {})
    .get("field_wide_npk_baseline_comparison", {})
    .get("age_based_npk_product_target", {})
)
_age_adjusted_npk_total_kg = (
    float(_age_adjusted_npk_target["target_npk_product_kg_farm_year"])
    if _age_adjusted_npk_target.get("available")
    and _age_adjusted_npk_target.get("target_npk_product_kg_farm_year") is not None
    else None
)

# Carry every blend option from the notebook's age-adjusted blend table into the
# NPK fertilizer record. Farm totals are reconstructed from exact bags/ha so
# current blend and alternatives use one consistent basis.
_age_adjusted_npk_quantities = []
_npk_comparison = combined_result.get("combined_fertilizer_calculations", {}).get(
    "field_wide_npk_baseline_comparison", {}
)
_age_multiplier_available = bool(_age_adjusted_npk_target.get("available"))
_age_farm_area_ha = _age_adjusted_npk_target.get("farm_area_ha")
_age_tree_count = _age_adjusted_npk_target.get("detected_tree_count")
_age_frequency = _age_adjusted_npk_target.get("fertilizer_applications_per_year")
for _blend in _npk_comparison.get("blend_purchase_price_estimates", []):
    _adjusted_bags_ha = _blend.get("age_adjusted_exact_bags_per_ha_year")
    _baseline_bags_ha = _blend.get("baseline_exact_bags_per_ha_year")
    _price_per_bag = _blend.get("price_per_bag_rp")
    if _age_multiplier_available and _adjusted_bags_ha is not None and _age_farm_area_ha is not None:
        _adjusted_kg_ha = float(_adjusted_bags_ha) * float(BAG_SIZE_KG)
        _adjusted_kg_farm = _adjusted_kg_ha * float(_age_farm_area_ha)
        _adjusted_bags_farm = float(_adjusted_bags_ha) * float(_age_farm_area_ha)
        _full_bags_farm = int(math.ceil(_adjusted_bags_farm - 1e-12)) if _adjusted_bags_farm > 0 else 0
        _adjusted_kg_tree_year = (
            _adjusted_kg_farm / int(_age_tree_count)
            if _age_tree_count else None
        )
        _adjusted_kg_tree_application = (
            _adjusted_kg_tree_year / float(_age_frequency)
            if _adjusted_kg_tree_year is not None and _age_frequency else None
        )
    else:
        _adjusted_kg_ha = _adjusted_kg_farm = _adjusted_bags_farm = None
        _full_bags_farm = None
        _adjusted_kg_tree_year = _adjusted_kg_tree_application = None
    _age_adjusted_npk_quantities.append({
        "blendOption": _blend.get("blend_option"),
        "labelOrder": "N-P2O5-K2O-MgO",
        "composition": _blend.get("label_percent_n_p2o5_k2o_mgo"),
        "limitingNutrient": _blend.get("limiting_nutrient"),
        "baselineExactBagsPerHaYear": _baseline_bags_ha,
        "ageMultiplier": _blend.get("age_multiplier"),
        "ageAdjustedExactBagsPerHaYear": _adjusted_bags_ha,
        "ageAdjustedFullBagsPerHaYear": _blend.get("age_adjusted_full_bags_per_ha_year"),
        "ageAdjustedExactBagsForFarmYear": _adjusted_bags_farm,
        "ageAdjustedFullBagsForFarmYear": _full_bags_farm,
        "bagKg": float(BAG_SIZE_KG),
        "ageAdjustedProductKgPerHaYear": _adjusted_kg_ha,
        "ageAdjustedProductKgPerFarmYear": _adjusted_kg_farm,
        "ageAdjustedProductKgPerTreeYear": _adjusted_kg_tree_year,
        "ageAdjustedDoseKgPerTreePerApplication": _adjusted_kg_tree_application,
        "pricePerBagRp": _price_per_bag,
        "ageAdjustedExactCostForFarmRp": (
            _adjusted_bags_farm * float(_price_per_bag)
            if _adjusted_bags_farm is not None and _price_per_bag is not None else None
        ),
        "ageAdjustedFullBagCostForFarmRp": (
            _full_bags_farm * float(_price_per_bag)
            if _full_bags_farm is not None and _price_per_bag is not None else None
        ),
        "available": _age_multiplier_available and _adjusted_bags_ha is not None,
        "unavailableReason": None if _age_multiplier_available else _age_adjusted_npk_target.get("reason"),
    })

_fertilizer_summary = {
    "latitude": _submission_farm.get("latitude"),
    "longitude": _submission_farm.get("longitude"),
    "fertilizerFreq": _submission_frequency,
    "fertilizers": [],
}
_supplement_attached = set()
for _entry in _history_entries:
    _fertilizer_output = {
        "name": _entry.get("name"),
        "bagKg": _entry.get("bagKg"),
        "id": _entry.get("id"),
    }
    if "composition" in _entry:
        _fertilizer_output["composition"] = _entry["composition"]
    _dose_per_tree = _entry.get("dosePerTree")
    if _dose_per_tree is None or _submission_frequency is None:
        _overall_product_kg = None
    else:
        _overall_product_kg = (
            float(_dose_per_tree) * float(_submission_frequency) * _submission_tree_count
        )
    _product_key = _summary_product_key(_entry.get("name"))
    if str(_entry.get("name", "")).strip().upper() == "NPK":
        # Use the age-adjusted whole-farm target from the NPK calculation,
        # not the farmer's submitted dose or the unadjusted baseline amount.
        _overall_product_kg = _age_adjusted_npk_total_kg
    if _product_key == "urea_N_expensive" or _product_key == _selected_mg_product:
        if _product_key not in _supplement_attached:
            _vari_extra_kg = _vari_supplement_kg_by_product.get(_product_key)
            _overall_product_kg = (
                None if _overall_product_kg is None or _vari_extra_kg is None
                else _overall_product_kg + _vari_extra_kg
            )
            _supplement_attached.add(_product_key)
    _fertilizer_output["overallDoseWeightInKg"] = _overall_product_kg
    _fertilizer_summary["fertilizers"].append(_fertilizer_output)

# Represent each alternative blend as another ordinary NPK fertilizer entry.
# The existing submitted NPK entry above represents the current blend.
_has_submitted_npk = any(
    str(_entry.get("name", "")).strip().upper() == "NPK" for _entry in _history_entries
)
for _npk_option in _age_adjusted_npk_quantities:
    if _npk_option.get("blendOption") == "Current blend" and _has_submitted_npk:
        continue
    _label = _npk_option.get("composition") or {}
    _fertilizer_summary["fertilizers"].append({
        "name": "NPK",
        "composition": {
            "N": _label.get("N"),
            "P": _label.get("P2O5"),
            "K": _label.get("K2O"),
            "Mg": _label.get("MgO"),
        },
        "bagKg": float(BAG_SIZE_KG),
        "id": None,
        "overallDoseWeightInKg": _npk_option.get("ageAdjustedProductKgPerFarmYear"),
    })

# Urea is the selected standalone N supplement in this summary. Include it even
# when it was not part of fertilizer history; its total is then the VARI supplement.
if "urea_N_expensive" not in _supplement_attached:
    _urea_supplement_kg = _vari_supplement_kg_by_product.get("urea_N_expensive")
    _fertilizer_summary["fertilizers"].append({
        "name": "Urea",
        "bagKg": FERTILIZER_BAG_SIZE_KG["urea_N_expensive"],
        "id": None,
        "overallDoseWeightInKg": _urea_supplement_kg,
    })

# If no user Mg product was entered, append the default Dolomite supplement option.
if not any(_key in _history_mg_keys for _key in ("dolomite_Mg_cheaper", "kieserite_Mg_expensive")):
    _default_dolomite_rate = FERTILIZER_BASE_PRODUCT_KG_PER_PALM_PER_YEAR.get("dolomite_Mg_cheaper")
    _default_dolomite_base_kg = (
        float(_default_dolomite_rate) * _submission_tree_count
        if _default_dolomite_rate is not None else 0.0
    )
    _dolomite_supplement_kg = _vari_supplement_kg_by_product.get("dolomite_Mg_cheaper")
    _default_dolomite_total_kg = (
        None if _dolomite_supplement_kg is None
        else _default_dolomite_base_kg + _dolomite_supplement_kg
    )
    _fertilizer_summary["fertilizers"].append({
        "name": "Dolomite",
        "bagKg": FERTILIZER_BAG_SIZE_KG["dolomite_Mg_cheaper"],
        "id": None,
        "overallDoseWeightInKg": _default_dolomite_total_kg,
    })

with Path(OUTPUT_FERTILIZER_HISTORY_JSON).open("w", encoding="utf-8") as _summary_file:
    json.dump(_fertilizer_summary, _summary_file, indent=2, ensure_ascii=False, allow_nan=False)
print(f"Fertilizer history + VARI supplement summary saved to: {Path(OUTPUT_FERTILIZER_HISTORY_JSON).resolve()}")

# ## Example — real oil-palm NPKMg blend with standalone top-ups
#
# This example uses PUSRI's **NPK 13-6-27-4+0.65B**, listed for mature oil palm and sold in 25 kg and 50 kg packaging. See the [PUSRI product page](https://www.pusri.co.id/index.php/en/product/npk/Npk-13-6-27-4) for the product and package sizes. The grade is expressed as N–P₂O₅–K₂O–MgO; confirm the guaranteed analysis on the actual bag before applying.
#
# At **10 bags × 50 kg/ha/year**, the blend provides approximately N 65.00, P 13.09, K 112.07, and Mg 12.06 kg elemental/ha/year. Against the current baseline (N 90, P 10, K 111, Mg 13), N is 27.8% below and Mg is 7.2% below, while P and K meet or exceed baseline. Since the largest N/P/K gap is within the configured 30% routing threshold, the example evaluates standalone N, P, K, and Mg options instead of recommending another NPK bag. Here, only N and Mg have shortfalls; P and K tables show no standalone top-up is needed. The P/K calculations will recommend SP-36 or KCl/MOP whenever their elemental gaps are positive. This is a comparison example, not an application-rate recommendation; check the product label and field-specific advice.


# ===== Notebook code cell 56 =====
# Real commercial NPKMg formulation and 50 kg bag, linked in the note above.
EXAMPLE_PRODUCT = "PUSRI NPK 13-6-27-4+0.65B"
EXAMPLE_LABEL = "13-6-27-4"  # N-P2O5-K2O-MgO label percentages; B is not part of this NPKMg comparison.
EXAMPLE_BAG_SIZE_KG = 50.0
EXAMPLE_BAGS_PER_HA_PER_YEAR = 10.0
example_analysis = parse_analysis(EXAMPLE_LABEL)
example_supply, _, example_product_kg = build_annual_nutrients(
    example_analysis, EXAMPLE_BAGS_PER_HA_PER_YEAR, bag_size_kg=EXAMPLE_BAG_SIZE_KG
)
example_rows = []
for nutrient, baseline in BASELINE_ELEMENTAL_KG_HA_YEAR.items():
    supplied = example_supply[nutrient]
    difference = supplied - baseline
    difference_pct = difference / baseline * 100.0 if baseline else 0.0
    status = (f"ABOVE by {difference_pct:.2f}%" if difference > 0
              else f"BELOW by {abs(difference_pct):.2f}%" if difference < 0
              else "MATCH")
    example_rows.append([nutrient, npk_fmt(baseline), npk_fmt(supplied),
                         f"{difference:+.{OUTPUT_DECIMALS}f}", status])
print(f"Example: {EXAMPLE_PRODUCT}; {EXAMPLE_BAGS_PER_HA_PER_YEAR:g} bags/ha/year × "
      f"{EXAMPLE_BAG_SIZE_KG:g} kg/bag = {npk_fmt(example_product_kg)} kg product/ha/year")
example_table = "<table style='border-collapse:collapse;font-size:14px'><tr style='background:#263238;color:white'>"
example_table += "".join(f"<th style='padding:6px'>{_npk_html.escape(h)}</th>" for h in
                        ["Nutrient", "Baseline kg/ha/year", "Blend supplies kg/ha/year", "Difference kg/ha/year", "Result"])
example_table += "</tr>"
for row in example_rows:
    example_table += "<tr>" + "".join(
        f"<td style='padding:6px;border-bottom:1px solid #ddd'>{_npk_html.escape(value)}</td>"
        for value in row
    ) + "</tr>"
example_table += "</table>"
_npk_display(_NPKHTML(example_table))
example_npk_gap_pct = max(
    max(0.0, BASELINE_ELEMENTAL_KG_HA_YEAR[n] - example_supply[n])
    / BASELINE_ELEMENTAL_KG_HA_YEAR[n] * 100.0
    for n in ("N", "P", "K")
)
if example_npk_gap_pct <= NPK_GAP_ROUTE_THRESHOLD_PERCENT:
    print(f"Largest N/P/K shortfall = {example_npk_gap_pct:.3f}%, within the "
          f"{NPK_GAP_ROUTE_THRESHOLD_PERCENT:.1f}% cutoff: no extra NPK bag is suggested.")
else:
    print(f"Largest N/P/K shortfall = {example_npk_gap_pct:.3f}%, above the configured cutoff; "
          "this is not the small-gap route.")
example_buffer = _NPKStringIO()
with _npk_redirect_stdout(example_buffer):
    print_standalone_n_recommendation(example_supply)
    print_standalone_p_recommendation(example_supply)
    print_standalone_k_recommendation(example_supply)
    print_standalone_mg_recommendation(example_supply)
_npk_display(_NPKHTML(
    "<pre style='white-space:pre;overflow-x:auto;padding:12px;background:#f7f8fa;"
    "border:1px solid #d6d9de;border-radius:8px'>"
    + _npk_html.escape(example_buffer.getvalue()) + "</pre>"
))
print("The NPK-versus-baseline comparison does not override product-label or field-specific application directions.")
print("If Part 1 recommends a positive zone boost, use its zone-level amounts separately/in coordination with these field-wide single-nutrient top-ups.")

# ## Final overview — submitted data, zone products, and annual NPK budget fit
#
# This recap uses the submitted farm/profile values and fertilizer history, the zone product totals already calculated above, and the annual age-adjusted NPK blend options. Each blend age multiplier is reduced when needed to keep whole-bag annual cost within the submitted budget plus the configurable margin; the baseline blend stays at a minimum of 1×. Budget fit compares estimated whole-bag annual purchases for each NPK option plus the selected zone Mg product and Urea, when their prices are configured. This is a cost comparison; nutrient fit and field-specific advice still matter. The budget is treated as an annual whole-farm budget in Rupiah.


# ===== Notebook code cell 58 =====
# Compact recap of the input, zone-level product quantities, and NPK budget fit.
_OverviewHTML = HTML
_overview_display = display
import html as _overview_html
import math as _overview_math
import re as _overview_re

_overview_decimals = int(OUTPUT_DECIMALS)
def _overview_num(value, decimals=_overview_decimals):
    if value is None:
        return "Not provided"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)

def _overview_rp(value):
    return "N/A" if value is None else f"Rp{float(value):,.0f}"

def _overview_table(title, headers, rows):
    html = [f"<h3>{_overview_html.escape(title)}</h3>", "<table style='border-collapse:collapse;width:100%;margin:8px 0 20px'>"]
    html.append("<thead><tr>" + "".join(
        f"<th style='text-align:left;padding:7px;border-bottom:2px solid #999;background:#eef2f5'>{_overview_html.escape(str(h))}</th>"
        for h in headers
    ) + "</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>" + "".join(
            f"<td style='padding:7px;border-bottom:1px solid #ddd;vertical-align:top'>{_overview_html.escape(str(value))}</td>"
            for value in row
        ) + "</tr>")
    html.append("</tbody></table>")
    _overview_display(_OverviewHTML("".join(html)))

# 1) Present every submitted field as readable category / field / value rows.
_overview_input_rows = []
def _overview_label(key):
    return _overview_re.sub(r"(?<!^)(?=[A-Z])", " ", str(key)).replace("_", " ").strip().capitalize()
def _overview_flatten(category, prefix, value):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _overview_flatten(category, f"{prefix} / {_overview_label(child_key)}", child_value)
    elif isinstance(value, list):
        if not value:
            _overview_input_rows.append([category, prefix, "None"])
        for index, item in enumerate(value, start=1):
            _overview_flatten(category, f"{prefix} {index}", item)
    else:
        shown = "Not provided" if value is None else ("Yes" if value is True else "No" if value is False else str(value))
        _overview_input_rows.append([category, prefix, shown])

for _key, _value in (UI_SUBMISSION_DATA.get("farm") or {}).items():
    _overview_flatten("Farm", _overview_label(_key), _value)
_overview_flatten("Review", "Month", UI_SUBMISSION_DATA.get("month"))
_overview_flatten("Fertilizer history", "Applications per year", (UI_SUBMISSION_DATA.get("fertilizerHistory") or {}).get("freq"))
for _index, _fert in enumerate((UI_SUBMISSION_DATA.get("fertilizerHistory") or {}).get("fertilizers", []), start=1):
    _overview_flatten("Fertilizer history", f"Product {_index}", _fert)
for _key, _value in (UI_SUBMISSION_DATA.get("harvest") or {}).items():
    _overview_flatten("Harvest", _overview_label(_key), _value)
_overview_flatten("Budget", "Annual fertilizer budget (Rp)", UI_SUBMISSION_DATA.get("fertilizerBudgetRp"))
_overview_table("1. User-submitted farm and fertilizer data", ["Category", "Field", "Value"], _overview_input_rows)

# 2) Display the annual base + VARI product quantity per zone for Dolomite and Urea.
_overview_tree_counts = {f"Z{int(z['zone_id'])}": int(z.get("tree_count", 0)) for z in zone_json}
_overview_zone_rows = []
_overview_zone_product_keys = {
    "dolomite_Mg_cheaper": "Dolomite",
    "urea_N_expensive": "Urea",
}
for _row in product_bag_rows:
    _key = _row.get("product")
    _zone_id = _row.get("zone_id")
    if _key not in _overview_zone_product_keys or _zone_id == "FIELD TOTAL":
        continue
    _total_kg = _row.get("base_plus_supplement_product_kg_per_zone_per_year")
    _bag_kg = _row.get("bag_size_kg")
    _exact_bags = _row.get("exact_bag_equivalent_for_zone")
    _zone_buy_bags = (
        int(_overview_math.ceil(float(_exact_bags) - 1e-12))
        if _exact_bags is not None and float(_exact_bags) > 0 else 0 if _exact_bags is not None else None
    )
    _overview_zone_rows.append([
        str(_zone_id), _overview_zone_product_keys[_key], str(_overview_tree_counts.get(str(_zone_id), "—")),
        "N/A" if _total_kg is None else f"{float(_total_kg):,.{_overview_decimals}f}",
        "N/A" if _exact_bags is None else f"{float(_exact_bags):,.{_overview_decimals}f}",
        "N/A" if _zone_buy_bags is None else str(_zone_buy_bags),
    ])
_overview_table(
    "2. Annual zone fertilizer quantities (base + VARI supplement)",
    ["Zone", "Product", "Detected trees", "Total kg/zone/year", "Bag equivalents", "Full bags if purchased per zone"],
    _overview_zone_rows or [["—", "No zone product rows available", "—", "—", "—", "—"]],
)
print("Per-zone full bags are rounded independently for planning; actual field purchasing can differ if product is pooled across zones.")

# Whole-field cost for the selected Mg option plus Urea, using whole-bag estimates.
_overview_selected_zone_product_keys = {_selected_mg_product, "urea_N_expensive"}
_overview_zone_topup_cost_rp = 0.0
_overview_zone_cost_known = True
_overview_zone_topup_labels = []
for _row in product_bag_rows:
    if _row.get("zone_id") == "FIELD TOTAL" and _row.get("product") in _overview_selected_zone_product_keys:
        _cost = _row.get("full_bag_purchase_cost_rp")
        if _cost is None:
            _overview_zone_cost_known = False
        else:
            _overview_zone_topup_cost_rp += float(_cost)
        _overview_zone_topup_labels.append(_overview_zone_product_keys.get(_row.get("product"), _row.get("product")))
if not _overview_zone_topup_labels:
    _overview_zone_cost_known = False
_overview_zone_topup_cost = _overview_zone_topup_cost_rp if _overview_zone_cost_known else None

# 3) Compare annual NPK/NPKMg options with the submitted annual fertilizer budget.
_overview_budget = UI_SUBMISSION_DATA.get("fertilizerBudgetRp")
try:
    _overview_budget = float(_overview_budget) if _overview_budget is not None else None
    if _overview_budget is not None and (not _overview_math.isfinite(_overview_budget) or _overview_budget < 0):
        _overview_budget = None
except (TypeError, ValueError):
    _overview_budget = None

_overview_npk_options = [item for item in _age_adjusted_npk_quantities if item.get("available")]
_overview_candidate_rows = []
_overview_comparable = []
for _option in _overview_npk_options:
    _npk_cost = _option.get("ageAdjustedFullBagCostForFarmRp")
    _zone_cost = _overview_zone_topup_cost
    _combined_cost = float(_npk_cost) + float(_zone_cost) if _npk_cost is not None and _zone_cost is not None else None
    _delta = _combined_cost - _overview_budget if _combined_cost is not None and _overview_budget is not None else None
    _candidate = {
        "option": _option,
        "npk_cost": _npk_cost,
        "zone_cost": _zone_cost,
        "combined_cost": _combined_cost,
        "delta": _delta,
    }
    _overview_comparable.append(_candidate)
    _overview_candidate_rows.append([
        str(_option.get("blendOption")),
        "-".join(str(_option.get("composition", {}).get(k, 0)) for k in ("N", "P2O5", "K2O", "MgO")),
        "N/A" if _option.get("ageAdjustedFullBagsForFarmYear") is None else str(_option.get("ageAdjustedFullBagsForFarmYear")),
        _overview_rp(_npk_cost),
        _overview_rp(_zone_cost),
        _overview_rp(_combined_cost),
        _overview_rp(_overview_budget),
        "N/A" if _delta is None else (f"{_overview_rp(abs(_delta))} under budget" if _delta < 0 else f"{_overview_rp(_delta)} over budget" if _delta > 0 else "Exact match"),
        "",
    ])

_overview_rankable = [item for item in _overview_comparable if item["combined_cost"] is not None]
_overview_closest = min(_overview_rankable, key=lambda item: abs(item["delta"])) if _overview_budget is not None and _overview_rankable else None
for _candidate, _display_row in zip(_overview_comparable, _overview_candidate_rows):
    if _overview_budget is None:
        _display_row[-1] = "Enter budget to rank"
    elif _candidate["combined_cost"] is None:
        _display_row[-1] = "Cost unavailable"
    elif _candidate is _overview_closest:
        _display_row[-1] = "CLOSEST TO BUDGET"
    else:
        _display_row[-1] = ""

overview_headers = ["Blend option", "Label N-P₂O₅-K₂O-MgO (%)", "NPK full bags/farm/year", "NPK cost", "Zone Mg/Urea cost", "Combined estimate", "Budget target", "Budget difference", "Budget fit"]
if _overview_npk_options:
    _overview_table(
        "3. Annual NPK/NPKMg matching — closest option to entered budget",
        overview_headers,
        _overview_candidate_rows,
    )
else:
    print("\n3. Annual NPK/NPKMg recap unavailable: age-adjusted blend quantities are unavailable.")

if _overview_budget is None:
    print("Enter a nonnegative annual Rupiah value in fertilizerBudgetRp to identify which available option is closest to the budget.")
elif not _overview_rankable:
    print("Cannot rank options against the budget because required configured bag prices or zone product costs are unavailable.")
else:
    print(f"Annual fertilizer budget: {_overview_rp(_overview_budget)}")
    print(f"Zone top-up cost included: {_overview_rp(_overview_zone_topup_cost)} "
          f"({', '.join(_overview_zone_topup_labels) if _overview_zone_topup_labels else 'unavailable'}; whole bags).")
    print("Ranking uses the smallest absolute difference between the budget and "
          "age-adjusted NPK whole-bag cost plus selected zone Mg/Urea whole-bag cost.")
    print(f"Closest by estimated cost: {_overview_closest['option'].get('blendOption')} — "
          f"{_overview_rp(_overview_closest['combined_cost'])}, a difference of "
          f"{_overview_rp(abs(_overview_closest['delta']))} from budget. Nutrient matching still matters.")
