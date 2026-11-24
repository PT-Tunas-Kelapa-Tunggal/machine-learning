"""Command line: run one photo through the Iteration 2 VARI and fertilizer pipeline.

    python -m krops_vari.main --check
    python -m krops_vari.main --image orthomosaic.jpeg \
        --user-request-submission-json '{"farm":{"areaHa":1.0}}' \
        --output-dir outputs/

Only concrete v0.2 model (krops_model_v0.2_yolo26l.mlpackage) is supported; fallback is strictly disabled.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import CONCRETE_MODEL_NAME, resolve_model_path
from .v2_engine import run_v2_scan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Krops VARI Engine (v2) palm detection, zoning, and fertilizer analysis."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Perform engine self-check (verify model and dependencies) and exit.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to the drone photo or orthomosaic (RGB JPEG/PNG/TIFF).",
    )
    parser.add_argument(
        "--user-request-submission-json",
        type=str,
        help="The complete user submission as a single JSON string.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Destination directory for all output files (default: outputs).",
    )

    args = parser.parse_args(argv)
    if not args.check:
        if not args.image or not args.user_request_submission_json:
            parser.error("--image and --user-request-submission-json are required unless --check is specified")
    return args


def check_engine() -> int:
    """Perform runtime self-check: verify model file and critical runtime libraries."""
    try:
        model_path = resolve_model_path()
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if not model_path.exists():
        print(f"Error: Core ML model '{CONCRETE_MODEL_NAME}' not found at {model_path}", file=sys.stderr)
        return 1

    try:
        import coremltools as ct
        import cv2
        import matplotlib
        import numpy as np
        import pandas as pd
        from PIL import Image

        # Load Core ML model to verify compilation and compatibility
        _ = ct.models.MLModel(str(model_path))
    except Exception as err:
        print(f"Error: Self-check failed during model/library load: {err}", file=sys.stderr)
        return 1

    print("VARI engine siap")
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.check:
        return check_engine()

    if not args.image.is_file():
        print(f"Error: Image not found at {args.image}", file=sys.stderr)
        return 1

    try:
        submission_data = json.loads(args.user_request_submission_json)
    except json.JSONDecodeError as err:
        print(f"Error: Invalid JSON string for --user-request-submission-json: {err}", file=sys.stderr)
        return 1

    if not isinstance(submission_data, dict):
        print("Error: --user-request-submission-json must decode to a JSON object", file=sys.stderr)
        return 1

    try:
        result = run_v2_scan(
            image_path=args.image,
            user_submission_data=submission_data,
            output_dir=args.output_dir,
        )
        tree_count = result.get("image", {}).get("trees", 0)
        zone_count = len(result.get("zones", []))
        print(
            f"VARI scan complete: {tree_count} trees, {zone_count} zones -> {args.output_dir.resolve()}",
            file=sys.stderr,
        )
        return 0
    except Exception as err:
        print(f"Error: VARI scan failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
