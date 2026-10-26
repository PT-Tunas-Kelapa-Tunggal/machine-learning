"""Command line: run one photo through the pipeline and write the result JSON.

    python -m krops_vari.main --photo ../segmentation/final_flow_ut1/dji_10.jpg --out result.json
"""
import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .pipeline import run_scan


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--photo", type=Path, required=True, help="the drone photo (RGB JPEG/PNG)")
    parser.add_argument("--out", type=Path, help="where to write the result JSON (default: print it)")
    parser.add_argument("--settings", type=Path, help="JSON object overriding Settings defaults")
    parser.add_argument("--scan-id", help="an id to carry into the result")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = Settings.from_json(args.settings) if args.settings else Settings()
    result = run_scan(args.photo, settings, scan_id=args.scan_id)
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text)
        print(f"{args.out}: {result['image']['trees']} trees, {len(result['zones'])} zones", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
