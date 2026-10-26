"""Command line: run one photo through the pipeline and write the result JSON.

    python -m krops_vari.main --photo ../segmentation/final_flow_ut1/dji_10.jpg --out result.json
    python -m krops_vari.main --photo dji_10.jpg --plot-id P01 --plots plots.json \\
        --flight-start 2026-10-26T02:40:00Z --flight-end 2026-10-26T03:05:00Z --out result.json

Without --plot-id the result is the photo alone (as integration.ipynb). With it, the spreading plan is added from the
weather; with flight times too, the L6 flight confidence. plots.json is a list of {plot_id, lat, lon}, kept on the
server: the result never contains the coordinates.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .config import REPO, Settings
from .pipeline import run_scan

DEFAULT_CACHE = REPO / "integration" / "data" / "weather_cache"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--photo", type=Path, required=True, help="the drone photo (RGB JPEG/PNG)")
    parser.add_argument("--out", type=Path, help="where to write the result JSON (default: print it)")
    parser.add_argument("--settings", type=Path, help="JSON object overriding Settings defaults")
    parser.add_argument("--scan-id", help="an id to carry into the result")
    parser.add_argument("--plot-id", help="the plot in --plots; adds the spreading plan from the weather")
    parser.add_argument("--plots", type=Path, default=REPO / "weather" / "plots.json",
                        help="JSON list of {plot_id, lat, lon} (default weather/plots.json)")
    parser.add_argument("--flight-start", help="take-off, UTC ISO (e.g. 2026-10-26T02:40:00Z); adds L6")
    parser.add_argument("--flight-end", help="landing, UTC ISO")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="climate cache per NASA POWER cell")
    parser.add_argument("--no-weatherkit", action="store_true", help="NASA POWER only (P30 fallback, no forecast)")
    return parser.parse_args(argv)


def find_plot(plots_path, plot_id):
    plots = json.loads(Path(plots_path).read_text())
    plot = next((p for p in plots if p["plot_id"] == plot_id), None)
    if plot is None:
        raise SystemExit(f"plot {plot_id} is not in {plots_path}")
    return plot


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_json(args.settings) if args.settings else Settings()
    if bool(args.flight_start) != bool(args.flight_end):
        raise SystemExit("give both --flight-start and --flight-end, or neither")
    flight = {"flight_start": args.flight_start, "flight_end": args.flight_end} if args.flight_start else None
    plot = weather = None
    if args.plot_id:
        from .weather import WeatherClient

        plot = find_plot(args.plots, args.plot_id)
        weather = WeatherClient(args.cache_dir, use_weatherkit=not args.no_weatherkit)
    result = run_scan(args.photo, settings, scan_id=args.scan_id, plot=plot, flight=flight, weather=weather)
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if args.out:
        args.out.write_text(text)
        plan = result["spreading"]
        print(f"{args.out}: {result['image']['trees']} trees, {len(result['zones'])} zones"
              + (f", L8 {plan['l8']['state']}" if plan else ""), file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
