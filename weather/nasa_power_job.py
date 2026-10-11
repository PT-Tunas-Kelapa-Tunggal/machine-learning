"""NASA POWER job: per grid cell, the Stage A climate table (yearly) and checked recent rain (daily), as JSON.

The app downloads its cell's `latest.json` at sync and runs the L5/L8 rules offline from it. One NASA
POWER request per grid cell per run, shared by every plot in that cell, as NASA asks.

What a run does, for each grid cell holding at least one plot:
  1. Climate (when missing, older than a year, or with --refresh-climate): 1991–2020 daily rain → monthly
     normals and the Stage A month table with gates.             → cells/<cell>/climate.json
  2. Recent rain: from the 1st of last month to today → drop months that fail the monthly check
     (against the normals and, when configured, WeatherKit) → checked daily rain and NASA P30.
                                                                  → cells/<cell>/latest.json
  3. Index of plot → cell file.                                   → plots.json
  4. Optional (--archive-forecast): today's WeatherKit forecast for plots marked "archive_forecast",
     kept for scoring forecasts against BMKG data later.         → archive/forecast/<plot>/<date>.json

Usage:
    python nasa_power_job.py --plots plots.json --out /srv/krops/weather
Cron, daily at 06:15 server time (NASA POWER updates its recent days once a day):
    15 6 * * *  cd /path/to/weather && /path/to/python nasa_power_job.py --plots plots.json --out /srv/krops/weather >> job.log 2>&1

Exit code 1 when any cell failed; the other cells are still written. Files are replaced atomically, so the
app never reads a half-written file.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import krops_rules as rules

HERE = Path(__file__).resolve().parent
log = logging.getLogger("nasa_power_job")

SCHEMA_VERSION = 1
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RAIN = "PRECTOTCORR"
SUN = "ALLSKY_SFC_SW_DWN"  # logged only (design doc), no rule reads it
CLIMATE_MAX_AGE_DAYS = 365
GRID_LAT, GRID_LON = 0.5, 0.625  # NASA POWER rain grid (MERRA-2)


# ============================================================
# GRID CELLS
# ============================================================

def _snap(value, step, offset=0.0):
    return math.floor((value + offset) / step + 0.5) * step - offset  # round half up, the same in Swift


def grid_cell(lat, lon):
    """The rain grid cell holding a point: its centre (where we query NASA POWER) and a stable id."""
    clat, clon = _snap(lat, GRID_LAT), _snap(lon, GRID_LON, 180.0)
    cell_id = f"{'n' if clat >= 0 else 's'}{abs(clat):.3f}_{'e' if clon >= 0 else 'w'}{abs(clon):.4f}"
    return {"id": cell_id, "lat": round(clat, 4), "lon": round(clon, 4)}


def group_plots(plots):
    """{cell id: {id, lat, lon, plots: [plot ids]}} so each cell is fetched once."""
    cells = {}
    for plot in plots:
        cell = grid_cell(float(plot["lat"]), float(plot["lon"]))
        cells.setdefault(cell["id"], cell | {"plots": []})["plots"].append(plot["plot_id"])
    return cells


# ============================================================
# SOURCES
# ============================================================

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
    if units.get(RAIN) != "mm/day":
        raise ValueError(f"unexpected rain unit from NASA POWER: {units.get(RAIN)}")
    return values, units, payload["header"].get("sources", [])


class WeatherKit:
    """Minimal WeatherKit REST client: daily values only. Server-side only, the .p8 key never ships in the app."""

    BASE = "https://weatherkit.apple.com"

    def __init__(self, team_id, key_id, service_id, key_path, session, tz):
        self.team_id, self.key_id, self.service_id = team_id, key_id, service_id
        self.key = Path(key_path).expanduser().read_text()
        self.session, self.tz = session, tz

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

    def daily(self, lat, lon, start=None, end=None):
        params = {"dataSets": "forecastDaily", "timezone": str(self.tz)}
        if start:
            params |= {"dailyStart": self._utc(start), "dailyEnd": self._utc(end + timedelta(days=1))}
        r = self.session.get(f"{self.BASE}/api/v1/weather/en/{lat}/{lon}", params=params,
                             headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        r.raise_for_status()
        return r.json()

    def history(self, lat, lon, start, end):
        """{date: mm} from past daily values, 10 days per request (Apple's limit)."""
        rain, day = {}, start
        while day <= end:
            stop = min(day + timedelta(days=9), end)
            for d in self.daily(lat, lon, day, stop).get("forecastDaily", {}).get("days", []):
                local = datetime.fromisoformat(d["forecastStart"].replace("Z", "+00:00")).astimezone(self.tz).date()
                if start <= local <= end and local not in rain:
                    rain[local] = d.get("precipitationAmount")
            day = stop + timedelta(days=1)
        return rain


# ============================================================
# BUILDING THE FILES (pure: no network, no disk)
# ============================================================

def build_climate(rain, today, period=rules.NORMAL_PERIOD):
    """Normals and the Stage A table from daily rain covering the period."""
    normals = rules.monthly_normals(rain, *period)
    table = rules.month_table(rain, *period)
    if len(normals) < 12 or len(table) < 12:
        raise ValueError(f"climate period {period} is missing months: {sorted(set(range(1, 13)) - set(table))}")
    return {
        "computed_on": today.isoformat(),
        "period": list(period),
        "normals_mm": {str(m): round(v, 1) for m, v in normals.items()},
        "stage_a": {str(m): {"avg_mm": round(row["avg_mm"], 1), "in_band_share": round(row["in_band_share"], 3),
                             "gate": rules.month_gate(row["avg_mm"])}
                    for m, row in table.items()},
    }


def build_recent(rain, normals_mm, weatherkit_rain=None):
    """Checked recent rain: suspect months dropped, plus the NASA POWER P30 from what is left."""
    normals = {int(m): v for m, v in normals_mm.items()}
    present = {d: v for d, v in rain.items() if v is not None}
    suspect = rules.suspect_months(present, normals, weatherkit_rain)
    checked = rules.drop_months(present, suspect)
    p30 = rules.compute_p30(checked, "nasa-power")
    return {
        "last_day": max(present).isoformat() if present else None,
        "checked_against": "normals and weatherkit" if weatherkit_rain else "normals",
        "suspect_months": [f"{y}-{m:02d}" for y, m in sorted(suspect)],
        "rain_mm": {d.isoformat(): v for d, v in sorted(checked.items())},
        "p30": None if p30 is None else p30 | {"p30_start_date": p30["p30_start_date"].isoformat(),
                                              "p30_end_date": p30["p30_end_date"].isoformat()},
    }


def build_snapshot(cell, climate, recent, irradiance, units, sources, generated_at):
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at,
        "cell": {"id": cell["id"], "lat": cell["lat"], "lon": cell["lon"], "plots": cell["plots"]},
        "climate": climate,
        "recent_rain": recent,
        "irradiance": {"param": SUN, "unit": units.get(SUN),
                       "days": {d.isoformat(): v for d, v in sorted(irradiance.items()) if v is not None}},
        "nasa_power_sources": sources,
    }


# ============================================================
# RUN
# ============================================================

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_climate(path, today):
    if not path.is_file():
        return None
    climate = json.loads(path.read_text())
    age = (today - date.fromisoformat(climate["computed_on"])).days
    return climate if age <= CLIMATE_MAX_AGE_DAYS else None


def run(plots, out, today, *, session, weatherkit=None, refresh_climate=False, archive_forecast=False,
        fetch=fetch_nasa_power):
    """Process every cell; returns the number of cells that failed."""
    out = Path(out)
    cells = group_plots(plots)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    failed = 0
    for cell in cells.values():
        try:
            climate_path = out / "cells" / cell["id"] / "climate.json"
            climate = None if refresh_climate else load_climate(climate_path, today)
            if climate is None:
                first, last = rules.NORMAL_PERIOD
                values, _, _ = fetch(cell["lat"], cell["lon"], date(first, 1, 1), date(last, 12, 31), (RAIN,), session)
                climate = build_climate(values[RAIN], today)
                write_json(climate_path, climate)
                log.info("%s: climate table rebuilt", cell["id"])

            start = (today - timedelta(days=45)).replace(day=1)  # the whole of last month, so its check is fair
            values, units, sources = fetch(cell["lat"], cell["lon"], start, today, (RAIN, SUN), session)
            present = [d for d, v in values[RAIN].items() if v is not None]
            wk_rain = None
            if weatherkit and present:
                wk_rain = weatherkit.history(cell["lat"], cell["lon"], start, max(present))
            recent = build_recent(values[RAIN], climate["normals_mm"], wk_rain)
            snapshot = build_snapshot(cell, climate, recent, values.get(SUN, {}), units, sources, generated_at)
            write_json(out / "cells" / cell["id"] / "latest.json", snapshot)
            p30 = recent["p30"]
            log.info("%s: %d plot(s), rain to %s, suspect %s, P30 %s", cell["id"], len(cell["plots"]),
                     recent["last_day"], recent["suspect_months"] or "none", p30 and p30["p30_mm"])
        except Exception:
            failed += 1
            log.exception("%s: failed", cell["id"])

    index = {p["plot_id"]: {"cell": grid_cell(float(p["lat"]), float(p["lon"]))["id"]} for p in plots}
    for entry in index.values():
        entry["path"] = f"cells/{entry['cell']}/latest.json"
    write_json(out / "plots.json", {"schema": SCHEMA_VERSION, "generated_at": generated_at, "plots": index})

    if archive_forecast:
        if not weatherkit:
            log.warning("--archive-forecast needs WeatherKit credentials; skipped")
        for plot in (p for p in plots if p.get("archive_forecast") and weatherkit):
            try:
                payload = weatherkit.daily(plot["lat"], plot["lon"])
                write_json(out / "archive" / "forecast" / plot["plot_id"] / f"{today.isoformat()}.json", payload)
            except Exception:
                failed += 1
                log.exception("%s: forecast archive failed", plot["plot_id"])
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plots", type=Path, default=HERE / "plots.json",
                        help="JSON list of {plot_id, lat, lon[, archive_forecast]} (default: plots.json)")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "job", help="output folder")
    parser.add_argument("--today", type=date.fromisoformat, help="run as of this date (YYYY-MM-DD), for testing")
    parser.add_argument("--timezone", default="Asia/Jakarta", help="day boundaries for WeatherKit (default WIB)")
    parser.add_argument("--refresh-climate", action="store_true", help="rebuild the climate tables now")
    parser.add_argument("--no-weatherkit", action="store_true", help="check NASA POWER against the normals only")
    parser.add_argument("--archive-forecast", action="store_true", help="save today's WeatherKit forecast too")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv(HERE / ".env")
    except ImportError:
        pass

    tz = ZoneInfo(args.timezone)
    today = args.today or datetime.now(tz).date()
    plots = json.loads(args.plots.read_text())
    session = requests.Session()
    weatherkit = None if args.no_weatherkit else WeatherKit.from_env(session, tz)
    log.info("%d plot(s), WeatherKit check %s", len(plots), "on" if weatherkit else "off")
    failed = run(plots, args.out, today, session=session, weatherkit=weatherkit,
                 refresh_climate=args.refresh_climate, archive_forecast=args.archive_forecast)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
