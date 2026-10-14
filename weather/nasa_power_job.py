"""NASA POWER job: per grid cell, the Stage A climate table (yearly), checked recent rain and each plot's P30 (daily).

The app downloads its cell's `latest.json` at sync and runs the L5/L8 rules offline from it. One NASA
POWER request per grid cell per run, shared by every plot in that cell, as NASA asks. Despite the name,
the job also fetches the WeatherKit rain that P30 now uses first.

What a run does, for each grid cell holding at least one plot:
  1. Climate (when missing, older than a year, or with --refresh-climate): 1991–2020 daily rain → monthly
     normals and the Stage A month table with gates.             → cells/<cell>/climate.json
  2. Recent NASA POWER rain: from the 1st of last month to today → drop months that fail the monthly check
     (against the normals and, when configured, WeatherKit; pro-rated while a month fills in) and months with
     too few days to judge → checked daily rain and NASA P30.
  3. P30 per plot (L5 data contract): WeatherKit history first, checked NASA POWER as the fallback. A 30-day window
     with a missing day is never summed as 0 mm or patched from the other source; the record counts the missing
     days per source (p30_missing_days).
     WeatherKit resolves about 10 km, so it is fetched per ~11 km plot area (0.1° box), not per NASA cell. A reply
     that comes back short is re-requested once per gap, with the gap mid-range (WeatherKit.history).
     Quota (weatherkit_calls_per_month): 4 requests per area and 5–8 per cell (the monthly check) a day, plus
     retries (~0.5%) and archived forecasts. 1,000 areas: about 126–129k a month if they share 34 cells, 136–145k
     in 100 cells, 271–362k if every area is its own cell, of the 500k included.
                                                                  → cells/<cell>/latest.json
  4. Backend-only QA log (the app never downloads it, so phones on a weak signal don't pay for it):
     `irradiance_check`, NASA POWER's daily sunlight checked against ALLSKY ≤ CLRSKY ≤ Ra over the last
     QA_LOOKBACK_DAYS (CLRSKY runs months behind, so the same single request reaches further back; rain and P30
     still start where they did), and `temperature` (T2M, T2M_MAX, T2M_MIN) over the same days as `irradiance`,
     for a future L5 water balance. Replaced every run, like latest.json: NASA POWER's own archive is the
     history.                                                     → cells/<cell>/qa_log.json
  5. Index of plot → cell file and WeatherKit area.               → plots.json
  6. Optional (--archive-forecast): today's WeatherKit forecast for plots marked "archive_forecast",
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
import l9_irradiance_check as irradiance

HERE = Path(__file__).resolve().parent
log = logging.getLogger("nasa_power_job")

SCHEMA_VERSION = 2
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RAIN = "PRECTOTCORR"
SUN = "ALLSKY_SFC_SW_DWN"  # logged only (design doc), no rule reads it
SUN_CLEAR = "CLRSKY_SFC_SW_DWN"  # irradiance check only (L9, log-only)
TEMPS = ("T2M", "T2M_MAX", "T2M_MIN")  # stored only, for a future L5 water balance; no rule reads them
RECENT_PARAMS = (RAIN, SUN, SUN_CLEAR, *TEMPS)  # one daily request per cell carries them all
QA_LOG_SCHEMA = 1  # cells/<cell>/qa_log.json, backend only
CLIMATE_MAX_AGE_DAYS = 365
GRID_LAT, GRID_LON = 0.5, 0.625  # NASA POWER rain grid (MERRA-2)
WK_AREA = 0.1                    # WeatherKit P30 area, about 11 km: WeatherKit resolves about 10 km
WK_P30_DAYS = 37                 # 30-day window plus the 7 days P30 may lag behind (rules.P30_MAX_AGE_DAYS)
QA_LOOKBACK_DAYS = 120  # GUESS – no source yet: CLRSKY ran ~73 days behind on 11 Oct 2026; 120 leaves ~45 days to check
RECENT_LOOKBACK_DAYS = 45        # recent rain starts on the 1st of the month holding today − 45: last month is whole
NASA_LAG_DAYS = 3                # NASA POWER rain runs this far behind (observed, notebook 1); for the quota estimate only
SHORT_REPLY_RATE = 3 / 592       # WeatherKit replies seen short so far (Oct 2026); each gap costs one retry


# ============================================================
# GRID CELLS
# ============================================================

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


def group_plots(plots):
    """{cell id: {id, lat, lon, plots: [plot ids]}} so each cell is fetched once."""
    cells = {}
    for plot in plots:
        cell = grid_cell(float(plot["lat"]), float(plot["lon"]))
        cells.setdefault(cell["id"], cell | {"plots": []})["plots"].append(plot["plot_id"])
    return cells


def _spans(days):
    """Runs of consecutive dates: [d1, d2, d3, d7] → [(d1, d3), (d7, d7)]."""
    spans = []
    for d in sorted(days):
        if spans and d == spans[-1][1] + timedelta(days=1):
            spans[-1] = (spans[-1][0], d)
        else:
            spans.append((d, d))
    return spans


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

    def daily(self, lat, lon, start=None, end=None):
        params = {"dataSets": "forecastDaily", "timezone": str(self.tz)}
        if start:
            params |= {"dailyStart": self._utc(start), "dailyEnd": self._utc(end + timedelta(days=1))}
        r = self.session.get(f"{self.BASE}/api/v1/weather/en/{lat}/{lon}", params=params,
                             headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        self.calls += 1
        r.raise_for_status()
        return r.json()

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
        goes through rules.daily_from_pairs: a date given twice counts once if the values agree, and as missing if not.
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
        return rules.daily_from_pairs(pairs)


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


def _iso_record(record):
    """A P30 record with its dates as ISO strings (None stays None)."""
    if record is None:
        return None
    return record | {k: (record[k].isoformat() if record[k] else None) for k in ("p30_start_date", "p30_end_date")}


def build_recent(rain, normals_mm, weatherkit_rain=None):
    """Checked recent NASA POWER rain: suspect and unchecked months dropped, plus the NASA-only P30 from what is left.

    An unchecked month (fewer than rules.SUSPECT_MIN_DAYS days) can't serve as the fallback until it can be judged.
    """
    normals = {int(m): v for m, v in normals_mm.items()}
    present = {d: v for d, v in rain.items() if v is not None}
    suspect = rules.suspect_months(present, normals, weatherkit_rain)
    unchecked = rules.unchecked_months(present) - suspect
    checked = rules.drop_months(present, suspect | unchecked)
    return {
        "last_day": max(present).isoformat() if present else None,
        "checked_against": "normals and weatherkit" if weatherkit_rain else "normals",
        "suspect_months": [f"{y}-{m:02d}" for y, m in sorted(suspect)],
        "unchecked_months": [f"{y}-{m:02d}" for y, m in sorted(unchecked)],
        "rain_mm": {d.isoformat(): v for d, v in sorted(checked.items())},
        "nasa_p30": _iso_record(rules.compute_p30(checked, "nasa-power")),
    }


def build_l5(plots, area_rain, recent, today):
    """The L5 data contract per plot: WeatherKit history for its area first, checked NASA POWER as the fallback.

    `plots` = [{plot_id, area}], `area_rain` = {area id: {date: mm} or None}, `recent` = build_recent's output.
    """
    nasa_checked = {date.fromisoformat(d): v for d, v in recent["rain_mm"].items()}
    out = {}
    for plot in plots:
        sources = {"weatherkit-history": area_rain.get(plot["area"]), "nasa-power": nasa_checked}
        out[plot["plot_id"]] = _iso_record(rules.choose_p30(sources, today)) | {"weatherkit_area": plot["area"]}
    return out


def build_temperature(values, units):
    """Daily temperature for the stored window: {date: {T2M, T2M_MAX, T2M_MIN}}, a missing value as None."""
    days = sorted({d for p in TEMPS for d, v in values.get(p, {}).items() if v is not None})
    return {
        "params": list(TEMPS),
        "units": {p: units.get(p) for p in TEMPS},
        "days": {d.isoformat(): {p: values.get(p, {}).get(d) for p in TEMPS} for d in days},
    }


def build_snapshot(cell, climate, recent, l5, areas, sun, units, sources, generated_at):
    """latest.json, schema 2: what the app downloads at sync."""
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": generated_at,
        "cell": {"id": cell["id"], "lat": cell["lat"], "lon": cell["lon"], "plots": cell["plots"]},
        "climate": climate,
        "l5": l5,
        "weatherkit_rain": areas,
        "recent_rain": recent,
        "irradiance": {"param": SUN, "unit": units.get(SUN),
                       "days": {d.isoformat(): v for d, v in sorted(sun.items()) if v is not None}},
        "nasa_power_sources": sources,
    }


def build_qa_log(cell, irradiance_check, temperature, generated_at):
    """qa_log.json: the cell's log-only data. No rule reads it and the app never downloads it."""
    return {
        "schema": QA_LOG_SCHEMA,
        "generated_at": generated_at,
        "cell": {"id": cell["id"], "lat": cell["lat"], "lon": cell["lon"]},
        "irradiance_check": irradiance_check,  # L9, log-only
        "temperature": temperature,            # stored only, for a future L5 water balance
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


def check_cell_irradiance(values, cell):
    """The log-only irradiance check for one cell, or None if it fails: it must never cost the cell its rain data."""
    try:
        # Elevation is left at 0 m: the FAO-56 term moves the bound by under 1% below 500 m
        return irradiance.check_irradiance(values.get(SUN), values.get(SUN_CLEAR), cell["lat"])
    except Exception:
        log.warning("%s: irradiance check failed; written as null", cell["id"], exc_info=True)
        return None


def weatherkit_calls_per_month(areas, cells, archived_plots=0, days=30):
    """(low, high) WeatherKit requests per month for the job, everything included.

    Per day: P30 history per area (WK_P30_DAYS, 10 days a request); the monthly check per cell, from the 1st of the
    month holding today − RECENT_LOOKBACK_DAYS to NASA POWER's last day (43–73 days, so 5–8 requests); one forecast
    per archived plot; and retries for short replies at the rate seen so far.
    """
    per_area = math.ceil(WK_P30_DAYS / WeatherKit.MAX_DAYS)
    shortest, longest = RECENT_LOOKBACK_DAYS - NASA_LAG_DAYS + 1, RECENT_LOOKBACK_DAYS - NASA_LAG_DAYS + 31
    per_cell = (math.ceil(shortest / WeatherKit.MAX_DAYS), math.ceil(longest / WeatherKit.MAX_DAYS))
    return tuple(round(days * (areas * per_area + cells * c + archived_plots) * (1 + SHORT_REPLY_RATE)) for c in per_cell)


def fetch_area_rain(weatherkit, area, today):
    """WeatherKit daily rain for one area over the last WK_P30_DAYS, ending yesterday; None if unavailable."""
    if weatherkit is None:
        return None
    try:
        return weatherkit.history(area["lat"], area["lon"], today - timedelta(days=WK_P30_DAYS), today - timedelta(days=1))
    except Exception:
        log.warning("%s: WeatherKit history failed; P30 falls back to NASA POWER", area["id"], exc_info=True)
        return None


def run(plots, out, today, *, session, weatherkit=None, refresh_climate=False, archive_forecast=False,
        fetch=fetch_nasa_power):
    """Process every cell; returns the number of cells that failed."""
    out = Path(out)
    cells = group_plots(plots)
    plot_area = {p["plot_id"]: weatherkit_area(float(p["lat"]), float(p["lon"])) for p in plots}
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    area_rain = {}  # area id → {date: mm} or None; shared across cells, fetched once per run
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

            start = (today - timedelta(days=RECENT_LOOKBACK_DAYS)).replace(day=1)  # all of last month: a fair check
            qa_start = min(start, today - timedelta(days=QA_LOOKBACK_DAYS))  # one request, reaching back for CLRSKY
            fetched, units, sources = fetch(cell["lat"], cell["lon"], qa_start, today, RECENT_PARAMS, session)
            values = {p: {d: v for d, v in series.items() if d >= start} for p, series in fetched.items()}  # as before
            present = [d for d, v in values[RAIN].items() if v is not None]
            wk_rain = None
            if weatherkit and present:
                try:
                    wk_rain = weatherkit.history(cell["lat"], cell["lon"], start, max(present))
                except Exception:
                    log.warning("%s: WeatherKit unavailable; checking NASA POWER against the normals only",
                                cell["id"], exc_info=True)
            recent = build_recent(values[RAIN], climate["normals_mm"], wk_rain)

            areas = {}
            for plot_id in cell["plots"]:
                area = plot_area[plot_id]
                if area["id"] not in area_rain:
                    area_rain[area["id"]] = fetch_area_rain(weatherkit, area, today)
                if area_rain[area["id"]] is not None:
                    areas[area["id"]] = {"lat": area["lat"], "lon": area["lon"],
                                         "rain_mm": {d.isoformat(): v for d, v in sorted(area_rain[area["id"]].items())}}
            l5 = build_l5([{"plot_id": p, "area": plot_area[p]["id"]} for p in cell["plots"]], area_rain, recent, today)

            snapshot = build_snapshot(cell, climate, recent, l5, areas, values.get(SUN, {}), units, sources, generated_at)
            qa_log = build_qa_log(cell, check_cell_irradiance(fetched, cell), build_temperature(values, units), generated_at)
            write_json(out / "cells" / cell["id"] / "latest.json", snapshot)
            write_json(out / "cells" / cell["id"] / "qa_log.json", qa_log)
            check = qa_log["irradiance_check"]
            log.info("%s: %d plot(s), NASA rain to %s, suspect %s; P30 %s; irradiance flags %s", cell["id"],
                     len(cell["plots"]), recent["last_day"], recent["suspect_months"] or "none",
                     ", ".join(f"{p} {r['p30_mm']} ({r['p30_source']}, {r['p30_quality']})" for p, r in l5.items()),
                     "n/a" if check is None else check["days_flagged"])
        except Exception:
            failed += 1
            log.exception("%s: failed", cell["id"])

    index = {p["plot_id"]: {"cell": grid_cell(float(p["lat"]), float(p["lon"]))["id"],
                            "weatherkit_area": plot_area[p["plot_id"]]["id"]} for p in plots}
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
    if weatherkit:
        log.info("WeatherKit requests this run: %d (%d retries for short replies)", weatherkit.calls, weatherkit.retries)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
