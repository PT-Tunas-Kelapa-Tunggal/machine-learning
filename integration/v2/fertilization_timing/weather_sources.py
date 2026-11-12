"""The weather the schedule needs, copied from the team's code so this folder runs on its own (README "Copied from"):

    climate      1991–2020 normals and the Stage A month table per NASA POWER cell (build_climate), cached in
                 <cache_dir>/cells/<cell>/climate.json for a year
    P30          the L5 record on a day (scan_p30: WeatherKit first, checked NASA POWER next)
    forecast     WeatherKit's 10-day daily forecast, for L8 Stage B

The functions are unchanged from weather/nasa_power_job.py, weather/l2_scan_rain.py, weather/l6_vari_confidence.py
and integration/krops_vari/weather.py (only their module prefixes dropped); the job's snapshots, QA, irradiance,
hourly values and L6 flight confidence are left out. Coordinates stay in here: nothing returned holds a location.
"""
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import krops_rules as rules

log = logging.getLogger("fertilization_timing.weather")
TZ = ZoneInfo("Asia/Jakarta")
ENV_PATH = Path(os.getenv("KROPS_WEATHER_ENV") or Path(__file__).resolve().parent / ".env")  # WeatherKit keys

# weather/nasa_power_job.py
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RAIN = "PRECTOTCORR"
CLIMATE_MAX_AGE_DAYS = 365
GRID_LAT, GRID_LON = 0.5, 0.625  # NASA POWER rain grid (MERRA-2)
WK_AREA = 0.1                    # WeatherKit P30 area, about 11 km: WeatherKit resolves about 10 km
RECENT_LOOKBACK_DAYS = 45        # recent rain starts on the 1st of the month holding today − 45: last month is whole


# ============================================================
# weather/nasa_power_job.py
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


# ============================================================
# weather/l2_scan_rain.py and weather/l6_vari_confidence.py
# ============================================================

def p30_at(sources, scan_date, order=rules.P30_SOURCE_ORDER):
    """The L5 P30 record as it stood on scan_date: each source cut to the days before it, then rules.choose_p30.

    The scan day itself is left out, as the job leaves out today: its rain isn't complete when the drone flies.
    """
    before = {name: {d: v for d, v in (daily or {}).items() if d < scan_date} for name, daily in sources.items()}
    return rules.choose_p30(before, scan_date, order=order)


def scan_p30(lat, lon, scan_day, normals_mm, *, weatherkit=None, nasa_fetch=fetch_nasa_power, session=None):
    """The L5 P30 record as it stood on the scan day: WeatherKit for the plot's area first, checked NASA POWER next.

    The decided rule, unchanged (l2_scan_rain.p30_at → rules.choose_p30). NASA POWER goes through the job's monthly
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


# ============================================================
# integration/krops_vari/weather.py
# ============================================================

def load_env():
    """Load the WeatherKit keys from ENV_PATH, as the weather job does; python-dotenv is optional."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_PATH)


class WeatherClient:
    """WeatherKit (if its keys are set) and NASA POWER, with the climate cached on disk and the forecast in memory."""

    def __init__(self, cache_dir, session=None, weatherkit=None, use_weatherkit=True):
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        if weatherkit is None and use_weatherkit:
            load_env()
            weatherkit = WeatherKit.from_env(self.session, TZ)
        self.weatherkit = weatherkit
        self._forecasts = {}

    def climate(self, lat, lon, today):
        cell = grid_cell(lat, lon)
        path = self.cache_dir / "cells" / cell["id"] / "climate.json"
        climate = load_climate(path, today)
        if climate is None:
            first, last = rules.NORMAL_PERIOD
            values, _, _ = fetch_nasa_power(cell["lat"], cell["lon"], datetime(first, 1, 1).date(),
                                            datetime(last, 12, 31).date(), (RAIN,), self.session)
            climate = build_climate(values[RAIN], today)
            write_json(path, climate)
        return climate

    def p30(self, lat, lon, day, normals_mm):
        return scan_p30(lat, lon, day, normals_mm, weatherkit=self.weatherkit, session=self.session)

    def forecast(self, lat, lon, today):
        """{local date: rain mm or None} for WeatherKit's daily forecast; {} without WeatherKit."""
        if self.weatherkit is None:
            return {}
        key = (round(lat, 4), round(lon, 4), today)
        if key not in self._forecasts:
            reply = self.weatherkit.daily(lat, lon)
            days = {}
            for d in reply.get("forecastDaily", {}).get("days", []):
                local = datetime.fromisoformat(d["forecastStart"].replace("Z", "+00:00")).astimezone(TZ).date()
                days[local] = d.get("precipitationAmount")
            self._forecasts = {key: days}  # one day's forecasts are enough to keep
        return self._forecasts[key]
