"""The weather the pipeline needs, from the existing backend code in weather/ (not copied):

    climate      1991–2020 normals and the Stage A month table per NASA POWER cell (nasa_power_job.build_climate),
                 cached in <cache_dir>/cells/<cell>/climate.json for a year, like the daily job
    P30          the L5 record on a day (l6_vari_confidence.scan_p30: WeatherKit first, checked NASA POWER next)
    forecast     WeatherKit's 10-day daily forecast, for L8 Stage B
    flight       the L6 flight confidence for a scan (l6_vari_confidence.confidence_for_scan)

Calls per scan: WeatherKit 1 forecast + 5–8 history (P30) + 1 hourly (L6, only with flight times); NASA POWER 1–2,
plus 1 per cell a year for the climate. Coordinates stay in here: nothing returned holds a location, and errors are
reported by type only (a request URL carries the coordinates).
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from .config import REPO


def _resolve_weather_dir() -> Path:
    if env_dir := os.getenv("KROPS_WEATHER_DIR"):
        return Path(env_dir)
    bundled = Path(__file__).resolve().parent.parent / "weather"
    if bundled.exists():
        return bundled
    return REPO / "weather"


WEATHER_DIR = _resolve_weather_dir()
if str(WEATHER_DIR) not in sys.path:
    sys.path.insert(0, str(WEATHER_DIR))

import krops_rules as rules  # noqa: E402  (weather/, on the path above)
import l6_vari_confidence as l6  # noqa: E402
import nasa_power_job as job  # noqa: E402

log = logging.getLogger("krops_vari.weather")
TZ = ZoneInfo("Asia/Jakarta")


def load_env():
    """Load weather/.env (WeatherKit keys), as the weather job does; python-dotenv is optional."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(WEATHER_DIR / ".env")


class WeatherClient:
    """WeatherKit (if its keys are set) and NASA POWER, with the climate cached on disk and the forecast in memory."""

    def __init__(self, cache_dir, session=None, weatherkit=None, use_weatherkit=True):
        self.cache_dir = Path(cache_dir)
        self.session = session or requests.Session()
        if weatherkit is None and use_weatherkit:
            load_env()
            weatherkit = job.WeatherKit.from_env(self.session, TZ)
        self.weatherkit = weatherkit
        self._forecasts = {}

    def climate(self, lat, lon, today):
        cell = job.grid_cell(lat, lon)
        path = self.cache_dir / "cells" / cell["id"] / "climate.json"
        climate = job.load_climate(path, today)
        if climate is None:
            first, last = rules.NORMAL_PERIOD
            values, _, _ = job.fetch_nasa_power(cell["lat"], cell["lon"], datetime(first, 1, 1).date(),
                                                datetime(last, 12, 31).date(), (job.RAIN,), self.session)
            climate = job.build_climate(values[job.RAIN], today)
            job.write_json(path, climate)
        return climate

    def p30(self, lat, lon, day, normals_mm):
        return l6.scan_p30(lat, lon, day, normals_mm, weatherkit=self.weatherkit, session=self.session)

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

    def flight_confidence(self, scan, plot, normals_mm):
        return l6.confidence_for_scan(scan, plot, normals_mm, weatherkit=self.weatherkit, session=self.session, tz=TZ)
