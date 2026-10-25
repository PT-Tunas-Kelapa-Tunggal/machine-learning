"""Daily rain for Uji 3's P30 and water_stress, per plot: each plot's own centroid, placed as the weather job places it.

A plot's centroid goes to job.grid_cell (the NASA POWER rain cell, 0.5° × 0.625°) and job.weatherkit_area (0.1°), as
weather/nasa_power_job.py places every plot. As in the job's run():
- NASA POWER rain is fetched once per cell and checked with job.build_recent (suspect and unchecked months dropped)
  against the cell's 1991–2020 normals and WeatherKit at the cell centre;
- WeatherKit rain for P30 is fetched once per area;
- each plot's sources are {"weatherkit-history": its area, "nasa-power": its cell, checked}, and P30 at a scene
  date comes from weather/l2_scan_rain.p30_at. Nothing here re-implements a rule.

The .env point is never used. Cells and areas are named by their coordinates, so they stay in memory: the job's
logger (which prints lat/lon) is silenced while fetching, and a failed request is re-raised by exception type only.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

WEATHER_DIR = Path(__file__).resolve().parents[1] / "weather"
sys.path.insert(0, str(WEATHER_DIR))
import krops_rules as wrules  # noqa: E402
import nasa_power_job as job  # noqa: E402

import s2_fetch as fetch  # noqa: E402

NASA_FROM = date(wrules.NORMAL_PERIOD[0], 1, 1)  # one request: the normals and the checked history
CHECK_FROM = date(2018, 12, 1)                     # P30 for a scene in Jan 2019 reaches back into Dec 2018
WK_FROM = date(2021, 8, 1)                         # WeatherKit history starts here


class RainFetchFailed(RuntimeError):
    """A rain request that failed. The message is the exception type only: request URLs carry coordinates."""


def centroid(plot):
    """(lat, lon) of the plot's centroid in WGS84."""
    c = plot["geometry"].centroid
    return c.y, c.x


def places(plots):
    """{plot_id: {"cell": job.grid_cell, "area": job.weatherkit_area}} from each plot's centroid."""
    out = {}
    for plot in plots:
        lat, lon = centroid(plot)
        out[plot["plot_id"]] = {"cell": job.grid_cell(lat, lon), "area": job.weatherkit_area(lat, lon)}
    return out


def rain_by_plot(plots, end, fetch_nasa, fetch_wk):
    """({plot_id: sources for l2_scan_rain.p30_at}, counts) for every plot.

    fetch_nasa(lat, lon, start, end) → {date: mm or None}; fetch_wk(lat, lon, start, end) → {date: mm or None},
    or None without WeatherKit. counts = {"nasa_cells", "weatherkit_areas", "weatherkit"}: numbers only.
    """
    where = places(plots)
    cells = {w["cell"]["id"]: w["cell"] for w in where.values()}
    areas = {w["area"]["id"]: w["area"] for w in where.values()}
    checked = {}
    for cell_id, cell in cells.items():
        rain = fetch_nasa(cell["lat"], cell["lon"], NASA_FROM, end)
        normals = wrules.monthly_normals(rain)
        recent = {d: v for d, v in rain.items() if d >= CHECK_FROM}
        wk_check = fetch_wk(cell["lat"], cell["lon"], WK_FROM, end)
        rain_mm = job.build_recent(recent, normals, wk_check)["rain_mm"]
        checked[cell_id] = {date.fromisoformat(d): v for d, v in rain_mm.items()}
    area_rain = {area_id: fetch_wk(a["lat"], a["lon"], WK_FROM, end) for area_id, a in areas.items()}
    sources = {pid: {"weatherkit-history": area_rain[w["area"]["id"]], "nasa-power": checked[w["cell"]["id"]]}
               for pid, w in where.items()}
    return sources, {"nasa_cells": len(cells), "weatherkit_areas": len(areas),
                     "weatherkit": any(r is not None for r in area_rain.values())}


# ============================================================
# NETWORK (cached; AS_OF_DATE refuses to fetch, as for the imagery)
# ============================================================

def _quiet(call):
    logger = logging.getLogger("nasa_power_job")
    was, logger.disabled = logger.disabled, True
    try:
        return call()
    except fetch.NotCached:
        raise
    except Exception as e:  # noqa: BLE001 – requests and WeatherKit errors carry the URL, so only the type goes on
        raise RainFetchFailed(type(e).__name__) from None
    finally:
        logger.disabled = was


def _days(daily):
    return {d.isoformat(): v for d, v in sorted(daily.items())}


def _undays(saved):
    return {date.fromisoformat(d): v for d, v in saved.items()}


def network_fetchers(session, weatherkit):
    """(fetch_nasa, fetch_wk) for rain_by_plot, both cached under s2-feasibility/data/cache/ by a hash of the request."""
    def fetch_nasa(lat, lon, start, end):
        def get():
            values, _, _ = job.fetch_nasa_power(lat, lon, start, end, (job.RAIN,), session)
            return _days(values[job.RAIN])
        return _undays(fetch.cached_json("rain-nasa", fetch._key(lat, lon, start, end), lambda: _quiet(get)))

    def fetch_wk(lat, lon, start, end):
        if weatherkit is None:
            return None
        saved = fetch.cached_json("rain-wk", fetch._key(lat, lon, start, end),
                                  lambda: _quiet(lambda: _days(weatherkit.history(lat, lon, start, end))))
        return _undays(saved)

    return fetch_nasa, fetch_wk


def rain_for(plots, end):
    """rain_by_plot over the network (and the cache): WeatherKit when weather/.env has its keys."""
    import requests

    session = requests.Session()
    weatherkit = job.WeatherKit.from_env(session, fetch.TZ)
    return rain_by_plot(plots, end, *network_fetchers(session, weatherkit))
