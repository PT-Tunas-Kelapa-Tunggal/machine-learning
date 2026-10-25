"""L6 input, on the backend after sync: how far a scan's VARI can be trusted as a sign of nutrient status.

VARI = (G − R) / (G + R − B) also moves for reasons that aren't nutrients:
- the flight: a low sun casts long shadows, broken cloud changes the light between photos, wet leaves look darker;
- water: a palm short of water can pale like a palm short of nutrients (L5: P30 under 60 mm).
Each becomes a factor between 0 and 1. The confidence is their product, with a label (tinggi / sedang / rendah) and
the reasons. An input nobody knows never scores 1. The decided L5 rule (P30, WeatherKit first, checked NASA POWER as
the fallback) and the decided 25 mm heavy-rain line are reused unchanged; every other number here is a guess until L9
calibrates it against repeat flights or field checks. The process step by step: VARI_CONFIDENCE.md.

    python l6_vari_confidence.py --scan scan.json --plots plots.json --weather-dir /srv/krops/weather --out result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import krops_rules as rules
import l2_scan_rain as l2
import nasa_power_job as job
import solar

log = logging.getLogger("l6_vari_confidence")

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


# ============================================================
# FACTORS (pure)
# ============================================================

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
    if last_24h_mm is not None and last_24h_mm > rules.HEAVY_RAIN_MM:  # strictly over, as in L8 Stage B
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
        text = (f"P30 {_num(f['p30']['p30_mm'])} mm, di bawah {_num(rules.P30_THRESHOLD_MM)} mm: tajuk bisa pucat karena "
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


# ============================================================
# FLIGHT WEATHER FROM THE SOURCES
# ============================================================

def lowest_sun(lat, lon, start, end):
    """The lowest sun elevation (degrees) at the start, middle and end of the flight."""
    mid = start + (end - start) / 2
    return round(min(solar.solar_position(lat, lon, t)["elevation_deg"] for t in (start, mid, end)), 1)


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
    clear = solar.clear_sky_fao56(solar.extraterrestrial_daily(lat, solar.day_of_year(day)))
    return round(allsky / clear, 3) if clear > 0 else None


def flight_conditions(lat, lon, start, end, *, weatherkit=None, nasa_fetch=job.fetch_nasa_power, session=None, tz=TZ):
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
        cell = job.grid_cell(lat, lon)
        try:
            values, _, _ = nasa_fetch(cell["lat"], cell["lon"], day, day, (job.SUN,), session)
            clearness = nasa_clearness(values.get(job.SUN, {}).get(day), cell["lat"], day)
        except Exception:
            log.warning("NASA POWER unavailable for %s", day, exc_info=True)
            clearness = None
        if clearness is not None:
            cloud = {"cover": None, "class": clearness_class(clearness), "source": "nasa-power", "clearness": clearness}
    return cloud, rain


def scan_p30(lat, lon, scan_day, normals_mm, *, weatherkit=None, nasa_fetch=job.fetch_nasa_power, session=None):
    """The L5 P30 record as it stood on the scan day: WeatherKit for the plot's area first, checked NASA POWER next.

    The decided rule, unchanged (l2_scan_rain.p30_at → rules.choose_p30). NASA POWER goes through the job's monthly
    check first, here against WeatherKit at the plot's area (the job uses the cell centre).
    """
    first = (scan_day - timedelta(days=job.RECENT_LOOKBACK_DAYS)).replace(day=1)
    last = scan_day - timedelta(days=1)
    area, cell = job.weatherkit_area(lat, lon), job.grid_cell(lat, lon)
    wk_rain = None
    if weatherkit:
        try:
            wk_rain = weatherkit.history(area["lat"], area["lon"], first, last)
        except Exception:
            log.warning("WeatherKit history unavailable before %s", scan_day, exc_info=True)
    checked = None
    if nasa_fetch:
        try:
            values, _, _ = nasa_fetch(cell["lat"], cell["lon"], first, last, (job.RAIN,), session)
            recent = job.build_recent(values[job.RAIN], normals_mm, wk_rain)
            checked = {date.fromisoformat(d): v for d, v in recent["rain_mm"].items()}
        except Exception:
            log.warning("NASA POWER unavailable before %s", scan_day, exc_info=True)
    return job._iso_record(l2.p30_at({"weatherkit-history": wk_rain, "nasa-power": checked}, scan_day))


def _utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def confidence_for_scan(scan, plot, normals_mm, *, weatherkit=None, nasa_fetch=job.fetch_nasa_power, session=None,
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


# ============================================================
# CLI
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scan", type=Path, required=True, help="the scan manifest the phone uploaded")
    parser.add_argument("--plots", type=Path, default=job.HERE / "plots.json", help="JSON list of {plot_id, lat, lon}")
    parser.add_argument("--weather-dir", type=Path, help="the job's --out folder, for the cell's climate.json")
    parser.add_argument("--out", type=Path, help="where to write the result (default: <scan>.confidence.json)")
    parser.add_argument("--no-weatherkit", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv

        load_dotenv(job.HERE / ".env")
    except ImportError:
        pass

    scan = json.loads(args.scan.read_text())
    plot = next((p for p in json.loads(args.plots.read_text()) if p["plot_id"] == scan["plot_id"]), None)
    if plot is None:
        log.error("plot %s is not in %s", scan["plot_id"], args.plots)
        return 1
    session = requests.Session()
    weatherkit = None if args.no_weatherkit else job.WeatherKit.from_env(session, TZ)
    cell = job.grid_cell(float(plot["lat"]), float(plot["lon"]))
    today = datetime.now(TZ).date()
    climate = job.load_climate(args.weather_dir / "cells" / cell["id"] / "climate.json", today) if args.weather_dir else None
    if climate is None:
        first, last = rules.NORMAL_PERIOD
        values, _, _ = job.fetch_nasa_power(cell["lat"], cell["lon"], date(first, 1, 1), date(last, 12, 31), (job.RAIN,),
                                            session)
        climate = job.build_climate(values[job.RAIN], today)
    result = confidence_for_scan(scan, plot, climate["normals_mm"], weatherkit=weatherkit,
                                 nasa_fetch=job.fetch_nasa_power, session=session)
    out = args.out or args.scan.with_suffix(".confidence.json")
    job.write_json(out, result)
    log.info("%s: confidence %.2f (%s)%s", scan["scan_id"], result["confidence"], result["label"],
             "" if result["final"] else ", not final: some inputs unknown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
