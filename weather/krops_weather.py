"""Shared plumbing for the Krops weather notebooks: config, API clients, caching, loaders and scoring.

The L5 and L8 rules are explained step by step in notebooks 1 and 2; krops_rules.py is the reference the backend
and the Swift app run.
"""
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import jwt
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

import krops_rules as rules

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")


# ============================================================
# CONFIG
# ============================================================

SITE_NAME = os.getenv("SITE_NAME", "Tanjung Pinang")
LAT = float(os.getenv("LAT", "0.92"))
LON = float(os.getenv("LON", "104.45"))
TZ = ZoneInfo("Asia/Jakarta")
COUNTRY = "ID"

RAW_DIR = HERE / "data" / "raw"
GAUGE_CSV = HERE / "data" / "gauge.csv"
BMKG_FILES = [HERE / "data" / "bmkg.xlsx", HERE / "data" / "bmkg.csv"]

P30_THRESHOLD_MM = 60  # L5: proposal rule
HEAVY_RAIN_MM = 25     # L8 Stage B: proposal rule
SUSPECT_RATIO = 3      # a NASA POWER month this many times both references is not trusted

AS_OF_DATE = os.getenv("AS_OF_DATE")  # rerun as of that day, on the responses saved then; never fetches
TODAY = date.fromisoformat(AS_OF_DATE) if AS_OF_DATE else datetime.now(TZ).date()
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RAW ARCHIVE AND CACHE
# ============================================================

def save_raw(source, kind, payload):
    stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S")
    path = RAW_DIR / f"{source}_{kind}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


def cached(source, kind, fetch, forever=False):
    """Reuse a saved response instead of calling the API again.

    forever=False: only a response saved on TODAY (forecasts, recent rain).
    forever=True: any saved response (past days, which don't change).
    With AS_OF_DATE set, only responses saved on or before that day count, and a missing one raises: an as-of
    run reproduces that day and never asks an API again.
    """
    pattern = f"{source}_{kind}_*.json" if forever else f"{source}_{kind}_{TODAY:%Y%m%d}T*.json"
    hits = sorted(RAW_DIR.glob(pattern))
    if AS_OF_DATE:
        hits = [h for h in hits if h.stem.rsplit("_", 1)[1][:8] <= f"{TODAY:%Y%m%d}"]
        if not hits:
            raise RuntimeError(f"AS_OF_DATE={AS_OF_DATE}: no {source} {kind} response saved by then; not fetching")
    if hits:
        return json.loads(hits[-1].read_text())
    payload = fetch()
    save_raw(source, kind, payload)
    return payload


def _is_past(day):
    return day < TODAY - timedelta(days=30)


# ============================================================
# APPLE WEATHERKIT (REST)
# ============================================================

WK_TEAM_ID = os.getenv("WK_TEAM_ID")
WK_KEY_ID = os.getenv("WK_KEY_ID")
WK_SERVICE_ID = os.getenv("WK_SERVICE_ID")
WK_KEY_PATH = Path(os.getenv("WK_KEY_PATH", "")).expanduser()

WK_READY = all([WK_TEAM_ID, WK_KEY_ID, WK_SERVICE_ID]) and WK_KEY_PATH.is_file()
WK_BASE = "https://weatherkit.apple.com"


def weatherkit_token(ttl_s=3600):
    """ES256 JWT. Header: kid + id (TEAM.SERVICE). Claims: only iss, iat, exp, sub; Apple rejects others."""
    now = int(time.time())
    return jwt.encode(
        {"iss": WK_TEAM_ID, "iat": now, "exp": now + ttl_s, "sub": WK_SERVICE_ID},
        WK_KEY_PATH.read_text(),
        algorithm="ES256",
        headers={"kid": WK_KEY_ID, "id": f"{WK_TEAM_ID}.{WK_SERVICE_ID}"},
    )


def wk_get(path, params=None):
    r = requests.get(
        f"{WK_BASE}{path}",
        params=params,
        headers={"Authorization": f"Bearer {weatherkit_token()}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json() if r.content else {}


def wk_availability():
    return wk_get(f"/api/v1/availability/{LAT}/{LON}", {"country": COUNTRY})


def _utc(day):
    start = datetime.combine(day, datetime.min.time(), TZ)
    return start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_weatherkit_daily(start=None, end=None, forever=None):
    """forecastDaily for the site. No dates = the live forecast; start/end = those days (inclusive).

    forever=True reuses any saved response (to rerun on an earlier day's data); None = only for days over 30 days old.
    """
    params = {"dataSets": "forecastDaily", "timezone": str(TZ), "countryCode": COUNTRY}
    if start:
        params |= {"dailyStart": _utc(start), "dailyEnd": _utc(end + timedelta(days=1))}
    kind = "forecast" if start is None else f"daily_{start:%Y%m%d}_{end:%Y%m%d}"
    if forever is None:
        forever = start is not None and _is_past(end)
    return cached("weatherkit", kind, lambda: wk_get(f"/api/v1/weather/en/{LAT}/{LON}", params), forever=forever)


def tidy_weatherkit_days(payload):
    rows = []
    for d in payload.get("forecastDaily", {}).get("days", []):
        daytime = d.get("daytimeForecast") or {}
        rows.append({
            "date": pd.Timestamp(d["forecastStart"]).tz_convert(TZ).date(),
            "rain_mm": d.get("precipitationAmount"),
            "rain_chance": d.get("precipitationChance"),
            "daytime_rain_mm": daytime.get("precipitationAmount"),
            "daytime_rain_chance": daytime.get("precipitationChance"),
            "tmax_c": d.get("temperatureMax"),
            "tmin_c": d.get("temperatureMin"),
        })
    return pd.DataFrame(rows, columns=["date", "rain_mm", "rain_chance", "daytime_rain_mm", "daytime_rain_chance",
                                       "tmax_c", "tmin_c"])


def fetch_weatherkit_history(start, end, chunk_days=10, forever=None):
    """Past daily values (from 1 Aug 2021), 10 days per request. Model values, not observations.

    A date returned twice is kept once when both rows agree and dropped (missing) when they don't, as in
    rules.daily_from_pairs: a duplicate must never let a 30-day window with a gap look complete.
    """
    parts, day = [], start
    while day <= end:
        stop = min(day + timedelta(days=chunk_days - 1), end)
        parts.append(tidy_weatherkit_days(fetch_weatherkit_daily(day, stop, forever)))
        day = stop + timedelta(days=1)
    df = pd.concat(parts).drop_duplicates()
    df = df[~df["date"].duplicated(keep=False)].set_index("date").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def fetch_weatherkit_hourly(start, end, chunk_days=10, forever=None):
    """Past hourly values for the site, local days start to end (inclusive), 10 days per request (Apple's limit).

    One row per hour (UTC index): cloud_cover, rain_mm, daylight, condition. An hour returned twice is kept once when
    the rows agree and dropped (missing) when they don't, as in fetch_weatherkit_history. Model values.
    """
    parts, day = [], start
    while day <= end:
        stop = min(day + timedelta(days=chunk_days - 1), end)
        params = {"dataSets": "forecastHourly", "timezone": str(TZ),
                  "hourlyStart": _utc(day), "hourlyEnd": _utc(stop + timedelta(days=1))}
        payload = cached("weatherkit", f"hourly_{day:%Y%m%d}_{stop:%Y%m%d}",
                         lambda: wk_get(f"/api/v1/weather/en/{LAT}/{LON}", params),
                         forever=_is_past(stop) if forever is None else forever)
        parts.append(pd.DataFrame([{
            "time": pd.Timestamp(h["forecastStart"]),
            "cloud_cover": h.get("cloudCover"),
            "rain_mm": h.get("precipitationAmount"),
            "daylight": h.get("daylight"),
            "condition": h.get("conditionCode"),
        } for h in payload.get("forecastHourly", {}).get("hours", [])],
            columns=["time", "cloud_cover", "rain_mm", "daylight", "condition"]))
        day = stop + timedelta(days=1)
    df = pd.concat(parts).drop_duplicates()
    return df[~df["time"].duplicated(keep=False)].set_index("time").sort_index()


def load_forecast_snapshots():
    """Every saved WeatherKit forecast, one row per (read, day), with the lead time in days."""
    parts = []
    for path in sorted(RAW_DIR.glob("weatherkit_forecast_*.json")):
        payload = json.loads(path.read_text())
        read_time = payload["forecastDaily"].get("metadata", {}).get("readTime")
        read_date = (
            pd.Timestamp(read_time).tz_convert(TZ).date()
            if read_time
            else datetime.strptime(path.stem.split("_")[-1], "%Y%m%dT%H%M%S").date()
        )
        df = tidy_weatherkit_days(payload)
        df["read_date"] = read_date
        df["lead_days"] = [(d - read_date).days for d in df["date"]]
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ============================================================
# NASA POWER
# ============================================================

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_power(start, end, params=("PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN"), forever=None):
    """Daily values for the site; fill values (−999) become missing. Returns (frame, units, sources).

    forever=True reuses any saved response for the same request; None = only when `end` is over 30 days old.
    """
    def fetch():
        r = requests.get(POWER_URL, params={
            "parameters": ",".join(params),
            "community": "AG",
            "latitude": LAT,
            "longitude": LON,
            "start": f"{start:%Y%m%d}",
            "end": f"{end:%Y%m%d}",
            "format": "JSON",
        }, timeout=180)
        r.raise_for_status()
        return r.json()

    kind = f"daily_{start:%Y%m%d}_{end:%Y%m%d}_{len(params)}p"
    payload = cached("nasa-power", kind, fetch, forever=_is_past(end) if forever is None else forever)
    df = pd.DataFrame(payload["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d").date
    df = df.mask(df == payload["header"]["fill_value"])
    units = {k: v["units"] for k, v in payload["parameters"].items()}
    return df, units, payload["header"]["sources"]


def merra2_cell(lat, lon):
    """Approximate bounds of the 0.5° × 0.625° rain cell holding the point."""
    clat = round(lat / 0.5) * 0.5
    clon = round((lon + 180) / 0.625) * 0.625 - 180
    return clat - 0.25, clat + 0.25, clon - 0.3125, clon + 0.3125


# ============================================================
# OPEN-METEO (off by default: the free API is for non-commercial use only)
# ============================================================

USE_OPEN_METEO = os.getenv("USE_OPEN_METEO", "0") == "1"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_open_meteo_archive(start, end):
    """Daily precipitation_sum from Open-Meteo's historical API (ERA5-based reanalysis), Asia/Jakarta days."""
    def fetch():
        r = requests.get(OPEN_METEO_ARCHIVE, params={
            "latitude": LAT, "longitude": LON,
            "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
            "daily": "precipitation_sum", "timezone": str(TZ),
        }, timeout=120)
        r.raise_for_status()
        return r.json()

    payload = cached("open-meteo", f"archive_{start:%Y%m%d}_{end:%Y%m%d}", fetch, forever=_is_past(end))
    daily = payload["daily"]
    return pd.Series(daily["precipitation_sum"], index=pd.to_datetime(daily["time"]).date, name="rain_mm").dropna()


# ============================================================
# BMKG PUBLIC FORECAST API
# ============================================================

BMKG_API = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
BMKG_ADM4 = os.getenv("BMKG_ADM4", "21.72.01.1001")  # kelurahan Tanjung Pinang Barat, closest to the site


def fetch_bmkg_forecast():
    """3-hourly forecast for one kelurahan (adm4 code). Returns (location, frame)."""
    def fetch():
        r = requests.get(BMKG_API, params={"adm4": BMKG_ADM4}, timeout=60)
        r.raise_for_status()
        return r.json()

    payload = cached("bmkg-api", "forecast", fetch)
    steps = [step for day in payload["data"][0]["cuaca"] for step in day]
    df = pd.DataFrame(steps)[["local_datetime", "tp", "weather_desc_en"]]
    df["local_datetime"] = pd.to_datetime(df["local_datetime"])
    return payload["lokasi"], df


# ============================================================
# REFERENCE RAIN: RAIN GAUGE AND BMKG STATION EXPORTS
# ============================================================

BMKG_STATION = {"name": "Stasiun Meteorologi Raja Haji Fisabilillah", "lat": 0.923, "lon": 104.532}
BMKG_SHIFT_DAYS = int(os.getenv("BMKG_SHIFT_DAYS", "0"))  # -1 if BMKG files each 07:00 reading under the morning it was read
BMKG_NOT_MEASURABLE, BMKG_NO_DATA = 8888, 9999


def load_rain_file(path, shift_days=0):
    """Daily rain from a BMKG Data Online export (Tanggal + RR columns) or a plain date,rain_mm file."""
    if path.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raw = pd.read_csv(path, header=None, dtype=str)
    is_header = raw.apply(lambda row: row.str.strip().str.lower().isin(["tanggal", "date"]).any(), axis=1)
    if not is_header.any():
        raise ValueError(f"{path}: no 'Tanggal' or 'date' column found")
    header = is_header.idxmax()
    df = raw.loc[header + 1:].set_axis(raw.loc[header].str.strip().str.lower(), axis=1)
    date_col = "tanggal" if "tanggal" in df else "date"
    rain_col = "rr" if "rr" in df else "rain_mm"

    dates = pd.to_datetime(df[date_col].str.strip(), format="%d-%m-%Y", errors="coerce")
    dates = dates.fillna(pd.to_datetime(df[date_col].str.strip(), format="ISO8601", errors="coerce"))
    rain = pd.to_numeric(df[rain_col].str.strip().str.replace(",", "."), errors="coerce")
    rain = rain.mask(rain == BMKG_NO_DATA).where(rain != BMKG_NOT_MEASURABLE, 0.0)

    keep = dates.notna() & rain.notna()
    days = (dates[keep] + pd.Timedelta(days=shift_days)).dt.date
    # A date listed twice counts once if the readings agree and is missing if they don't (rules.daily_from_pairs)
    daily = rules.daily_from_pairs(zip(days.values, rain[keep].values))
    return pd.Series({d: v for d, v in daily.items() if v is not None}, name="rain_mm", dtype=float).sort_index()


def load_references():
    """The measured rain available: the trial-site gauge and/or the BMKG station export."""
    refs = {}
    if GAUGE_CSV.is_file():
        refs["gauge (trial site)"] = load_rain_file(GAUGE_CSV)
    bmkg_file = next((p for p in BMKG_FILES if p.is_file()), None)
    if bmkg_file:
        refs["BMKG station"] = load_rain_file(bmkg_file, BMKG_SHIFT_DAYS)
    return refs


def bmkg_distance_km():
    # 1° ≈ 111 km in both directions near the equator
    return 111.32 * ((BMKG_STATION["lat"] - LAT) ** 2 + (BMKG_STATION["lon"] - LON) ** 2) ** 0.5


# ============================================================
# COMPARING SERIES
# ============================================================

def monthly_mm(rain):
    rain = rain.dropna()
    return rain.groupby(pd.PeriodIndex(pd.to_datetime(rain.index), freq="M")).sum()


def monthly_normals(rain, first=1991, last=2020):
    """Median rain per calendar month (1–12) over a climate period, default the 1991–2020 WMO normal."""
    years = pd.to_datetime(rain.index).year
    months = monthly_mm(rain[(years >= first) & (years <= last)])
    return months.groupby(months.index.month).median()


def suspect_months(nasa, normals, wk=None, ratio=SUSPECT_RATIO):
    """Months where NASA POWER is over `ratio` × its normal AND over `ratio` × WeatherKit (where WeatherKit exists)."""
    months = monthly_mm(nasa)
    check = pd.DataFrame({
        "days": nasa.groupby(pd.PeriodIndex(pd.to_datetime(nasa.index), freq="M")).size(),
        "NASA POWER (mm)": months,
        "1991–2020 normal (mm)": [normals[m.month] for m in months.index],
    })
    too_high = check["NASA POWER (mm)"] > ratio * check["1991–2020 normal (mm)"]
    if wk is not None:
        check["WeatherKit (mm)"] = monthly_mm(wk)
        too_high &= ~(check["NASA POWER (mm)"] <= ratio * check["WeatherKit (mm)"])  # no WeatherKit month → the normal alone decides
    check["suspect"] = too_high
    return check.round(0)


def drop_months(rain, months):
    return rain[~pd.PeriodIndex(pd.to_datetime(rain.index), freq="M").isin(months)]


def agreement(a, b, mm):
    """Days both are over mm, only a is, only b is (strictly over, as in the Stage B rule)."""
    a_hit, b_hit = a > mm, b > mm
    return int((a_hit & b_hit).sum()), int((a_hit & ~b_hit).sum()), int((~a_hit & b_hit).sum())


def score(estimate, reference, heavy_mm=HEAVY_RAIN_MM):
    both = pd.concat({"est": estimate, "obs": reference}, axis=1).dropna()
    hits, false_alarms, misses = agreement(both["est"], both["obs"], heavy_mm)
    return {
        "days": len(both),
        "bias_mm": round((both["est"] - both["obs"]).mean(), 2),
        "mae_mm": round((both["est"] - both["obs"]).abs().mean(), 2),
        "heavy_hits": hits,
        "heavy_misses": misses,
        "false_alarms": false_alarms,
    }


def rank_corr(a, b):
    """Spearman ρ as Pearson on ranks (no scipy needed)."""
    both = pd.concat({"a": a, "b": b}, axis=1).dropna()
    return round(both.rank().corr().iloc[0, 1], 2) if len(both) > 2 else None


def date_shift_check(reference, sources, shifts=(-1, 0, 1)):
    """Rank correlation of a reference with each source when its dates are shifted by each amount."""
    out = {}
    for shift in shifts:
        moved = pd.Series(reference.values, index=[d + timedelta(days=shift) for d in reference.index])
        out[f"shift {BMKG_SHIFT_DAYS + shift:+d} d"] = {name: rank_corr(moved, s) for name, s in sources.items()}
    return pd.DataFrame(out)


def score_table(references, sources):
    table = pd.DataFrame({
        (ref_name, source): score(estimate, reference)
        for ref_name, reference in references.items()
        for source, estimate in sources.items()
    }).T
    table.index.names = ["reference", "source"]
    counts = ["days", "heavy_hits", "heavy_misses", "false_alarms"]
    table[counts] = table[counts].astype(int)
    return table[table["days"] > 0]  # sources with no overlapping day are hidden


# ============================================================
# CHART STYLE
# ============================================================

NASA_COLOR, WK_COLOR, THIRD_COLOR = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1–3
INK, MUTED, GRID, REMOVED = "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"


def use_chart_style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def threshold_line(ax, y, label, x=None):
    ax.axhline(y, color=MUTED, linestyle="--", linewidth=1)
    xmin, xmax = ax.get_xlim()
    ax.annotate(label, (xmax if x is None else x, y), xytext=(0, 4), textcoords="offset points",
                ha="right" if x is None else "left", color=MUTED, fontsize=9)
