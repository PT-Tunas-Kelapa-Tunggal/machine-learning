"""Sun geometry and clear-sky radiation in plain math: the reference for the Swift port.

No numpy, no pvlib, no files. Angles are degrees in and out, times are UTC, and every step maps line by line
onto Foundation's math functions. `tests/solar_cases.json` holds reference values (pvlib's NREL SPA and the
FAO-56 worked example) that this module and the Swift port must both match within the stated tolerances.

Where it is used
- L0 QA / L9 log (iteration 2+): sun position and Haurwitz clear-sky light at scan time.
- L9 irradiance check (iteration 1, log-only): daily extraterrestrial radiation Ra and the FAO-56 clear-sky bound.
- L5 water balance (iteration 2+, hypothesis): Ra for Hargreaves ET0.
"""
from __future__ import annotations

import math

SOLAR_CONSTANT_MJ_MIN = 0.0820          # Gsc, MJ m-2 min-1 (FAO-56 eq. 21)
HAURWITZ_A, HAURWITZ_B = 1098.0, 0.059  # clear-sky GHI = A·cosZ·exp(−B/cosZ), W m-2 (Haurwitz 1945)


def _sin(deg):
    return math.sin(math.radians(deg))


def _cos(deg):
    return math.cos(math.radians(deg))


def _tan(deg):
    return math.tan(math.radians(deg))


def _clamp(x):
    return min(1.0, max(-1.0, x))  # keeps acos/asin defined when rounding lands just outside [-1, 1]


# ============================================================
# SUN POSITION (NOAA solar calculator, after Meeus)
# ============================================================

def julian_day(when_utc):
    """Julian day of a timezone-aware datetime, from its Unix time (Swift: date.timeIntervalSince1970)."""
    if when_utc.tzinfo is None:
        raise ValueError("when_utc must be timezone-aware")
    return when_utc.timestamp() / 86400.0 + 2440587.5


def _refraction_deg(elevation):
    """Atmospheric refraction to add to the true elevation (NOAA's piecewise fit)."""
    if elevation > 85.0:
        return 0.0
    if elevation > 5.0:
        t = _tan(elevation)
        arcsec = 58.1 / t - 0.07 / t ** 3 + 0.000086 / t ** 5
    elif elevation > -0.575:
        e = elevation
        arcsec = 1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    else:
        arcsec = -20.772 / _tan(elevation)
    return arcsec / 3600.0


def solar_position(lat, lon, when_utc):
    """Where the sun is, seen from (lat, lon) at when_utc.

    zenith_deg and elevation_deg include atmospheric refraction (where the sun appears, what a camera sees);
    true_zenith_deg leaves it out. azimuth_deg runs clockwise from north. NOAA quotes about one arc-minute of
    accuracy for 1800–2100 while the sun is clear of the horizon.
    """
    jc = (julian_day(when_utc) - 2451545.0) / 36525.0  # Julian centuries since J2000
    minutes_utc = (when_utc.timestamp() % 86400.0) / 60.0  # Swift: use a remainder that stays positive

    mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    centre = (_sin(mean_anom) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + _sin(2 * mean_anom) * (0.019993 - 0.000101 * jc)
              + _sin(3 * mean_anom) * 0.000289)
    omega = 125.04 - 1934.136 * jc
    app_long = mean_long + centre - 0.00569 - 0.00478 * _sin(omega)
    mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * _cos(omega)
    decl = math.degrees(math.asin(_clamp(_sin(obliq) * _sin(app_long))))

    y = _tan(obliq / 2.0) ** 2
    eq_time_min = 4.0 * math.degrees(
        y * _sin(2 * mean_long) - 2 * ecc * _sin(mean_anom) + 4 * ecc * y * _sin(mean_anom) * _cos(2 * mean_long)
        - 0.5 * y * y * _sin(4 * mean_long) - 1.25 * ecc * ecc * _sin(2 * mean_anom))
    true_solar_min = (minutes_utc + eq_time_min + 4.0 * lon) % 1440.0  # in [0, 1440)
    hour_angle = true_solar_min / 4.0 - 180.0

    cos_zen = _clamp(_sin(lat) * _sin(decl) + _cos(lat) * _cos(decl) * _cos(hour_angle))
    true_zenith = math.degrees(math.acos(cos_zen))
    elevation = 90.0 - true_zenith + _refraction_deg(90.0 - true_zenith)

    denom = _cos(lat) * _sin(true_zenith)
    if abs(denom) < 1e-12:
        azimuth = 180.0  # sun straight overhead (or observer at a pole): azimuth is undefined
    else:
        a = math.degrees(math.acos(_clamp((_sin(lat) * cos_zen - _sin(decl)) / denom)))
        azimuth = (a + 180.0) % 360.0 if hour_angle > 0 else (540.0 - a) % 360.0
    return {
        "zenith_deg": 90.0 - elevation,
        "elevation_deg": elevation,
        "true_zenith_deg": true_zenith,
        "azimuth_deg": azimuth,
    }


# ============================================================
# DAILY RADIATION (FAO-56)
# ============================================================

def day_of_year(day):
    """1 for 1 January … 365 or 366 (FAO-56's J)."""
    return day.timetuple().tm_yday


def extraterrestrial_daily(lat, doy):
    """Ra, MJ m-2 day-1: the day's sunlight on a level surface at the top of the atmosphere (FAO-56 eqs. 21–25)."""
    phi = math.radians(lat)
    b = 2.0 * math.pi * doy / 365.0
    dr = 1.0 + 0.033 * math.cos(b)                 # inverse relative Earth–Sun distance (eq. 23)
    decl = 0.409 * math.sin(b - 1.39)              # solar declination, radians (eq. 24)
    ws = math.acos(_clamp(-math.tan(phi) * math.tan(decl)))  # sunset hour angle (eq. 25); clamped for polar day/night
    return (24.0 * 60.0 / math.pi * SOLAR_CONSTANT_MJ_MIN * dr
            * (ws * math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.sin(ws)))


def clear_sky_fao56(ra, elevation_m=0.0):
    """Rso, MJ m-2 day-1: clear-sky sunlight at the surface from Ra and the site's elevation (FAO-56 eq. 37)."""
    return (0.75 + 2e-5 * elevation_m) * ra


# ============================================================
# CLEAR-SKY LIGHT AT ONE MOMENT
# ============================================================

def haurwitz_ghi(zenith_deg):
    """Clear-sky global horizontal irradiance, W m-2, from the apparent zenith angle; 0 when the sun is down.

    Haurwitz's model has no humidity or aerosol term, so in the humid, often hazy tropics it reads high: treat it
    as an upper reference, not a forecast of the light at a scan.
    """
    cos_zen = _cos(zenith_deg)
    if cos_zen <= 0.0:
        return 0.0
    return HAURWITZ_A * cos_zen * math.exp(-HAURWITZ_B / cos_zen)
