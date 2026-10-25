"""L5 water balance (iteration 2+, HYPOTHESIS): rain minus reference evapotranspiration (ET0) over 30 days.

P30 < 60 only asks how much it rained. A water balance also asks how much water the air took back out, so a hot,
sunny month with 70 mm can count as dry and a cloudy one with 55 mm may not. Notebook 03 compares the two on
the trial plot's history. No rule uses this module, the L5 rule (`water_stress = p30_mm < 60`) is unchanged,
and nothing here ships to the app.

Pure functions in plain math, ready to port. Constants are FAO-56's (Allen et al. 1998) or Priestley & Taylor's
(1972) unless marked GUESS. Daily rain and ET0 are {date: mm}; a missing day is never 0.
"""
from __future__ import annotations

import math
from datetime import timedelta

import solar

HARGREAVES_COEF = 0.0023      # FAO-56 eq. 52
MJ_TO_MM = 0.408              # MJ m-2 day-1 of energy → mm/day of evaporation, = 1/λ (FAO-56 eq. 20)
LATENT_HEAT_MJ_KG = 2.45      # λ (FAO-56)
ALBEDO = 0.23                 # grass reference surface (FAO-56 eq. 38)
STEFAN_BOLTZMANN = 4.903e-9   # MJ K-4 m-2 day-1 (FAO-56 eq. 39)
PT_ALPHA = 1.26               # Priestley & Taylor (1972), for a wet surface

WB_WINDOW_DAYS = 30           # the same 30 days as P30
WB30_DEFICIT_MM = 0.0         # GUESS – no source yet: flag when 30 days of rain fell short of 30 days of ET0


# ============================================================
# ATMOSPHERE (FAO-56 chapter 3)
# ============================================================

def saturation_vapour_pressure(t_c):
    """e°(T), kPa (FAO-56 eq. 11)."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def vapour_pressure_slope(t_c):
    """Δ, kPa °C-1, slope of the saturation curve at T (FAO-56 eq. 13)."""
    return 4098.0 * saturation_vapour_pressure(t_c) / (t_c + 237.3) ** 2


def psychrometric_constant(elevation_m=0.0):
    """γ, kPa °C-1, from the standard pressure at the site's elevation (FAO-56 eqs. 7–8)."""
    pressure = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    return 0.665e-3 * pressure


# ============================================================
# ET0
# ============================================================

def hargreaves_et0(tmax, tmin, ra):
    """ET0, mm/day, from daily Tmax and Tmin (°C) and Ra (MJ m-2 day-1) (FAO-56 eq. 52).

    Needs only temperature, which is why it's here; it is known to read high in humid climates, where the small
    day–night range understates how cloudy it is.
    """
    tmean = (tmax + tmin) / 2.0
    return HARGREAVES_COEF * (tmean + 17.8) * math.sqrt(max(tmax - tmin, 0.0)) * MJ_TO_MM * ra


def net_radiation(rs, tmax, tmin, ra, elevation_m=0.0, ea=None):
    """Rn, MJ m-2 day-1, from measured sunlight Rs (FAO-56 eqs. 37–40).

    Without humidity data, the actual vapour pressure ea is taken as e°(Tmin), as FAO-56 eq. 48 allows for humid
    sites where the night air reaches its dew point. Rs/Rso is held to 0.3–1.0 (ASCE-EWRI 2005, eq. 18): below about
    0.26 FAO-56's cloud term turns negative and very dark days would gain energy.
    """
    if ea is None:
        ea = saturation_vapour_pressure(tmin)
    rso = solar.clear_sky_fao56(ra, elevation_m)
    relative = min(max(rs / rso, 0.3), 1.0) if rso > 0 else 0.3
    net_short = (1.0 - ALBEDO) * rs
    net_long = (STEFAN_BOLTZMANN * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0
                * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * relative - 0.35))
    return net_short - net_long


def priestley_taylor_et0(rs, tmax, tmin, ra, elevation_m=0.0):
    """ET0, mm/day: α·Δ/(Δ+γ)·Rn/λ, with the day's soil heat flux taken as 0 (FAO-56 eq. 42).

    A day with negative Rn gives 0, not negative evaporation.
    """
    tmean = (tmax + tmin) / 2.0
    delta = vapour_pressure_slope(tmean)
    gamma = psychrometric_constant(elevation_m)
    rn = net_radiation(rs, tmax, tmin, ra, elevation_m)
    return max(PT_ALPHA * delta / (delta + gamma) * rn / LATENT_HEAT_MJ_KG, 0.0)


# ============================================================
# THE 30-DAY BALANCE
# ============================================================

def water_balance(rain, et0, end, days=WB_WINDOW_DAYS):
    """Rain − ET0 summed over the `days` days ending on `end`; None if either is missing on any day."""
    window = [end - timedelta(days=i) for i in range(days - 1, -1, -1)]
    if any(rain.get(d) is None or et0.get(d) is None for d in window):
        return None
    return round(sum(float(rain[d]) - float(et0[d]) for d in window), 1)


def deficit(wb_mm, threshold=WB30_DEFICIT_MM):
    """True when the balance is under the threshold; None (unknown) when there is no balance, never False."""
    return None if wb_mm is None else wb_mm < threshold
