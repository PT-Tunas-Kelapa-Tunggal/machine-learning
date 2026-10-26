"""Writes solar_cases.json: reference values for solar.py and its Swift port. Needs pvlib (tests only).

    cd weather && python tests/make_solar_cases.py

Sun positions come from pvlib's NREL SPA (method "nrel_numpy"), a far more exact algorithm than NOAA's, so a
small tolerance remains. Ra comes from the FAO-56 worked example and from pvlib's extraterrestrial irradiance
integrated over the day; FAO-56's simple declination formula keeps Ra within about 1% of that in the tropics.
"""
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

OUT = Path(__file__).with_name("solar_cases.json")


def spa(lat, lon, when):
    p = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([when]), lat, lon, method="nrel_numpy").iloc[0]
    return {"zenith_deg": float(p["apparent_zenith"]), "elevation_deg": float(p["apparent_elevation"]),
            "true_zenith_deg": float(p["zenith"]), "azimuth_deg": float(p["azimuth"])}


def position_case(name, lat, lon, when):
    expected = spa(lat, lon, when)
    tol = {"true_zenith_deg": 0.02}
    elevation = expected["elevation_deg"]
    if elevation > 5:
        tol |= {"zenith_deg": 0.02, "elevation_deg": 0.02}
    elif elevation > -1:
        tol |= {"zenith_deg": 0.5, "elevation_deg": 0.5}  # refraction models differ most at the horizon
    if expected["true_zenith_deg"] > 15:
        tol["azimuth_deg"] = 0.05  # near the zenith, azimuth swings with tiny position differences
    return {"function": "solar_position", "name": name,
            "input": {"lat": lat, "lon": lon, "utc": when.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "expected": {k: round(v, 6) for k, v in expected.items() if k in tol}, "tolerance": tol,
            "source": "pvlib NREL SPA"}


def integrated_ra(lat, doy, year=2026):
    start = pd.Timestamp(date(year, 1, 1) + timedelta(days=doy - 1), tz="UTC")
    times = pd.date_range(start, start + pd.Timedelta(days=1), freq="1min", inclusive="left")
    zenith = pvlib.solarposition.get_solarposition(times, lat, 0.0, method="nrel_numpy")["zenith"]
    i0 = pvlib.irradiance.get_extra_radiation(times)
    return float((i0 * np.clip(np.cos(np.radians(zenith)), 0, None)).sum() * 60 / 1e6)  # W m-2 × s → MJ m-2


def ra_case(name, lat, doy):
    ra = integrated_ra(lat, doy)
    return {"function": "extraterrestrial_daily", "name": name, "input": {"lat": lat, "doy": doy},
            "expected": round(ra, 4), "tolerance": round(0.01 * ra, 4),
            "source": "pvlib extraterrestrial irradiance integrated over the day (1-min steps), ±1%"}


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def main():
    cases = [
        {"function": "julian_day", "name": "julian day: J2000 epoch", "input": {"utc": "2000-01-01T12:00:00Z"},
         "expected": 2451545.0, "tolerance": 1e-9, "source": "definition"},
        {"function": "julian_day", "name": "julian day: Unix epoch", "input": {"utc": "1970-01-01T00:00:00Z"},
         "expected": 2440587.5, "tolerance": 1e-9, "source": "definition"},
        {"function": "day_of_year", "name": "day of year: 3 Sep (FAO-56 Example 8)", "input": {"date": "2026-09-03"},
         "expected": 246, "tolerance": 0, "source": "FAO-56 Example 8"},
        {"function": "day_of_year", "name": "day of year: 31 Dec of a leap year", "input": {"date": "2024-12-31"},
         "expected": 366, "tolerance": 0, "source": "calendar"},

        position_case("sun: trial site, 10:00 WIB (a morning scan)", 0.92, 104.45, utc(2026, 10, 11, 3, 0)),
        position_case("sun: trial site, near noon", 0.92, 104.45, utc(2026, 10, 11, 5, 0)),
        position_case("sun: trial site, 15:30 WIB", 0.92, 104.45, utc(2026, 10, 11, 8, 30)),
        position_case("sun: trial site, just after sunrise", 0.92, 104.45, utc(2026, 3, 20, 23, 20)),
        position_case("sun: trial site, night", 0.92, 104.45, utc(2026, 10, 11, 15, 0)),
        position_case("sun: Jakarta, June solstice", -6.2, 106.8, utc(2026, 6, 21, 4, 0)),
        position_case("sun: Dumai, December solstice", 1.67, 101.45, utc(2026, 12, 21, 6, 30)),
        position_case("sun: southern mid-latitude", -33.9, 151.2, utc(2025, 1, 15, 2, 0)),
        position_case("sun: west longitude, afternoon", 40.0, -105.0, utc(2026, 7, 4, 21, 0)),
        position_case("sun: J2000 epoch at 0°, 0°", 0.0, 0.0, utc(2000, 1, 1, 12, 0)),

        {"function": "extraterrestrial_daily", "name": "Ra: FAO-56 Example 8 (20°S, 3 Sep)",
         "input": {"lat": -20.0, "doy": 246}, "expected": 32.2, "tolerance": 0.05, "source": "FAO-56 Example 8"},
        ra_case("Ra: trial site, 1 Jan", 0.92, 1),
        ra_case("Ra: trial site, March equinox", 0.92, 80),
        ra_case("Ra: trial site, June solstice", 0.92, 172),
        ra_case("Ra: trial site, 3 Sep", 0.92, 246),
        ra_case("Ra: 6.2°S, December solstice", -6.2, 355),

        {"function": "clear_sky_fao56", "name": "Rso: sea level is 0.75 Ra", "input": {"ra": 36.0, "elevation_m": 0.0},
         "expected": 27.0, "tolerance": 1e-9, "source": "FAO-56 eq. 37"},
        {"function": "clear_sky_fao56", "name": "Rso: 500 m adds 1% of Ra", "input": {"ra": 36.0, "elevation_m": 500.0},
         "expected": 27.36, "tolerance": 1e-9, "source": "FAO-56 eq. 37"},
    ]
    for z in (0.0, 30.0, 60.0, 85.0, 89.5, 90.0, 120.0):
        ghi = float(pvlib.clearsky.haurwitz(pd.Series([z])).iloc[0, 0])
        cases.append({"function": "haurwitz_ghi", "name": f"haurwitz: zenith {z:g}°", "input": {"zenith_deg": z},
                      "expected": round(ghi, 6), "tolerance": 1e-5, "source": "pvlib clearsky.haurwitz"})

    OUT.write_text(json.dumps({
        "version": 1,
        "about": "Reference values for weather/solar.py. The Swift port runs the same cases.",
        "encoding": {"utc": "ISO 8601, UTC", "date": "YYYY-MM-DD", "angles": "degrees",
                     "tolerance": "absolute, in the unit of the value; per key for solar_position",
                     "solar_position": "only the keys listed under tolerance are compared"},
        "generated_with": f"pvlib {pvlib.__version__}",
        "cases": cases,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
