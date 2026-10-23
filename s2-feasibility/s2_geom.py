"""Plot geometry for the Sentinel-2 feasibility tests: plot boundaries, the inward buffer, full-footprint pixel masks.

Coordinates stay in memory. Nothing here prints, logs or returns them in a form that reaches outputs/.
"""
from __future__ import annotations

import json
import math

import numpy as np
import shapely
from pyproj import Transformer
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Point, shape
from shapely.ops import transform as reproject

import s2_rules as rules

WGS84 = "EPSG:4326"
PROXY_ID = "proxy-env-point"


def utm_crs(lon, lat):
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


def project(geom, src, dst):
    return reproject(Transformer.from_crs(src, dst, always_xy=True).transform, geom)


def load_plots(path, lat, lon):
    """Plots from the private GeoJSON (WGS84, RFC 7946), or one proxy plot around the .env point.

    Each plot: plot_id, geometry (WGS84), year_planted, age_years, source, proxy.
    """
    if path.is_file():
        features = json.loads(path.read_text())["features"]
        return [{
            "plot_id": str(f["properties"]["plot_id"]),
            "geometry": shape(f["geometry"]),
            "year_planted": f["properties"].get("year_planted"),
            "age_years": f["properties"].get("age_years"),
            "source": f["properties"].get("source"),
            "proxy": False,
        } for f in features]
    utm = utm_crs(lon, lat)
    centre = project(Point(lon, lat), WGS84, utm)
    circle = project(centre.buffer(rules.PROXY_BUFFER_M, quad_segs=64), utm, WGS84)
    return [{"plot_id": PROXY_ID, "geometry": circle, "year_planted": None, "age_years": None,
             "source": f"proxy: {rules.PROXY_BUFFER_M} m buffer around the .env point (no plots.geojson)",
             "proxy": True}]


def plot_window(geom, transform, pad_px=2):
    """The tile window around a geometry (in the tile's CRS), snapped to the 20 m grid, padded by pad_px."""
    a, c, e, f = transform.a, transform.c, transform.e, transform.f
    minx, miny, maxx, maxy = geom.bounds
    col0, row0 = math.floor((minx - c) / a) - pad_px, math.floor((maxy - f) / e) - pad_px
    col1, row1 = math.ceil((maxx - c) / a) + pad_px, math.ceil((miny - f) / e) + pad_px
    return Window(col0, row0, col1 - col0, row1 - row0)


def window_inside(window, height, width):
    return window.col_off >= 0 and window.row_off >= 0 and \
        window.col_off + window.width <= width and window.row_off + window.height <= height


def full_footprint_mask(geom, transform, shape_hw):
    """True for each pixel whose whole square footprint lies inside geom (touching the edge from inside counts).

    This is the brief's method for Uji 2: not all_touched, and not pixel centres.
    """
    if geom.is_empty:
        return np.zeros(shape_hw, dtype=bool)
    rows, cols = np.indices(shape_hw)
    x0 = transform.c + cols * transform.a
    y0 = transform.f + rows * transform.e
    boxes = shapely.box(x0, y0 + transform.e, x0 + transform.a, y0)
    shapely.prepare(geom)
    return shapely.contains(geom, boxes)


def plot_masks(geom_tile, tile_transform, window, buffer_m=rules.INNER_BUFFER_M):
    """(raw, interior) masks over the window: full-footprint pixels inside the plot, and inside it buffered inward."""
    t = window_transform(window, tile_transform)
    shape_hw = (int(window.height), int(window.width))
    return full_footprint_mask(geom_tile, t, shape_hw), full_footprint_mask(geom_tile.buffer(-buffer_m), t, shape_hw)
