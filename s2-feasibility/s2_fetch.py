"""Sentinel-2 L2A from the Earth Search STAC (element84, collection sentinel-2-l2a), read as COG windows and cached.

Every STAC response and every clipped window is saved under data/cache/ (gitignored) and reused, so a rerun
never asks again. With AS_OF_DATE set, only what was saved on or before that day counts, and anything missing
raises instead of fetching, as in weather/krops_weather.py: an as-of run reproduces that day.

Fallback: Microsoft Planetary Computer serves the same product but needs signed URLs (the planetary-computer
package); it is not wired in. Ask before adding it.

Privacy: cache file names are hashes. A failed read is reported by exception type only, because GDAL messages
carry the tile path.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import shapely.geometry
from rasterio.transform import Affine

import s2_rules as rules

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data" / "cache"
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
ASSETS = {"B05": "rededge1", "B07": "rededge3", "B8A": "nir08", "SCL": "scl"}  # all 20 m
TZ = ZoneInfo("Asia/Jakarta")

AS_OF_DATE = os.getenv("AS_OF_DATE")  # rerun as of that day, on what was cached then; never fetches
TODAY = date.fromisoformat(AS_OF_DATE) if AS_OF_DATE else datetime.now(TZ).date()

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
}
READ_ATTEMPTS = 3


class NotCached(RuntimeError):
    pass


class ReadFailed(RuntimeError):
    """A COG read that failed every attempt. The message is the exception type only, never GDAL's text."""


# ============================================================
# CACHE
# ============================================================

def _key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _cached(kind, key, ext, load, save, fetch):
    hits = sorted(CACHE_DIR.glob(f"{kind}_{key}_*.{ext}"))
    if AS_OF_DATE:
        hits = [h for h in hits if h.stem.rsplit("_", 1)[1] <= f"{TODAY:%Y%m%d}"]
    if hits:
        return load(hits[-1])
    if AS_OF_DATE:
        raise NotCached(f"AS_OF_DATE={AS_OF_DATE}: no {kind} saved by then; not fetching")
    value = fetch()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save(CACHE_DIR / f"{kind}_{key}_{datetime.now(TZ):%Y%m%d}.{ext}", value)
    return value


def cached_json(kind, key, fetch):
    return _cached(kind, key, "json", lambda p: json.loads(p.read_text()),
                   lambda p, v: p.write_text(json.dumps(v)), fetch)


def cached_array(kind, key, fetch):
    return _cached(kind, key, "npy", np.load, np.save, fetch)


# ============================================================
# STAC
# ============================================================

def search_items(geom, start, end):
    """The STAC features intersecting geom (WGS84) between start and end, inclusive."""
    def fetch():
        from pystac_client import Client
        search = Client.open(STAC_URL).search(collections=[COLLECTION], intersects=shapely.geometry.mapping(geom),
                                              datetime=f"{start}/{end}")
        return search.item_collection_as_dict()

    return cached_json("stac", _key(COLLECTION, geom.wkb_hex, start, end), fetch)["features"]


def item_record(feature):
    """The fields the tests use. hrefs and the grid stay in memory and are never written to outputs."""
    p, assets = feature["properties"], feature["assets"]
    hrefs = {band: assets[name]["href"] for band, name in ASSETS.items() if name in assets}
    scl = assets.get("scl", {})
    return {
        "id": feature["id"],
        "date": date.fromisoformat(p["datetime"][:10]),
        "tile": p.get("grid:code"),
        "baseline": p.get("s2:processing_baseline"),
        "boa_offset_applied": p.get("earthsearch:boa_offset_applied"),
        "updated": p.get("updated"),
        "cog": len(hrefs) == len(ASSETS) and all(h.startswith("https://") and h.endswith(".tif")
                                                for h in hrefs.values()),
        "hrefs": hrefs,
        "crs": p.get("proj:code") or (f"EPSG:{p['proj:epsg']}" if p.get("proj:epsg") else None),
        "transform": Affine(*scl["proj:transform"][:6]) if scl.get("proj:transform") else None,
        "shape": tuple(scl["proj:shape"]) if scl.get("proj:shape") else None,
    }


# ============================================================
# COG READS
# ============================================================

def _read(href, window=None, overview=False):
    import rasterio
    last = None
    for attempt in range(READ_ATTEMPTS):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(href) as ds:
                if overview:
                    f = ds.overviews(1)[-1]
                    return ds.read(1, out_shape=(ds.height // f, ds.width // f))
                return ds.read(1, window=window)
        except Exception as e:  # noqa: BLE001 – GDAL raises several types; all are retried, none is echoed
            last = type(e).__name__
            time.sleep(2 * (attempt + 1))
    raise ReadFailed(last)


def read_window(item, band, window):
    """One band of one item over a window of its tile, as stored (uint16 DN or uint8 SCL)."""
    key = _key(item["id"], band, window.col_off, window.row_off, window.width, window.height)
    return cached_array("win", key, lambda: _read(item["hrefs"][band], window))


def dark_evidence(item):
    """rules.dark_share over the item's whole tile, read at its coarsest overview (B5 and SCL)."""
    def fetch():
        b5 = _read(item["hrefs"]["B05"], overview=True)
        scl = _read(item["hrefs"]["SCL"], overview=True)
        n = min(b5.shape[0], scl.shape[0]), min(b5.shape[1], scl.shape[1])
        return {"dark_share": rules.dark_share(b5[:n[0], :n[1]], scl[:n[0], :n[1]])}

    return cached_json("dark", _key(item["id"], "B05-SCL-overview"), fetch)["dark_share"]
