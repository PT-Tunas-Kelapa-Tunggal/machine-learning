"""Stage 3: VARI per pixel, and VARI statistics over the leaf pixels inside a box."""
import cv2
import numpy as np


def vari_map(image_bgr):
    """VARI = (G − R) / (G + R − B) per pixel of the original photo; NaN where the denominator is about 0."""
    b, g, r = (c.astype(np.float32) for c in cv2.split(image_bgr))
    denominator = g + r - b
    out = np.full(denominator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    np.divide(g - r, denominator, out=out, where=valid)
    return out


def summarize(values):
    """VARI statistics of the given leaf-pixel values; the leaf pixel count is the zone confidence's input."""
    if values.size == 0:
        return {"vegetation_pixel_count": 0, "mean_vari": None, "median_vari": None, "std_vari": None,
                "mad_vari": None, "min_vari": None, "max_vari": None}
    median = float(np.median(values))
    return {
        "vegetation_pixel_count": int(values.size),
        "mean_vari": float(np.mean(values)),
        "median_vari": median,
        "std_vari": float(np.std(values)),
        "mad_vari": float(np.median(np.abs(values - median))),
        "min_vari": float(np.min(values)),
        "max_vari": float(np.max(values)),
    }


def stats_for_box(vmap, exg_mask, bbox):
    """VARI statistics over the leaf pixels inside a pixel box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return summarize(np.asarray([], dtype=np.float32))
    vari_crop = vmap[y1:y2, x1:x2]
    mask_crop = exg_mask[y1:y2, x1:x2] > 0
    return summarize(vari_crop[mask_crop & np.isfinite(vari_crop)])
