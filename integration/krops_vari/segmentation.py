"""Stage 1: the leaf (vegetation) mask from Excess Green and Otsu's threshold."""
import cv2
import numpy as np


def segment_vegetation_exg(image_bgr):
    """(mask, segmented): mask is 255 on leaf pixels; segmented is the photo with every other pixel black.

    ExG = 2G − R − B, min-max scaled to 0–255, split by Otsu's threshold, then a 3 × 3 opening and closing. A
    heuristic vegetation mask, not a crown model.
    """
    b, g, r = cv2.split(image_bgr)
    exg = 2.0 * g.astype(np.float32) - r.astype(np.float32) - b.astype(np.float32)
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return mask, segmented
