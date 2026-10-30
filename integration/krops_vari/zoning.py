"""Stages 4–5: trees grouped into zones of up to rows × cols, and each zone's VARI over its members' leaf pixels."""
import numpy as np

from .vari import summarize


def group_trees_into_zones(trees, settings, image_width=None, image_height=None):
    """Zones of nearby trees: rows (by centre height), bands of `zone_grid_rows` rows, then blocks of
    `zone_grid_cols` trees per row in the band. A gap over max_zone_distance (a road, a clearing) splits too."""
    grid_rows, grid_cols = settings.zone_grid_rows, settings.zone_grid_cols
    if grid_rows < 1 or grid_cols < 1:
        raise ValueError("zone_grid_rows and zone_grid_cols must be at least 1.")
    if not trees:
        return []
    heights = [max(1, tree["bbox"][3] - tree["bbox"][1]) for tree in trees]
    median_height = float(np.median(heights))
    row_tolerance = max(1.0, median_height * settings.row_tolerance_multiplier)
    max_zone_distance = max(1.0, median_height * settings.max_zone_distance_multiplier)
    padding = max(0, int(median_height * settings.zone_padding_multiplier))

    # 1. Rows: each tree joins the row whose centre line is nearest, within the row tolerance
    rows = []
    for tree in sorted(trees, key=lambda t: (t["center"][1], t["center"][0])):
        cy = tree["center"][1]
        best_row, best_distance = None, float("inf")
        for row in rows:
            distance = abs(cy - float(np.median([item["center"][1] for item in row])))
            if distance <= row_tolerance and distance < best_distance:
                best_row, best_distance = row, distance
        if best_row is None:
            rows.append([tree])
        else:
            best_row.append(tree)
    rows.sort(key=lambda row: float(np.median([t["center"][1] for t in row])))
    for row in rows:
        row.sort(key=lambda t: t["center"][0])

    # 2. Bands of up to grid_rows consecutive rows
    row_bands, current_band = [], []
    for row in rows:
        if not current_band:
            current_band.append(row)
            continue
        prev_y = float(np.median([t["center"][1] for t in current_band[-1]]))
        curr_y = float(np.median([t["center"][1] for t in row]))
        if len(current_band) >= grid_rows or (curr_y - prev_y) > max_zone_distance:
            row_bands.append(current_band)
            current_band = [row]
        else:
            current_band.append(row)
    if current_band:
        row_bands.append(current_band)

    # 3. In each band: split at wide horizontal gaps, then cut into blocks of (rows in band × grid_cols) trees
    zones = []
    for band_idx, band in enumerate(row_bands, start=1):
        band_trees = sorted((t for row in band for t in row), key=lambda t: t["center"][0])
        clusters, current = [], [band_trees[0]]
        for tree in band_trees[1:]:
            if tree["center"][0] - current[-1]["center"][0] <= max_zone_distance:
                current.append(tree)
            else:
                clusters.append(current)
                current = [tree]
        clusters.append(current)
        for cluster in clusters:
            capacity = len(band) * grid_cols
            for offset in range(0, len(cluster), capacity):
                members = cluster[offset:offset + capacity]
                x1 = min(t["bbox"][0] for t in members) - padding
                y1 = min(t["bbox"][1] for t in members) - padding
                x2 = max(t["bbox"][2] for t in members) + padding
                y2 = max(t["bbox"][3] for t in members) + padding
                if image_width is not None:
                    x1, x2 = max(0, x1), min(image_width, x2)
                if image_height is not None:
                    y1, y2 = max(0, y1), min(image_height, y2)
                zones.append({
                    "zone_id": len(zones) + 1,
                    "approx_row": band_idx,
                    "trees": members,
                    "bbox": [x1, y1, x2, y2],
                    "is_undersized": len(members) < grid_rows * grid_cols,
                })
    return zones


def pixel_bbox_record(bbox, image_width, image_height):
    x1, y1, x2, y2 = map(int, bbox)
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "normalized_xyxy": [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height],
    }


def zone_records(zones, vmap, exg_mask, width, height):
    """One record per zone: its VARI over the leaf pixels inside the union of its members' boxes (an overlap counts
    once), its tree count, pixel box and member trees."""
    records = []
    for zone in zones:
        zx1, zy1, zx2, zy2 = zone["bbox"]
        union = np.zeros((zy2 - zy1, zx2 - zx1), dtype=bool)
        tree_records = []
        for tree in zone["trees"]:
            tx1, ty1, tx2, ty2 = tree["bbox"]
            tree_records.append({"tree_id": tree["tree_id"], "confidence": tree["confidence"],
                                 "bbox": pixel_bbox_record(tree["bbox"], width, height), "vari": tree["vari"]})
            lx1, ly1 = max(0, tx1 - zx1), max(0, ty1 - zy1)
            lx2, ly2 = min(zx2 - zx1, tx2 - zx1), min(zy2 - zy1, ty2 - zy1)
            if lx2 > lx1 and ly2 > ly1:
                union[ly1:ly2, lx1:lx2] = True
        vari_crop = vmap[zy1:zy2, zx1:zx2]
        mask_crop = (exg_mask[zy1:zy2, zx1:zx2] > 0) & union
        records.append({
            "zone_id": zone["zone_id"],
            "approx_row": zone["approx_row"],
            "tree_count": len(tree_records),
            "bbox": pixel_bbox_record(zone["bbox"], width, height),
            "vari": summarize(vari_crop[mask_crop & np.isfinite(vari_crop)]),
            "trees": tree_records,
        })
    return records
