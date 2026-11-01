"""Stage 2: palm detection with the Core ML YOLO model, then the filter for trees cut off by the photo edge.

Core ML runs on macOS only. Everything model-specific stays in this module behind `Detector.detect`, so moving to
another runtime (an ONNX export on Linux) changes this file alone.
"""
import threading

import numpy as np
from PIL import Image


def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_threshold=0.45):
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    selected = []
    while len(order) > 0:
        current = int(order[0])
        selected.append(current)
        order = np.asarray([
            int(j) for j in order[1:]
            if iou(boxes[current], boxes[int(j)]) <= iou_threshold
        ], dtype=int)
    return selected


def parse_predictions(raw, settings):
    """Normalized boxes and scores from the model's [1, 5, N] / [1, N, 5] output: rows of [cx, cy, w, h, score] in
    model-input pixels. Boxes under the score threshold are dropped, then duplicates by NMS."""
    raw = np.squeeze(np.asarray(raw))
    if raw.ndim != 2:
        raise ValueError(f"Unexpected model output shape {raw.shape}; expected a 2D array after squeeze.")
    if raw.shape[0] == 5:
        rows = raw.T
    elif raw.shape[1] == 5:
        rows = raw
    else:
        raise ValueError(f"Model output shape {raw.shape} does not have a 5-value detection axis.")
    size = settings.model_input_size
    boxes, scores = [], []
    for row in rows:
        cx, cy, width, height, confidence = map(float, row[:5])
        if not np.isfinite([cx, cy, width, height, confidence]).all() or confidence < settings.conf_threshold:
            continue
        box = [
            np.clip((cx - width / 2) / size, 0, 1),
            np.clip((cy - height / 2) / size, 0, 1),
            np.clip((cx + width / 2) / size, 0, 1),
            np.clip((cy + height / 2) / size, 0, 1),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append([float(v) for v in box])
            scores.append(confidence)
    keep = nms(boxes, scores, settings.iou_threshold)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def normalized_box_to_pixels(box, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0, min(width, int(round(x1 * width)))),
        max(0, min(height, int(round(y1 * height)))),
        max(0, min(width, int(round(x2 * width)))),
        max(0, min(height, int(round(y2 * height)))),
    ]


def boxes_to_trees(normalized_boxes, scores, width, height, settings):
    """(trees, skipped_partial): pixel boxes, minus trees under min_tree_completeness of the typical tree.

    The typical width, height and area are medians over boxes away from the photo edge (all boxes if none are).
    """
    raw_trees = []
    for box, score in zip(normalized_boxes, scores):
        bbox = normalized_box_to_pixels(box, width, height)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bw <= 0 or bh <= 0:
            continue
        raw_trees.append({
            "confidence": float(score),
            "bbox": bbox,
            "width": bw,
            "height": bh,
            "area": bw * bh,
            "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            "bbox_normalized": [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height],
        })
    margin = settings.border_margin_px
    interior = [t for t in raw_trees
                if t["bbox"][0] > margin and t["bbox"][1] > margin
                and t["bbox"][2] < (width - margin) and t["bbox"][3] < (height - margin)]
    ref_pool = interior if interior else raw_trees
    if ref_pool:
        median_w = float(np.median([t["width"] for t in ref_pool]))
        median_h = float(np.median([t["height"] for t in ref_pool]))
        median_area = float(np.median([t["area"] for t in ref_pool]))
    else:
        median_w = median_h = median_area = 0.0
    trees, skipped = [], 0
    for t in raw_trees:
        w_ratio = t["width"] / median_w if median_w > 0 else 1.0
        h_ratio = t["height"] / median_h if median_h > 0 else 1.0
        area_ratio = t["area"] / median_area if median_area > 0 else 1.0
        if min(w_ratio, h_ratio, area_ratio) < settings.min_tree_completeness:
            skipped += 1
            continue
        t["tree_id"] = len(trees) + 1
        trees.append(t)
    return trees, skipped


class Detector:
    """The Core ML model, loaded once. `detect` is serialized by a lock: Core ML models aren't shared across threads."""

    def __init__(self, model_path):
        import coremltools as ct  # macOS only; imported here so the rest of the package loads anywhere

        self.model = ct.models.MLModel(str(model_path))
        self.input_name = next(iter(self.model.input_description), None)
        if self.input_name is None:
            raise RuntimeError("The Core ML model has no declared input.")
        self._lock = threading.Lock()

    def raw_predict(self, rgb_image, input_size):
        pil_image = Image.fromarray(rgb_image).convert("RGB").resize((input_size, input_size))
        with self._lock:
            prediction = self.model.predict({self.input_name: pil_image})
        return prediction[next(iter(prediction))]

    def detect(self, segmented_rgb, settings):
        """(trees, skipped_partial) on the leaf-only image: pixel boxes, scores and the edge filter."""
        height, width = segmented_rgb.shape[:2]
        boxes, scores = parse_predictions(self.raw_predict(segmented_rgb, settings.model_input_size), settings)
        return boxes_to_trees(boxes, scores, width, height, settings)
