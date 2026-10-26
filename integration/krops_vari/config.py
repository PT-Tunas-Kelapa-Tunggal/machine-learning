"""Every setting of the VARI pipeline in one object, with the values of integration.ipynb.

A Settings object is passed to every stage instead of module-level constants, so two scans with different settings
can run side by side (the API does). Override a value with `Settings(...)`, `dataclasses.replace(...)` or
`Settings.from_json(path)`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO / "segmentation" / "final_flow_ut1" / "yolo26m-krops-v0.1.mlpackage"

MG_PER_MGO = 24.305 / 40.304


@dataclass
class Settings:
    # ---- detection (Core ML)
    model_path: str = str(DEFAULT_MODEL_PATH)
    conf_threshold: float = 0.25          # discard detections under this score
    iou_threshold: float = 0.45           # non-maximum suppression overlap
    model_input_size: int = 640           # square model input, px
    min_tree_completeness: float = 0.70   # keep a tree only if width, height and area are each ≥ 70% of the median
    border_margin_px: int = 5             # a box this close to the edge doesn't set the typical tree size

    # ---- zoning
    zone_grid_rows: int = 3
    zone_grid_cols: int = 3
    row_tolerance_multiplier: float = 0.75
    max_zone_distance_multiplier: float = 3.0
    zone_padding_multiplier: float = 0.15

    # ---- boost and fertilizer
    min_p95_p25_vari_gap_for_blanket_boost: float = 0.03
    max_nutrient_kg_ha_per_year: dict = field(default_factory=lambda: {"Mg": 22.0, "N": 145.0})
    fertilizer_nutrient_by_product: dict = field(default_factory=lambda: {
        "kieserite_Mg_expensive": "Mg", "dolomite_Mg_cheaper": "Mg",
        "ZA_N_cheaper": "N", "urea_N_expensive": "N",
    })
    fertilizer_product_nutrient_fractions: dict = field(default_factory=lambda: {
        "kieserite_Mg_expensive": 0.255 * MG_PER_MGO,  # label: min 25.5% MgO
        "dolomite_Mg_cheaper": 0.22 * MG_PER_MGO,      # varies by product: use the actual label
        "ZA_N_cheaper": 0.21,
        "urea_N_expensive": 0.46,
    })
    fertilizer_price_per_kg: dict = field(default_factory=dict)
    zone_area_ha_by_id: dict = field(default_factory=dict)
    uniform_zone_area_ha: float | None = None
    default_tree_spacing_m: float = 10.0       # midpoint of 9–11 m: 0.01 ha per palm
    use_spacing_based_zone_area_default: bool = True

    # ---- zone confidence (team decisions, 23 Oct 2026)
    zone_confidence_by_id: dict = field(default_factory=dict)  # explicit per-zone values; empty = computed
    default_zone_confidence: float = 0.60      # stands in for the flight confidence until L6 is known
    zone_confidence_full_trees: int = 9        # GUESS: tree factor = min(1, sqrt(n / this))
    zone_leaf_pixel_full_ratio: float = 0.70   # GUESS: leaf factor = min(1, ratio / this)
    zone_reference_min_factor: float = 0.70    # decided: a lower zone factor doesn't set P95/P25
    zone_confidence_floor: float = 0.10        # the boost accepts 0.10–1.0
    min_reference_zones_suggested: int = 5     # proposal, NOT decided: fewer only adds a run flag
    use_flight_confidence: bool = True         # multiply in L6 when it was computed (team decision)

    # ---- spreading plan (weather_fertilizer.ipynb)
    rounds_per_year: int = 2                   # GUESS – no source yet
    round_gap_months: int = 6                  # GUESS – no source yet
    # Days from spreading Mg to spreading N. Dolomite is a lime: higher pH near its particles raises NH3 loss from
    # urea/ammonium; practice is ≥ 2 weeks (14–28 days), no PPKS/MPOB number. Kieserite: 1 = a different day (GUESS).
    min_days_mg_to_n: dict = field(default_factory=lambda: {"dolomite_Mg_cheaper": 14, "kieserite_Mg_expensive": 1})

    @property
    def trees_per_zone(self):
        return self.zone_grid_rows * self.zone_grid_cols

    @classmethod
    def from_json(cls, path):
        """Settings with the values in a JSON object overriding the defaults; unknown keys are refused."""
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
