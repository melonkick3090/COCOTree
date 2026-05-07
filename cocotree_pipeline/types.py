from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import mask_utils as U


@dataclass
class GeneratorConfig:
    input_resize_longest_side: int = 2048

    # Preprocess
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    # SAM thresholds
    sam_capture_threshold: float = 0.3
    root_selection_threshold: float = 0.5
    child_selection_threshold_default: float = 0.5
    child_selection_threshold_small: float = 0.4
    child_small_area_pct_of_full_image: float = 5.0
    selection_mode: str = "per_candidate" # "median" | "per_candidate"

    # Tree / crop / filtering
    pad: int = 4
    min_area_pct: float = 0.0
    max_depth: int = 10
    max_children: int = 20
    max_total_nodes: int = 1000
    llm_batch_size: int = 32
    llm_crop_max_side: int = 1500

    root_max_mask_area_frac: float = 0.8
    candidate_cover_parent_max: float = 0.95
    parent_child_iou_max: float = 0.9
    sibling_containment_thr: float = 0.9

    # Subcrop policy
    subcrop_split_fill_ratio: float = 3.0
    subcrop_min_cc_area_frac: float = 0.02
    subcrop_min_cc_area_floor: int = 30
    max_subcrops: Optional[int] = None  # None => unlimited

    # Others policy
    others_min_component_area_frac: float = 0.01

    # Persistence
    debug_save: bool = False
    save_raw_hmaps: bool = True
    save_hmaps_vis: bool = False
    save_candidate_png: bool = False
    save_subcrop_debug: bool = False
    save_llm_raw_responses: bool = True
    save_candidate_cache: bool = True
    save_raw_candidate_masks: bool = True
    save_processed_candidate_masks: bool = True

    # LLM temperatures / retries
    initial_temperature: float = 0.0
    batch_temperature: float = 0.0
    fix_temperature: float = 0.0
    single_temperature: float = 0.0
    initial_max_retry: int = 5
    llm_max_retry: int = 3

    # Provenance / schema
    generator_schema_version: str = "cocotree_pipeline"
    prompt_set_name: str = "prompts.py"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeInfo:
    node_id: str
    label: str
    path: List[str]
    parent_id: Optional[str]
    merged_mask: np.ndarray  # uint8 0/255, processed-root image space
    crop_bbox: U.BBox
    pad: int
    node_dir: str
    merged_mask_path: str
    mask_rgb_path: str
    bbox_rgb_path: str
    mask_bundle_path: str
    instance_mask_paths: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    @staticmethod
    def bbox_to_dict(b: U.BBox) -> Dict[str, int]:
        return {"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])}

    def to_json(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "path": list(self.path),
            "parent_id": self.parent_id,
            "status": "success",
            "crop_bbox": self.bbox_to_dict(self.crop_bbox),
            "pad": int(self.pad),
            "image_space": "processed_root",
            "files": {
                "node_dir": self.node_dir,
                "merged_mask": self.merged_mask_path,
                "mask_rgb": self.mask_rgb_path,
                "bbox_rgb": self.bbox_rgb_path,
                "mask_bundle": self.mask_bundle_path,
                "instance_masks": list(self.instance_mask_paths),
            },
            "children": list(self.children),
        }


@dataclass
class RunResult:
    output_dir: str
    summary: Dict[str, Any]
    tree_str: str
    final_mask_path: str
    tree_path: str
