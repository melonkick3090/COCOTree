from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
try:
    import pycocotools.mask as coco_mask
except Exception:  # pragma: no cover - optional dependency
    coco_mask = None

from . import mask_utils as U
from .image_utils import masked_crop_rgb

if TYPE_CHECKING:
    from .types import GeneratorConfig, NodeInfo


UTC_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now_str() -> str:
    return time.strftime(UTC_TS_FMT, time.gmtime())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str, max_len: int = 80) -> str:
    s = (name or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "item"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def save_json(path: str, payload: Any, *, indent: int = 2) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    return path


def save_json_gz(path: str, payload: Any) -> str:
    ensure_dir(os.path.dirname(path))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def save_jsonl_gz(path: str, rows: Iterable[Any]) -> str:
    ensure_dir(os.path.dirname(path))
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def save_text(path: str, text: str) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def save_mask_png(path: str, mask_u8_0_255: np.ndarray) -> str:
    ensure_dir(os.path.dirname(path))
    Image.fromarray(mask_u8_0_255).save(path)
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonable(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(blob)


def try_get_git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def hash_python_sources(root_dir: str) -> str:
    root = Path(root_dir)
    files = sorted(str(p) for p in root.rglob("*.py") if p.is_file())
    h = hashlib.sha256()
    for path in files:
        h.update(path.encode("utf-8"))
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def mask_to_coco_rle(mask_u8: np.ndarray) -> Dict[str, Any]:
    mb = (mask_u8 > 0).astype(np.uint8)
    if coco_mask is not None:
        rle = coco_mask.encode(np.asfortranarray(mb))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"format": "coco_rle", "size": [int(mask_u8.shape[0]), int(mask_u8.shape[1])], "counts": counts}

    flat = mb.flatten(order="F")
    counts: List[int] = []
    prev = 0
    run = 0
    for val in flat:
        val = int(val)
        if val == prev:
            run += 1
        else:
            counts.append(int(run))
            run = 1
            prev = val
    counts.append(int(run))
    return {"format": "simple_rle_v1", "size": [int(mask_u8.shape[0]), int(mask_u8.shape[1])], "counts": counts}


def decode_saved_rle(payload: Dict[str, Any]) -> np.ndarray:
    fmt = payload.get("format", "coco_rle")
    size = payload.get("size") or [0, 0]
    h, w = int(size[0]), int(size[1])
    if fmt == "coco_rle":
        if coco_mask is None:
            raise ImportError("pycocotools is required to decode coco_rle payloads")
        mask = coco_mask.decode({"size": [h, w], "counts": payload["counts"]})
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.uint8) * 255

    flat = []
    value = 0
    for run in payload.get("counts", []):
        flat.extend([value] * int(run))
        value = 1 - value
    arr = np.asarray(flat, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"Invalid simple_rle_v1 payload: expected {h*w} values, got {arr.size}")
    return arr.reshape((w, h)).T.astype(np.uint8) * 255


def bbox_to_dict(b: U.BBox) -> Dict[str, int]:
    return {"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])}


def node_dir_from_path(base_dir: str, path_parts: List[str]) -> str:
    safe_parts = [sanitize_filename(p) for p in path_parts]
    dirname = "__".join(safe_parts)
    return os.path.join(base_dir, dirname)


def save_node_masks(
    *,
    pil_full: Image.Image,
    node_dir: str,
    prompt_label: str,
    instances_u8: List[np.ndarray],
    merged_u8: np.ndarray,
    min_area_px: int,
    pad: int,
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str], Optional[U.BBox], Optional[str]]:
    h, w = merged_u8.shape[:2]
    safe_label = sanitize_filename(prompt_label)
    area = int((merged_u8 > 0).sum())
    if area <= 0 or area < int(min_area_px):
        print(f"[DROP][AREA] '{safe_label}' not added: merged_area={area}px < {min_area_px}px")
        return None, None, None, [], None, None

    ensure_dir(node_dir)
    merged_path = save_mask_png(os.path.join(node_dir, f"{safe_label}.mask.png"), merged_u8)

    instance_paths: List[str] = []
    instance_rles: List[Dict[str, Any]] = []
    for idx, mask in enumerate(instances_u8, start=1):
        if mask is None:
            continue
        mu8 = (mask > 0).astype(np.uint8) * 255
        if int((mu8 > 0).sum()) == 0:
            continue
        inst_path = save_mask_png(os.path.join(node_dir, f"{safe_label}_{idx}.mask.png"), mu8)
        instance_paths.append(inst_path)
        instance_rles.append({
            "index": int(idx),
            "area_px": int((mu8 > 0).sum()),
            "rle": mask_to_coco_rle(mu8),
        })

    bbox = U.bbox_from_mask(merged_u8)
    if bbox is None:
        return None, None, None, instance_paths, None, None
    bbox_pad = U.pad_bbox(bbox, pad=int(pad), w=int(w), h=int(h))

    mask_original_pil = masked_crop_rgb(pil_full, merged_u8, bbox_pad)
    mask_original_path = os.path.join(node_dir, f"{safe_label}.mask.original.png")
    mask_original_pil.save(mask_original_path)

    bbox_rgb_pil = pil_full.crop(bbox_pad)
    bbox_rgb_path = os.path.join(node_dir, f"{safe_label}.bbox.rgb.png")
    bbox_rgb_pil.save(bbox_rgb_path)

    bundle = {
        "label": prompt_label,
        "image_space": "processed_root",
        "image_size": {"width": int(w), "height": int(h)},
        "crop_bbox": bbox_to_dict(bbox_pad),
        "merged": {
            "area_px": int((merged_u8 > 0).sum()),
            "rle": mask_to_coco_rle(merged_u8),
            "png": merged_path,
        },
        "instances": instance_rles,
        "instance_pngs": list(instance_paths),
        "mask_original_png": mask_original_path,
        "bbox_rgb_png": bbox_rgb_path,
    }
    bundle_path = save_json(os.path.join(node_dir, f"{safe_label}.masks.json"), bundle)
    return merged_path, mask_original_path, bbox_rgb_path, instance_paths, bbox_pad, bundle_path


def render_tree_text(nodes: Dict[str, "NodeInfo"], root_children: List[str], *, child_overrides: Optional[Dict[str, List[str]]] = None) -> str:
    lines: List[str] = ["└─ root"]

    def rec(node_id: str, prefix: str, is_last: bool) -> None:
        node = nodes[node_id]
        branch = "└─ " if is_last else "├─ "
        lines.append(f"{prefix}{branch}{node.label}")
        child_prefix = prefix + ("   " if is_last else "│  ")
        children = child_overrides.get(node_id) if child_overrides is not None else list(getattr(node, "children", []) or [])
        children = [cid for cid in (children or []) if cid in nodes]
        for idx, cid in enumerate(children):
            rec(cid, child_prefix, idx == len(children) - 1)

    alive_roots = [cid for cid in root_children if cid in nodes]
    for idx, cid in enumerate(alive_roots):
        rec(cid, "   ", idx == len(alive_roots) - 1)
    return "\n".join(lines) + "\n"


def save_tree_summary_provenance(
    *,
    image_out_dir: str,
    tree_payload: Dict[str, Any],
    tree_text: str,
    summary: Dict[str, Any],
    provenance: Dict[str, Any],
    tree_filename: str = "tree.json",
    tree_text_filename: str = "tree.txt",
    summary_filename: str = "run_summary.json",
    provenance_filename: str = "provenance.json",
) -> Tuple[str, str, str, str]:
    tree_json_path = save_json(os.path.join(image_out_dir, tree_filename), tree_payload)
    tree_txt_path = save_text(os.path.join(image_out_dir, tree_text_filename), tree_text)
    summary_path = save_json(os.path.join(image_out_dir, summary_filename), summary)
    provenance_path = save_json(os.path.join(image_out_dir, provenance_filename), provenance)
    return tree_json_path, tree_txt_path, summary_path, provenance_path


def write_done_marker(image_out_dir: str, payload: Dict[str, Any]) -> str:
    return save_json(os.path.join(image_out_dir, "_DONE.json"), payload)


def build_basic_provenance(
    *,
    cfg: "GeneratorConfig",
    prompt_payload: Dict[str, str],
    image_source_path: Optional[str],
    image_sha256: Optional[str],
    package_dir: str,
    model_name: str,
) -> Dict[str, Any]:
    provenance = {
        "created_at_utc": utc_now_str(),
        "python_version": sys.version,
        "config": cfg.to_dict(),
        "config_hash": sha256_jsonable(cfg.to_dict()),
        "prompts": prompt_payload,
        "prompt_hash": sha256_jsonable(prompt_payload),
        "model_name": model_name,
        "image_source_path": image_source_path,
        "image_source_sha256": image_sha256,
        "package_source_hash": hash_python_sources(package_dir),
        "git_commit": try_get_git_commit(package_dir),
    }
    return provenance
