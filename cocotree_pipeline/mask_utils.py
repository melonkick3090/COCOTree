from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) exclusive

__all__ = [
    "BBox",
    "clamp",
    "odd",
    "clip_bbox",
    "pad_bbox",
    "bbox_from_mask",
    "bbox_area",
    "bbox_union",
    "bbox_intersection",
    "bbox_iou",
    "bbox_overlaps",
    "bbox_center",
    "bbox_to_slices",
    "merge_bboxes_greedy",
    "to_mb01",
    "to_u8_255",
    "ensure_u8",
    "mask_area",
    "mask_bbox_area_ratio",
    "mask_union",
    "mask_intersection",
    "mask_apply_bbox",
    "mask_iou_u8",
    "containment_ratio",
    "morph_open",
    "morph_close",
    "morph_dilate",
    "morph_open_close",
    "fill_holes01",
    "connected_components",
    "components_bboxes",
    "contours_from_mask",
    "safe_resize_u8",
    "crop_by_bbox",
]


def clamp(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def odd(v: int) -> int:
    if v <= 0:
        return 0
    return int(v) | 1


def clip_bbox(b: BBox, *, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = b
    x1 = clamp(x1, 0, w)
    y1 = clamp(y1, 0, h)
    x2 = clamp(x2, 0, w)
    y2 = clamp(y2, 0, h)
    if x2 < x1:
        x2 = x1
    if y2 < y1:
        y2 = y1
    return (x1, y1, x2, y2)


def pad_bbox(b: BBox, *, pad: int, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = b
    return clip_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w=w, h=h)


def bbox_from_mask(mask_u8: np.ndarray) -> Optional[BBox]:
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_area(b: BBox) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def bbox_intersection(a: BBox, b: BBox) -> Optional[BBox]:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_iou(a: BBox, b: BBox) -> float:
    inter = bbox_intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = bbox_area(inter)
    union_area = bbox_area(a) + bbox_area(b) - inter_area
    return float(inter_area) / float(union_area) if union_area > 0 else 0.0


def bbox_overlaps(a: BBox, b: BBox) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def bbox_to_slices(b: BBox) -> Tuple[slice, slice]:
    return (slice(b[1], b[3]), slice(b[0], b[2]))


def merge_bboxes_greedy(boxes: Sequence[BBox], *, target_k: int) -> List[BBox]:
    if target_k <= 0:
        target_k = 1
    out = list(boxes)
    if len(out) <= target_k:
        return out

    while len(out) > target_k:
        areas = [bbox_area(b) for b in out]
        i = int(np.argmin(np.asarray(areas, dtype=np.int64)))
        bi = out[i]
        ci = bbox_center(bi)

        best_j = -1
        best_d = None
        for j, bj in enumerate(out):
            if j == i:
                continue
            cj = bbox_center(bj)
            d = (ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_j = j

        if best_j < 0:
            break
        merged = bbox_union(bi, out[best_j])
        for idx in sorted((i, best_j), reverse=True):
            out.pop(idx)
        out.append(merged)
    return out


def to_mb01(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def to_u8_255(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def ensure_u8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def mask_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def mask_bbox_area_ratio(mask: np.ndarray, bbox: BBox) -> float:
    area = bbox_area(bbox)
    if area <= 0:
        return 0.0
    sy, sx = bbox_to_slices(bbox)
    return float((mask[sy, sx] > 0).sum()) / float(area)


def mask_union(masks: Sequence[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    out = to_mb01(masks[0])
    for mask in masks[1:]:
        out = np.maximum(out, to_mb01(mask))
    return out


def mask_intersection(masks: Sequence[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    out = masks[0] > 0
    for mask in masks[1:]:
        out &= mask > 0
    return out.astype(np.uint8)


def mask_apply_bbox(mask: np.ndarray, bbox: BBox, *, fill: int = 0) -> np.ndarray:
    out = np.full_like(mask, fill)
    sy, sx = bbox_to_slices(bbox)
    out[sy, sx] = mask[sy, sx]
    return out


def mask_iou_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    a = a_u8 > 0
    b = b_u8 > 0
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def containment_ratio(small_u8: np.ndarray, big_u8: np.ndarray) -> float:
    s = small_u8 > 0
    b = big_u8 > 0
    area_s = int(s.sum())
    if area_s <= 0:
        return 0.0
    inter = int((s & b).sum())
    return inter / float(area_s)


def morph_open(u8: np.ndarray, k: int, *, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    k = odd(int(k))
    if k <= 1:
        return u8
    kernel = cv2.getStructuringElement(shape, (k, k))
    return cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel)


def morph_close(u8: np.ndarray, k: int, *, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    k = odd(int(k))
    if k <= 1:
        return u8
    kernel = cv2.getStructuringElement(shape, (k, k))
    return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)


def morph_dilate(u8: np.ndarray, k: int, *, iters: int = 1, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    k = odd(int(k))
    if k <= 1 or iters <= 0:
        return u8
    kernel = cv2.getStructuringElement(shape, (k, k))
    return cv2.dilate(u8, kernel, iterations=int(iters))


def morph_open_close(u8: np.ndarray, k: int, *, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    u8 = morph_open(u8, k, shape=shape)
    u8 = morph_close(u8, k, shape=shape)
    return u8


# def fill_holes01(mb01: np.ndarray) -> np.ndarray:
#     mb01 = to_mb01(mb01).astype(np.uint8)
#     h, w = mb01.shape
#     if h < 3 or w < 3:
#         return mb01

#     out = mb01.copy()
#     border = (out[0, :].copy(), out[-1, :].copy(), out[:, 0].copy(), out[:, -1].copy())
#     out[[0, -1], :] = 0
#     out[:, [0, -1]] = 0

#     inv = (out == 0).astype(np.uint8)
#     ff = inv.copy()
#     cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
#     out[ff == 1] = 1
#     out[0, :], out[-1, :] = border[0], border[1]
#     out[:, 0], out[:, -1] = border[2], border[3]
#     return out

def fill_holes01(mb01: np.ndarray) -> np.ndarray:
    mb01 = to_mb01(mb01).astype(np.uint8)
    h, w = mb01.shape
    if h < 3 or w < 3:
        return mb01
    
    inv = (1 - mb01).astype(np.uint8)
    inv_pad = np.pad(inv, ((1, 1), (1, 1)), mode="constant", constant_values=1)
    ff = inv_pad.copy()
    flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8) 
    cv2.floodFill(ff, flood_mask, (0, 0), 2)
    holes_pad = ((ff == 1) & (inv_pad == 1)).astype(np.uint8)
    holes = holes_pad[1:-1, 1:-1]
    out = mb01.copy()
    out[holes > 0] = 1
    return out


def connected_components(mask01: np.ndarray, *, connectivity: int = 8):
    m = to_mb01(mask01)
    return cv2.connectedComponentsWithStats(m, connectivity=int(connectivity))


def components_bboxes(mask01: np.ndarray, *, min_area: int = 0) -> List[BBox]:
    min_area = int(min_area)
    num, _labels, stats, _ = connected_components(mask01, connectivity=8)
    out: List[BBox] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append((x, y, x + w, y + h))
    return out


def contours_from_mask(mask01: np.ndarray, *, mode: int = cv2.RETR_EXTERNAL, method: int = cv2.CHAIN_APPROX_SIMPLE):
    m = to_u8_255(mask01)
    res = cv2.findContours(m, mode, method)
    return res[0] if len(res) == 2 else res[1]


def safe_resize_u8(u8: np.ndarray, *, w: int, h: int, interp: int = cv2.INTER_NEAREST) -> np.ndarray:
    if u8 is None:
        raise ValueError("safe_resize_u8: input is None")
    if u8.dtype != np.uint8:
        raise ValueError(f"safe_resize_u8: expected uint8, got {u8.dtype}")
    if w <= 0 or h <= 0:
        raise ValueError(f"safe_resize_u8: invalid target size w={w}, h={h}")
    return cv2.resize(u8, (int(w), int(h)), interpolation=int(interp))


def crop_by_bbox(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    sy, sx = bbox_to_slices(bbox)
    return arr[sy, sx]
