from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import mask_utils as U


def compute_pp_params_from_area(
    parent_area_px: int,
    *,
    min_comp_frac: float = 0.0015,
    min_comp_floor: int = 30,
    min_comp_ceil: int = 20000,
    k_sqrt_div: float = 100.0,
    k_min: int = 3,
    k_max: int = 11,
    open_ratio: float = 0.8,
    fill_holes: bool = True,
) -> Dict[str, int | bool]:
    area = max(int(parent_area_px), 0)
    min_comp_area = int(area * float(min_comp_frac))
    min_comp_area = U.clamp(min_comp_area, min_comp_floor, min_comp_ceil)

    close_k = int(np.sqrt(area) / k_sqrt_div)
    close_k = U.clamp(close_k, k_min, k_max)
    close_k = max(1, close_k)

    open_k = int(round(close_k * open_ratio))
    open_k = U.clamp(open_k, 0, close_k)
    if close_k <= 3:
        open_k = min(open_k, 2)

    return {
        "min_comp_area": int(min_comp_area),
        "close_k": int(close_k),
        "open_k": int(open_k),
        "fill_holes": bool(fill_holes),
    }


def postprocess_mask_u8(
    mb: np.ndarray,
    *,
    min_comp_area: int = 20,
    close_k: int = 5,
    open_k: int = 3,
    fill_holes: bool = True,
) -> np.ndarray:
    mb01 = U.to_mb01(mb)
    u8 = U.to_u8_255(mb01)

    if close_k > 1:
        u8 = U.morph_close(u8, close_k)
    if open_k > 1:
        u8 = U.morph_open(u8, open_k)
    mb01 = U.to_mb01(u8)

    thr = int(min_comp_area)
    if thr > 0:
        num, labels, stats, _ = U.connected_components(mb01, connectivity=8)
        out = np.zeros_like(mb01, dtype=np.uint8)
        for i in range(1, int(num)):
            if int(stats[i, cv2.CC_STAT_AREA]) >= thr:
                out[labels == i] = 1
        mb01 = out

    if fill_holes:
        mb01 = U.fill_holes01(mb01)

    # 작은 구멍만 채우고 싶은 경우
    # if fill_holes:
    #     before = int(mb01.sum())
    #     filled = U.fill_holes01(mb01)
    #     added = int(filled.sum()) - before

    # # if added <= 64 and added <= max(1, int(before * 0.1)):
    # if added <= max(1, int(before * 0.1)):
    #     mb01 = filled

    return mb01.astype(np.uint8)


def split_parent_into_subcrops(
    *,
    parent_mask_full_u8: np.ndarray,
    parent_bbox_pad: U.BBox,
    pad: int,
    w_full: int,
    h_full: int,
    split_fill_ratio: float = 3.0,
    min_cc_area_frac: float = 0.02,
    min_cc_area_floor: int = 50,
    max_subcrops: Optional[int] = None,
    include_full_parent: bool = True,
    # include_full_parent: bool = False,
) -> List[Tuple[U.BBox, np.ndarray]]:
    x1, y1, x2, y2 = U.clip_bbox(parent_bbox_pad, w=int(w_full), h=int(h_full))
    if x2 <= x1 or y2 <= y1:
        whole = U.to_mb01(parent_mask_full_u8[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]).astype(np.uint8)
        return [((x1, y1, x2, y2), whole)]

    crop01 = U.to_mb01(parent_mask_full_u8[y1:y2, x1:x2]).astype(np.uint8)
    mask_area_px = int(crop01.sum())
    bbox_area_px = int(U.bbox_area((x1, y1, x2, y2)))
    if mask_area_px <= 0:
        return [((x1, y1, x2, y2), crop01)]
    if bbox_area_px <= int(mask_area_px * float(split_fill_ratio)):
        return [((x1, y1, x2, y2), crop01)]

    min_cc_area = max(int(mask_area_px * float(min_cc_area_frac)), int(min_cc_area_floor))

    kernel = np.ones((3, 3), np.uint8)
    crop01_eroded = cv2.erode(crop01, kernel, iterations=1)

    num, labels, stats, _ = U.connected_components((crop01_eroded * 255).astype(np.uint8),connectivity=8)
    # print(f"[SUBCROP] mask_area_px={mask_area_px}")
    # print(f"[SUBCROP] min_cc_area={min_cc_area}")
    # print(f"[SUBCROP] total_cc={int(num) - 1}")

    comps: List[Tuple[int, int, U.BBox]] = []
    for i in range(1, int(num)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_cc_area):
            continue
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        comps.append((area, i, (cx, cy, cx + cw, cy + ch)))

    # print(f"[SUBCROP] kept_cc={len(comps)}\n")

    if len(comps) <= 1:
        return [((x1, y1, x2, y2), crop01)]

    comps.sort(key=lambda t: t[0], reverse=True)
    if max_subcrops is not None and int(max_subcrops) > 0:
        comps = comps[: int(max_subcrops)]
        
    # print(f"[SUBCROP] max_subcrops={max_subcrops}")
    # print(f"[SUBCROP] comps_after_cap={len(comps)}")

    out: List[Tuple[U.BBox, np.ndarray]] = []
    if include_full_parent:
        out.append(((x1, y1, x2, y2), crop01))

    # for _area, cc_id, (cx1, cy1, cx2, cy2) in comps:
    #     bx1 = x1 + int(cx1)
    #     by1 = y1 + int(cy1)
    #     bx2 = x1 + int(cx2)
    #     by2 = y1 + int(cy2)
    #     sb_full = U.pad_bbox((bx1, by1, bx2, by2), pad=int(pad), w=int(w_full), h=int(h_full))
    #     sx1, sy1, sx2, sy2 = sb_full

    #     cc_crop01 = (labels == int(cc_id)).astype(np.uint8)
    #     sub_h = int(sy2 - sy1)
    #     sub_w = int(sx2 - sx1)
    #     sub_mask01 = np.zeros((sub_h, sub_w), dtype=np.uint8)

    #     src_x1 = max(0, int(sx1 - x1))
    #     src_y1 = max(0, int(sy1 - y1))
    #     src_x2 = min(int(cc_crop01.shape[1]), int(sx2 - x1))
    #     src_y2 = min(int(cc_crop01.shape[0]), int(sy2 - y1))
    #     dst_x1 = max(0, int(x1 - sx1))
    #     dst_y1 = max(0, int(y1 - sy1))
    #     dst_x2 = dst_x1 + (src_x2 - src_x1)
    #     dst_y2 = dst_y1 + (src_y2 - src_y1)

    #     if (src_x2 > src_x1) and (src_y2 > src_y1) and (dst_x2 > dst_x1) and (dst_y2 > dst_y1):
    #         dst_x2 = min(dst_x2, sub_w)
    #         dst_y2 = min(dst_y2, sub_h)
    #         src_x2 = src_x1 + (dst_x2 - dst_x1)
    #         src_y2 = src_y1 + (dst_y2 - dst_y1)
    #         sub_mask01[dst_y1:dst_y2, dst_x1:dst_x2] = cc_crop01[src_y1:src_y2, src_x1:src_x2]

    #     if int(sub_mask01.sum()) <= 0:
    #         continue
    #     out.append((sb_full, sub_mask01))

    return out or [((x1, y1, x2, y2), crop01)]
