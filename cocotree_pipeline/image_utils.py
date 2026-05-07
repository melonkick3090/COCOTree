from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import mask_utils as U

# Matplotlib-like tab10 palette in RGB [0,1]
COLORS: List[Tuple[float, float, float]] = [
    (0.1216, 0.4667, 0.7059),
    (1.0000, 0.4980, 0.0549),
    (0.1725, 0.6275, 0.1725),
    (0.8392, 0.1529, 0.1569),
    (0.5804, 0.4039, 0.7412),
    (0.5490, 0.3373, 0.2941),
    (0.8902, 0.4667, 0.7608),
    (0.4980, 0.4980, 0.4980),
    (0.7373, 0.7412, 0.1333),
    (0.0902, 0.7451, 0.8118),
]


def resize_longest_side(pil: Image.Image, max_longest_side: int) -> Image.Image:
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_longest_side:
        return pil
    scale = max_longest_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return pil.resize((new_w, new_h), Image.BICUBIC)


def apply_l_clahe_pil(
    pil_rgb: Image.Image,
    *,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> Image.Image:
    if pil_rgb.mode != "RGB":
        pil_rgb = pil_rgb.convert("RGB")
    img = np.array(pil_rgb)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb2)


def image_to_base64_jpeg(pil_image: Image.Image, *, quality: int = 85) -> str:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def masked_crop_rgb(
    pil_full: Image.Image,
    mask_full_u8: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    if pil_full.mode != "RGB":
        pil_full = pil_full.convert("RGB")
    x1, y1, x2, y2 = bbox
    img_np = np.asarray(pil_full).astype(np.uint8)
    crop_img = img_np[y1:y2, x1:x2].copy()
    crop_mask = mask_full_u8[y1:y2, x1:x2] > 0
    out = np.zeros_like(crop_img, dtype=np.uint8)
    out[crop_mask] = crop_img[crop_mask]
    return Image.fromarray(out, mode="RGB")


def prepare_llm_crop(
    pil_full: Image.Image,
    mask_full_u8: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    max_longest_side: int,
    quality: int = 85,
) -> Tuple[Image.Image, Image.Image, str]:
    crop_full = masked_crop_rgb(pil_full, mask_full_u8, bbox)
    crop_llm = resize_longest_side(crop_full, max_longest_side=max_longest_side)
    crop_b64 = image_to_base64_jpeg(crop_llm, quality=quality)
    return crop_full, crop_llm, crop_b64


def _safe_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)[:120]


def _color255(i: int) -> np.ndarray:
    return (np.array(COLORS[i % len(COLORS)]) * 255).astype(np.uint8)


def save_subcrop_debug(
    *,
    out_dir: str,
    pil_full: Image.Image,
    mask_full_u8: np.ndarray,
    sub_bbox: Tuple[int, int, int, int],
    idx: int,
    tag: str = "sam",
    jpeg_quality: int = 90,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sx1, sy1, sx2, sy2 = sub_bbox
    sx1 = max(0, min(pil_full.width, sx1))
    sx2 = max(0, min(pil_full.width, sx2))
    sy1 = max(0, min(pil_full.height, sy1))
    sy2 = max(0, min(pil_full.height, sy2))

    sub_parent_pil = masked_crop_rgb(pil_full, mask_full_u8, (sx1, sy1, sx2, sy2))
    sub_parent_pil.save(
        os.path.join(out_dir, f"__{tag}_subcrop_{idx:02d}.jpg"),
        format="JPEG",
        quality=int(jpeg_quality),
        optimize=True,
    )

    sub_mask01 = (mask_full_u8[sy1:sy2, sx1:sx2] > 0).astype(np.uint8)
    Image.fromarray(sub_mask01 * 255).save(os.path.join(out_dir, f"__{tag}_submask_{idx:02d}.png"))

    full_np = np.array(pil_full.convert("RGB")).copy()
    cv2.rectangle(full_np, (sx1, sy1), (sx2 - 1, sy2 - 1), (0, 255, 255), 3)
    txt = f"{tag}_sub{idx}"
    cv2.putText(full_np, txt, (sx1 + 6, max(18, sy1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(full_np, txt, (sx1 + 6, max(18, sy1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(full_np).save(os.path.join(out_dir, f"__{tag}_subbbox_{idx:02d}_on_full.png"))


def save_candidates_score_overlay_full(
    *,
    out_png_path: str,
    pil_full: Image.Image,
    crop_bbox: U.BBox,
    candidates: List[Tuple[Optional[float], np.ndarray]],
    score_thr: float,
    alpha: float = 0.45,
) -> None:
    def _anchor_on_rect(rect: U.BBox, toward_x: int, toward_y: int):
        x1, y1, x2, y2 = rect
        cx, cy = U.bbox_center(rect)
        dx = toward_x - cx
        dy = toward_y - cy
        if abs(dx) >= abs(dy):
            ax = (x2 - 1) if dx >= 0 else x1
            ay = U.clamp(toward_y, y1, y2 - 1)
        else:
            ay = (y2 - 1) if dy >= 0 else y1
            ax = U.clamp(toward_x, x1, x2 - 1)
        return int(ax), int(ay)

    def _seg_bbox(a, b):
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))

    def _draw_elbow_line(img, p1, p2, avoid_rects=(), thickness=2):
        x1, y1 = p1
        x2, y2 = p2
        elbows = [(x2, y1), (x1, y2)]

        def _path_ok(elbow):
            a = (x1, y1)
            e = elbow
            b = (x2, y2)
            seg1 = _seg_bbox(a, e)
            seg2 = _seg_bbox(e, b)
            for rect in avoid_rects:
                if U.bbox_overlaps(seg1, rect) or U.bbox_overlaps(seg2, rect):
                    return False
            return True

        elbow = elbows[0] if _path_ok(elbows[0]) else elbows[1]
        cv2.line(img, (x1, y1), elbow, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(img, elbow, (x2, y2), (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.line(img, (x1, y1), elbow, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.line(img, elbow, (x2, y2), (255, 255, 255), thickness, cv2.LINE_AA)

    img = np.array(pil_full.convert("RGB")).copy()
    H, W = img.shape[:2]
    x1, y1, x2, y2 = crop_bbox

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_th = 2
    placed_boxes: List[U.BBox] = []

    for i, (sc, mb) in enumerate(candidates):
        if mb is None:
            continue
        mask = (mb > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        target_h = y2 - y1
        target_w = x2 - x1
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        color255 = _color255(i)
        mask_bool = full_mask > 0
        for c in range(3):
            img[..., c] = np.where(mask_bool, (alpha * color255[c] + (1 - alpha) * img[..., c]).astype(np.uint8), img[..., c])

        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            continue
        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        bbox_rect: U.BBox = (bx1, by1, bx2, by2)
        cv2.rectangle(img, (bx1, by1), (bx2 - 1, by2 - 1), tuple(int(x) for x in color255.tolist()), 2)

        sc_txt = "None" if sc is None else f"{float(sc):.2f}"
        passed = (sc is not None) and (float(sc) >= float(score_thr))
        tag = "PASS" if passed else "FAIL"
        text = f"#{i + 1} {sc_txt} {tag}"

        tx = bx1
        ty = by1 - 8
        if ty < 10:
            ty = by2 + 24
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_th)
        pad = 6
        box_w = tw + pad * 2
        box_h = th + pad * 2

        def _make_label_rect(tx_: int, ty_: int) -> U.BBox:
            xA = int(tx_)
            yA = int(ty_ - box_h)
            xB = int(tx_ + box_w)
            yB = int(ty_ + baseline)
            return (xA, yA, xB, yB)

        label_rect = _make_label_rect(tx, ty)
        tries = 0
        while any(U.bbox_overlaps(label_rect, b) for b in placed_boxes) and tries < 30:
            ty += box_h + 6
            if ty > H - 5:
                ty = max(15, by1 - 8)
                tx += box_w + 6
            label_rect = _make_label_rect(tx, ty)
            tries += 1

        tx = max(0, min(W - box_w - 1, tx))
        ty = max(box_h + 1, min(H - 2, ty))
        label_rect = _make_label_rect(tx, ty)
        xA, yA, xB, yB = label_rect
        placed_boxes.append(label_rect)

        bg = (40, 180, 40) if passed else (180, 40, 40)
        roi = img[yA:yB, xA:xB].copy()
        ov = roi.copy()
        ov[:] = bg
        cv2.addWeighted(ov, 0.75, roi, 0.25, 0, roi)
        img[yA:yB, xA:xB] = roi
        cv2.rectangle(img, (xA, yA), (xB - 1, yB - 1), (0, 0, 0), 2)

        text_org = (xA + pad, yB - pad - baseline)
        cv2.putText(img, text, text_org, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, text_org, font, font_scale, (255, 255, 255), text_th, cv2.LINE_AA)

        bbox_cx, bbox_cy = U.bbox_center(bbox_rect)
        lab_cx, lab_cy = U.bbox_center(label_rect)
        p_from = _anchor_on_rect(bbox_rect, int(lab_cx), int(lab_cy))
        p_to = _anchor_on_rect(label_rect, int(bbox_cx), int(bbox_cy))
        _draw_elbow_line(img, p_from, p_to, avoid_rects=(bbox_rect, label_rect), thickness=2)

    Image.fromarray(img).save(out_png_path)


def visualize_union_on_image(
    *,
    pil_full: Image.Image,
    merged_masks_full: List[Tuple[str, np.ndarray | None]],
    alpha: float = 0.6,
    debug_dir: str | None = None,
    skip_if_cov_ge: float | None = None,
) -> Image.Image:
    img_np = np.array(pil_full.convert("RGB")).astype(np.uint8)
    h, w = img_np.shape[:2]
    overlay = img_np.copy()

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        Image.fromarray(img_np).save(os.path.join(debug_dir, "_00_full_rgb.png"))

    for i, (name, mask) in enumerate(merged_masks_full):
        if mask is None:
            continue
        mask_bool = mask > 0
        if mask_bool.shape != (h, w):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        cov = float(mask_bool.mean())
        if skip_if_cov_ge is not None and cov >= float(skip_if_cov_ge):
            continue

        safe = _safe_name(name)
        if debug_dir:
            Image.fromarray((mask_bool.astype(np.uint8) * 255)).save(os.path.join(debug_dir, f"mask_{i:03d}_{safe}.png"))

        color255 = _color255(i)
        if debug_dir:
            solo = img_np.copy()
            for c in range(3):
                solo[..., c][mask_bool] = (alpha * color255[c] + (1 - alpha) * solo[..., c][mask_bool]).astype(np.uint8)
            Image.fromarray(solo).save(os.path.join(debug_dir, f"solo_{i:03d}_{safe}.png"))

        for c in range(3):
            overlay[..., c][mask_bool] = (alpha * color255[c] + (1 - alpha) * overlay[..., c][mask_bool]).astype(np.uint8)
        if debug_dir:
            Image.fromarray(overlay).save(os.path.join(debug_dir, f"acc_{i:03d}_{safe}.png"))

    return Image.fromarray(overlay)


def union_fill_minus_boundary(masks_u8: List[np.ndarray], boundary_px: int = 1) -> Optional[np.ndarray]:
    if not masks_u8:
        return None
    H, W = masks_u8[0].shape
    fill = np.zeros((H, W), dtype=np.uint8)
    for mask in masks_u8:
        np.maximum(fill, (mask > 0).astype(np.uint8) * 255, out=fill)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * boundary_px + 1, 2 * boundary_px + 1))
    boundary = np.zeros((H, W), dtype=np.uint8)
    for mask in masks_u8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
        np.maximum(boundary, grad, out=boundary)
    fill[boundary > 0] = 0
    return fill


def render_masks_only_canvas(
    *,
    h: int,
    w: int,
    merged_masks_full: List[np.ndarray],
    background: str = "black",
) -> Image.Image:
    if background == "white":
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    else:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for i, mask in enumerate(merged_masks_full):
        if mask is None:
            continue
        mask_bool = mask > 0
        if mask_bool.shape != (h, w):
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        canvas[mask_bool] = _color255(i)
    return Image.fromarray(canvas)
