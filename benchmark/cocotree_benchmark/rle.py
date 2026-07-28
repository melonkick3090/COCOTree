from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def normalize_rle(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize compressed or readable uncompressed COCO RLE."""

    if not isinstance(raw, dict):
        raise ValueError("segmentation must be an RLE object")
    if "rle" in raw and isinstance(raw["rle"], dict):
        raw = raw["rle"]
    size = raw.get("size")
    counts = raw.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("RLE size must be [height, width]")
    if any(
        not isinstance(value, Integral) or isinstance(value, bool)
        for value in size
    ):
        raise ValueError("RLE size values must be integers")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError("RLE dimensions must be positive")
    if isinstance(counts, str):
        normalized: dict[str, Any] = {
            "size": [height, width],
            "counts": counts.encode("utf-8"),
        }
    elif isinstance(counts, bytes):
        normalized = {"size": [height, width], "counts": counts}
    elif isinstance(counts, list):
        if not counts or any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in counts
        ):
            raise ValueError("Uncompressed RLE counts must be integers")
        integer_counts = [int(value) for value in counts]
        if any(value < 0 for value in integer_counts):
            raise ValueError("Uncompressed RLE counts must be non-negative")
        if sum(integer_counts) != height * width:
            raise ValueError(
                "Uncompressed RLE counts do not match height*width: "
                f"{sum(integer_counts)} != {height * width}"
            )
        encoded = mask_utils.frPyObjects(
            {"size": [height, width], "counts": integer_counts},
            height,
            width,
        )
        if isinstance(encoded, list):
            encoded = encoded[0]
        normalized = {"size": [height, width], "counts": encoded["counts"]}
    else:
        raise ValueError("RLE counts must be a compressed string or integer list")
    try:
        decoded = mask_utils.decode(normalized)
    except Exception as exc:
        raise ValueError(f"Invalid COCO RLE: {exc}") from exc
    if tuple(decoded.shape[:2]) != (height, width):
        raise ValueError("Decoded RLE dimensions disagree with size")
    return normalized


def jsonable_rle(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_rle(raw)
    counts = normalized["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(normalized["size"]), "counts": counts}


def rle_area(raw: dict[str, Any]) -> float:
    return float(mask_utils.area(normalize_rle(raw)))


def iou_matrix(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    left_norm = [normalize_rle(rle) for rle in left]
    right_norm = [normalize_rle(rle) for rle in right]
    values = mask_utils.iou(left_norm, right_norm, [0] * len(right_norm))
    return np.asarray(values, dtype=np.float64).reshape(len(left), len(right))


def merge_rles(rles: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_rle(rle) for rle in rles]
    if not normalized:
        return {}
    if len(normalized) == 1:
        return normalized[0]
    return mask_utils.merge(normalized, intersect=False)


def decode_rle(raw: dict[str, Any]) -> np.ndarray:
    return np.asarray(mask_utils.decode(normalize_rle(raw)) > 0, dtype=np.uint8)


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask > 0, dtype=np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}
