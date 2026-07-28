from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_PROTOCOL_KEYS = {
    "schema",
    "protocol_id",
    "mask_iou_threshold",
    "hpq_match_threshold",
    "depth_scopes",
    "label_similarity",
}


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_PROTOCOL_KEYS - set(payload))
    if missing:
        raise ValueError(f"Protocol is missing keys: {missing}")
    if payload.get("schema") != "cocotree_metric_protocol_v1":
        raise ValueError(f"Unsupported protocol schema: {payload.get('schema')!r}")
    threshold = float(payload["mask_iou_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("mask_iou_threshold must be in [0, 1]")
    return payload


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

