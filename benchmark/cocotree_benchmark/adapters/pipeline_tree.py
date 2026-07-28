from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..metrics import derive_instance_parents
from ..models import ROOT_INSTANCE_ID, ROOT_SEMANTIC_ID, ImageTree
from ..rle import jsonable_rle
from ..validation import parse_prediction_payload


def _resolve_artifact(image_dir: Path, raw_path: Any) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates = [
        path,
        image_dir / path,
        image_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(image_dir.rglob(path.name))
    return matches[0].resolve() if matches else None


def _bundle_rles(
    image_dir: Path,
    node: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    files = node.get("files", {}) if isinstance(node.get("files"), dict) else {}
    bundle_path = _resolve_artifact(image_dir, files.get("mask_bundle"))
    if bundle_path is None:
        return [], [f"missing mask bundle for semantic node {node.get('node_id')}"]
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"cannot read {bundle_path}: {exc}"]
    instances = bundle.get("instances", [])
    rles = [
        item.get("rle")
        for item in instances
        if isinstance(item, dict) and isinstance(item.get("rle"), dict)
    ]
    if not rles:
        merged = bundle.get("merged", {})
        if isinstance(merged, dict) and isinstance(merged.get("rle"), dict):
            rles = [merged["rle"]]
            warnings.append(
                f"semantic node {node.get('node_id')} had no instance list; "
                "used merged mask as one instance"
            )
    return rles, warnings


def convert_pipeline_image(image_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    image_path = Path(image_dir)
    if not image_path.name.isdigit():
        raise ValueError(f"Image directory name must be numeric: {image_path}")
    tree_path = image_path / "tree.json"
    if not tree_path.is_file():
        raise FileNotFoundError(f"Missing pipeline tree: {tree_path}")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    raw_nodes = tree.get("nodes")
    if not isinstance(raw_nodes, dict):
        raise ValueError(f"{tree_path}: nodes must be an object")

    output_nodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    for semantic_id, raw_node in sorted(raw_nodes.items()):
        if not isinstance(raw_node, dict):
            warnings.append(f"ignored non-object semantic node {semantic_id}")
            continue
        node = dict(raw_node)
        node.setdefault("node_id", str(semantic_id))
        label = str(node.get("label", "")).strip()
        parent_semantic_id = str(
            node.get("semantic_parent_id")
            or node.get("parent_id")
            or ROOT_SEMANTIC_ID
        )
        rles, node_warnings = _bundle_rles(image_path, node)
        warnings.extend(node_warnings)
        for index, raw_rle in enumerate(rles, start=1):
            output_nodes.append(
                {
                    "instance_id": f"{semantic_id}#{index}",
                    "semantic_id": str(semantic_id),
                    "label": label,
                    "parent_instance_id": ROOT_INSTANCE_ID,
                    "parent_semantic_id": (
                        ROOT_SEMANTIC_ID
                        if parent_semantic_id in {"", "ROOT", "__root__", "n00000"}
                        else parent_semantic_id
                    ),
                    "segmentation": jsonable_rle(raw_rle),
                }
            )
    payload = {
        "schema": "cocotree_prediction_v1",
        "image_id": int(image_path.name),
        "method": {
            "adapter": "cocotree_pipeline_tree_v1",
            "instance_parent_policy": (
                "derive_first_positive_semantic_ancestor_max_iou"
            ),
            "source_tree": str(tree_path.resolve()),
        },
        "nodes": output_nodes,
    }
    parsed = derive_instance_parents(parse_prediction_payload(payload))
    parent_by_instance = {
        node.instance_id: node.parent_instance_id
        for node in parsed.nodes
    }
    for node in payload["nodes"]:
        node["parent_instance_id"] = parent_by_instance[node["instance_id"]]
    # Final validation locks the serialized result, not only the intermediate.
    parse_prediction_payload(payload)
    return payload, warnings


def convert_pipeline_run(
    input_root: str | Path,
    output_root: str | Path,
    manifest_ids: list[int],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_root = Path(input_root)
    destination_root = Path(output_root)
    converted = 0
    warnings: list[dict[str, Any]] = []
    missing: list[int] = []
    for image_id in manifest_ids:
        source_dir = source_root / f"{image_id:012d}"
        if not source_dir.is_dir():
            source_dir = source_root / str(image_id)
        if not source_dir.is_dir():
            missing.append(image_id)
            continue
        payload, image_warnings = convert_pipeline_image(source_dir)
        output_path = (
            destination_root
            / f"{image_id:012d}"
            / "prediction.json"
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; pass --overwrite explicitly"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        converted += 1
        if image_warnings:
            warnings.append(
                {"image_id": image_id, "warnings": image_warnings}
            )
    return {
        "schema": "cocotree_pipeline_conversion_report_v1",
        "requested_images": len(manifest_ids),
        "converted_images": converted,
        "missing_images": missing,
        "missing_image_count": len(missing),
        "warning_images": warnings,
        "warning_image_count": len(warnings),
        "input_root": str(source_root.resolve()),
        "output_root": str(destination_root.resolve()),
    }
