from __future__ import annotations

from dataclasses import replace
from typing import Any

from .labels import normalize_label
from .models import (
    ROOT_INSTANCE_ID,
    ROOT_SEMANTIC_ID,
    ImageTree,
    InstanceNode,
)
from .rle import normalize_rle, rle_area


class PredictionValidationError(ValueError):
    pass


def normalize_image_id(value: Any) -> int:
    text = str(value).strip()
    if not text or not text.isdigit():
        raise PredictionValidationError(f"image_id must contain only digits: {value!r}")
    return int(text)


def _root_instance(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "ROOT", "__root__", "n00000#1"}:
        return ROOT_INSTANCE_ID
    return text


def _root_semantic(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "ROOT", "__root__", "n00000"}:
        return ROOT_SEMANTIC_ID
    return text


def _cycle_check(
    parent_by_id: dict[str, str],
    *,
    root_id: str,
    graph_name: str,
) -> dict[str, int]:
    depths: dict[str, int] = {root_id: 0}

    def visit(node_id: str, path: set[str]) -> int:
        if node_id in depths:
            return depths[node_id]
        if node_id in path:
            raise PredictionValidationError(
                f"{graph_name} contains a cycle at {node_id!r}"
            )
        parent_id = parent_by_id.get(node_id)
        if parent_id is None:
            raise PredictionValidationError(
                f"{graph_name} node {node_id!r} has no parent declaration"
            )
        if parent_id != root_id and parent_id not in parent_by_id:
            raise PredictionValidationError(
                f"{graph_name} node {node_id!r} references missing parent "
                f"{parent_id!r}"
            )
        depth = visit(parent_id, path | {node_id}) + 1
        depths[node_id] = depth
        return depth

    for node_id in sorted(parent_by_id):
        visit(node_id, set())
    return depths


def parse_prediction_payload(payload: dict[str, Any]) -> ImageTree:
    if not isinstance(payload, dict):
        raise PredictionValidationError("Prediction must be a JSON object")
    schema = str(payload.get("schema", ""))
    if schema != "cocotree_prediction_v1":
        raise PredictionValidationError(
            f"Unsupported prediction schema {schema!r}; "
            "expected 'cocotree_prediction_v1'"
        )
    image_id = normalize_image_id(payload.get("image_id"))
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise PredictionValidationError("nodes must be a JSON array")

    nodes: list[InstanceNode] = []
    instance_ids: set[str] = set()
    image_size: tuple[int, int] | None = None
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise PredictionValidationError(f"nodes[{index}] must be an object")
        required_fields = (
            "instance_id",
            "semantic_id",
            "label",
            "parent_instance_id",
            "parent_semantic_id",
            "segmentation",
        )
        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in raw
        ]
        if missing_fields:
            raise PredictionValidationError(
                f"nodes[{index}] is missing required field(s): "
                f"{', '.join(missing_fields)}"
            )
        string_fields = (
            "instance_id",
            "semantic_id",
            "label",
            "parent_instance_id",
            "parent_semantic_id",
        )
        invalid_string_fields = [
            field_name
            for field_name in string_fields
            if not isinstance(raw[field_name], str)
            or not raw[field_name].strip()
        ]
        if invalid_string_fields:
            raise PredictionValidationError(
                f"nodes[{index}] field(s) must be non-empty strings: "
                f"{', '.join(invalid_string_fields)}"
            )
        instance_id = raw["instance_id"].strip()
        semantic_id = raw["semantic_id"].strip()
        label = raw["label"].strip()
        if not instance_id or instance_id == ROOT_INSTANCE_ID:
            raise PredictionValidationError(
                f"nodes[{index}].instance_id must be non-root and non-empty"
            )
        if instance_id in instance_ids:
            raise PredictionValidationError(
                f"Duplicate instance_id {instance_id!r} in image {image_id}"
            )
        instance_ids.add(instance_id)
        if not semantic_id or semantic_id == ROOT_SEMANTIC_ID:
            raise PredictionValidationError(
                f"nodes[{index}].semantic_id must be non-root and non-empty"
            )
        if not normalize_label(label):
            raise PredictionValidationError(f"nodes[{index}].label is empty")
        segmentation = raw.get("segmentation")
        try:
            normalized = normalize_rle(segmentation)
            area = rle_area(normalized)
        except Exception as exc:
            raise PredictionValidationError(
                f"nodes[{index}].segmentation is invalid: {exc}"
            ) from exc
        if area <= 0:
            raise PredictionValidationError(
                f"nodes[{index}].segmentation has zero area"
            )
        node_size = tuple(int(value) for value in normalized["size"])
        if image_size is None:
            image_size = node_size
        elif image_size != node_size:
            raise PredictionValidationError(
                f"RLE size mismatch in image {image_id}: {node_size} != {image_size}"
            )
        nodes.append(
            InstanceNode(
                image_id=image_id,
                instance_id=instance_id,
                semantic_id=semantic_id,
                label=label,
                parent_instance_id=_root_instance(raw["parent_instance_id"]),
                parent_semantic_id=_root_semantic(raw["parent_semantic_id"]),
                segmentation=normalized,
            )
        )

    instance_by_id = {node.instance_id: node for node in nodes}
    instance_parent = {
        node.instance_id: node.parent_instance_id
        for node in nodes
    }
    _cycle_check(
        instance_parent,
        root_id=ROOT_INSTANCE_ID,
        graph_name="instance tree",
    )

    semantic_signature: dict[str, tuple[str, str]] = {}
    for node in nodes:
        signature = (normalize_label(node.label), node.parent_semantic_id)
        previous = semantic_signature.setdefault(node.semantic_id, signature)
        if previous != signature:
            raise PredictionValidationError(
                f"semantic_id {node.semantic_id!r} has inconsistent "
                "label or parent_semantic_id"
            )
    semantic_parent = {
        semantic_id: parent_id
        for semantic_id, (_label, parent_id) in semantic_signature.items()
    }
    semantic_depths = _cycle_check(
        semantic_parent,
        root_id=ROOT_SEMANTIC_ID,
        graph_name="semantic tree",
    )
    for node in nodes:
        if node.parent_instance_id == ROOT_INSTANCE_ID:
            continue
        parent = instance_by_id[node.parent_instance_id]
        semantic_ancestors: set[str] = set()
        current = node.parent_semantic_id
        while current != ROOT_SEMANTIC_ID and current not in semantic_ancestors:
            semantic_ancestors.add(current)
            current = semantic_parent.get(current, ROOT_SEMANTIC_ID)
        if parent.semantic_id not in semantic_ancestors:
            raise PredictionValidationError(
                f"Instance parent of {node.instance_id!r} belongs to semantic "
                f"node {parent.semantic_id!r}, which is not an ancestor of "
                f"{node.semantic_id!r}"
            )
    nodes = [
        replace(node, semantic_depth=semantic_depths[node.semantic_id])
        for node in nodes
    ]
    return ImageTree(image_id=image_id, nodes=nodes)


def release_rows_to_image_tree(
    image_id: int,
    rows: list[dict[str, Any]],
) -> ImageTree:
    prediction_nodes: list[dict[str, Any]] = []
    for row in rows:
        prediction_nodes.append(
            {
                "instance_id": row.get("instance_node_id"),
                "semantic_id": row.get("semantic_node_id"),
                "label": row.get("final_label") or row.get("label"),
                "parent_instance_id": _root_instance(
                    row.get("parent_instance_node_id")
                ),
                "parent_semantic_id": _root_semantic(
                    row.get("parent_semantic_node_id")
                    or row.get("parent_instance_semantic_node_id")
                ),
                "segmentation": row.get("segmentation"),
            }
        )
    return parse_prediction_payload(
        {
            "schema": "cocotree_prediction_v1",
            "image_id": int(image_id),
            "nodes": prediction_nodes,
        }
    )


def validate_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tree = parse_prediction_payload(payload)
    return {
        "image_id": tree.image_id,
        "num_instance_nodes": len(tree.nodes),
        "num_semantic_nodes": len(tree.semantic_ids),
        "max_semantic_depth": max(
            (node.semantic_depth for node in tree.nodes),
            default=0,
        ),
    }
