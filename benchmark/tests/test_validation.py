from __future__ import annotations

import numpy as np
import pytest

from cocotree_benchmark.rle import encode_mask, iou_matrix
from cocotree_benchmark.validation import (
    PredictionValidationError,
    parse_prediction_payload,
)


def node(
    instance_id: str,
    semantic_id: str,
    *,
    parent_instance_id: str = "ROOT",
    parent_semantic_id: str = "ROOT",
) -> dict:
    return {
        "instance_id": instance_id,
        "semantic_id": semantic_id,
        "label": semantic_id,
        "parent_instance_id": parent_instance_id,
        "parent_semantic_id": parent_semantic_id,
        "segmentation": {"size": [2, 2], "counts": [0, 1, 3]},
    }


def payload(nodes: list[dict]) -> dict:
    return {
        "schema": "cocotree_prediction_v1",
        "image_id": 1,
        "nodes": nodes,
    }


def test_unknown_parent_rejected() -> None:
    with pytest.raises(PredictionValidationError, match="missing parent"):
        parse_prediction_payload(
            payload(
                [
                    node(
                        "child",
                        "child_sem",
                        parent_instance_id="missing",
                    )
                ]
            )
        )


@pytest.mark.parametrize(
    "field_name",
    ["parent_instance_id", "parent_semantic_id"],
)
def test_missing_required_parent_field_rejected(field_name: str) -> None:
    raw_node = node("item", "item_sem")
    raw_node.pop(field_name)
    with pytest.raises(PredictionValidationError, match="required field"):
        parse_prediction_payload(payload([raw_node]))


@pytest.mark.parametrize("invalid_value", [None, "", "   ", 0])
def test_parent_fields_require_nonempty_strings(invalid_value: object) -> None:
    raw_node = node("item", "item_sem")
    raw_node["parent_instance_id"] = invalid_value
    with pytest.raises(PredictionValidationError, match="non-empty strings"):
        parse_prediction_payload(payload([raw_node]))


def test_instance_cycle_rejected() -> None:
    with pytest.raises(PredictionValidationError, match="cycle"):
        parse_prediction_payload(
            payload(
                [
                    node(
                        "a",
                        "a_sem",
                        parent_instance_id="b",
                        parent_semantic_id="b_sem",
                    ),
                    node(
                        "b",
                        "b_sem",
                        parent_instance_id="a",
                        parent_semantic_id="a_sem",
                    ),
                ]
            )
        )


def test_instance_parent_may_skip_semantic_intermediate() -> None:
    parsed = parse_prediction_payload(
        payload(
            [
                node("person", "person_sem"),
                node(
                    "face",
                    "face_sem",
                    parent_instance_id="person",
                    parent_semantic_id="person_sem",
                ),
                node(
                    "eye",
                    "eye_sem",
                    parent_instance_id="person",
                    parent_semantic_id="face_sem",
                ),
            ]
        )
    )
    assert len(parsed.nodes) == 3


def test_non_ancestor_instance_parent_rejected() -> None:
    with pytest.raises(PredictionValidationError, match="not an ancestor"):
        parse_prediction_payload(
            payload(
                [
                    node("person", "person_sem"),
                    node("dog", "dog_sem"),
                    node(
                        "eye",
                        "eye_sem",
                        parent_instance_id="dog",
                        parent_semantic_id="person_sem",
                    ),
                ]
            )
        )


def test_compressed_and_uncompressed_rle_have_identical_iou() -> None:
    mask = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    compressed = encode_mask(mask)
    uncompressed = {"size": [2, 2], "counts": [0, 2, 2]}
    matrix = iou_matrix([compressed], [uncompressed])
    assert float(matrix[0, 0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "segmentation",
    [
        {"size": [2.5, 2], "counts": [0, 4]},
        {"size": [True, 2], "counts": [0, 2]},
        {"size": [2, 2], "counts": [0.9, 4.9]},
        {"size": [2, 2], "counts": [False, 4]},
    ],
)
def test_rle_rejects_non_integer_size_or_counts(segmentation: dict) -> None:
    raw_node = node("item", "item_sem")
    raw_node["segmentation"] = segmentation
    with pytest.raises(PredictionValidationError, match="integers"):
        parse_prediction_payload(payload([raw_node]))
