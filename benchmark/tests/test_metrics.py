from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cocotree_benchmark.labels import ExactLabelSimilarity
from cocotree_benchmark.metrics import (
    derive_instance_parents,
    evaluate_image,
    scoped_tree,
)
from cocotree_benchmark.models import ImageTree
from cocotree_benchmark.quality import toy_tree
from cocotree_benchmark.rle import encode_mask
from cocotree_benchmark.validation import parse_prediction_payload


@pytest.fixture()
def protocol() -> dict:
    return {
        "mask_iou_threshold": 0.5,
        "hpq_match_threshold": 0.5,
        "otq_excluded_normalized_labels": ["others"],
        "hpq_excluded_normalized_labels": [],
        "hpq_others_policy": "keep_top_bucket_promote_direct_children",
        "bq_empty_pair_accuracy": 1.0,
        "label_similarity": {"backend": "exact"},
    }


def evaluate(gt: ImageTree, pred: ImageTree, protocol: dict) -> dict:
    result, _matches = evaluate_image(
        gt,
        pred,
        label_scorer=ExactLabelSimilarity(),
        protocol=protocol,
    )
    return result


def test_identity_is_one(protocol: dict) -> None:
    result = evaluate(
        toy_tree(prediction=False),
        toy_tree(prediction=True),
        protocol,
    )
    for field in ("hpq", "otq", "tq", "bq", "mean_nq", "mq", "lq"):
        assert result[field] == pytest.approx(1.0)


def test_one_missing_mask_has_known_recovery(protocol: dict) -> None:
    prediction = toy_tree(prediction=True)
    prediction.nodes = [node for node in prediction.nodes if node.label == "car"]
    result = evaluate(toy_tree(prediction=False), prediction, protocol)
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 1
    assert result["bq"] == pytest.approx(1.0)
    assert result["tq"] == pytest.approx(2.0 / 3.0)
    assert result["otq"] == pytest.approx(2.0 / 3.0)


def test_wrong_label_changes_only_label_dependent_otq_parts(protocol: dict) -> None:
    perfect = toy_tree(prediction=True)
    wrong = ImageTree(
        image_id=1,
        nodes=[
            replace(node, label="tire") if node.label == "wheel" else node
            for node in perfect.nodes
        ],
    )
    perfect_result = evaluate(toy_tree(prediction=False), perfect, protocol)
    wrong_result = evaluate(toy_tree(prediction=False), wrong, protocol)
    assert wrong_result["tp"] == perfect_result["tp"]
    assert wrong_result["mq"] == pytest.approx(perfect_result["mq"])
    assert wrong_result["tq"] == pytest.approx(perfect_result["tq"])
    assert wrong_result["lq"] < perfect_result["lq"]
    assert wrong_result["mean_nq"] < perfect_result["mean_nq"]
    assert wrong_result["otq"] < perfect_result["otq"]


def test_parent_rewire_changes_tree_not_masks_or_labels(protocol: dict) -> None:
    perfect = toy_tree(prediction=True)
    rewired = ImageTree(
        image_id=1,
        nodes=[
            replace(
                node,
                parent_instance_id="ROOT",
                parent_semantic_id="ROOT",
                semantic_depth=1,
            )
            if node.label == "wheel"
            else node
            for node in perfect.nodes
        ],
    )
    result = evaluate(toy_tree(prediction=False), rewired, protocol)
    assert result["mq"] == pytest.approx(1.0)
    assert result["lq"] == pytest.approx(1.0)
    assert result["bq"] == pytest.approx(0.0)
    assert result["tq"] == pytest.approx(0.0)
    assert result["otq"] == pytest.approx(0.0)


def test_below_threshold_assignment_counts_fp_and_fn(protocol: dict) -> None:
    left_mask = encode_mask(
        np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    )
    right_mask = encode_mask(
        np.asarray([[0, 0], [0, 1]], dtype=np.uint8)
    )

    def tree(instance_id: str, mask: dict) -> ImageTree:
        return parse_prediction_payload(
            {
                "schema": "cocotree_prediction_v1",
                "image_id": 7,
                "nodes": [
                    {
                        "instance_id": instance_id,
                        "semantic_id": f"{instance_id}_semantic",
                        "label": "dot",
                        "parent_instance_id": "ROOT",
                        "parent_semantic_id": "ROOT",
                        "segmentation": mask,
                    }
                ],
            }
        )

    result = evaluate(tree("g", left_mask), tree("p", right_mask), protocol)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["otq"] == 0.0


def test_exact_iou_assignment_tie_uses_natural_instance_ids(
    protocol: dict,
) -> None:
    mask = {"size": [2, 2], "counts": [0, 4]}

    def two_label_tree(
        image_id: int,
        first_id: str,
        first_label: str,
        second_id: str,
        second_label: str,
    ) -> ImageTree:
        return parse_prediction_payload(
            {
                "schema": "cocotree_prediction_v1",
                "image_id": image_id,
                "nodes": [
                    {
                        "instance_id": first_id,
                        "semantic_id": f"{first_id}_sem",
                        "label": first_label,
                        "parent_instance_id": "ROOT",
                        "parent_semantic_id": "ROOT",
                        "segmentation": mask,
                    },
                    {
                        "instance_id": second_id,
                        "semantic_id": f"{second_id}_sem",
                        "label": second_label,
                        "parent_instance_id": "ROOT",
                        "parent_semantic_id": "ROOT",
                        "segmentation": mask,
                    },
                ],
            }
        )

    ground_truth = two_label_tree(14, "g1", "cat", "g2", "dog")
    aligned = two_label_tree(14, "p1", "cat", "p2", "dog")
    renamed = two_label_tree(14, "p2", "cat", "p1", "dog")
    aligned_result = evaluate(ground_truth, aligned, protocol)
    renamed_result = evaluate(ground_truth, renamed, protocol)
    assert aligned_result["mq"] == renamed_result["mq"] == 1.0
    assert aligned_result["lq"] == 1.0
    assert renamed_result["lq"] == 0.0


def test_multiple_instances_share_one_semantic_node(protocol: dict) -> None:
    base = {
        "schema": "cocotree_prediction_v1",
        "image_id": 8,
        "nodes": [
            {
                "instance_id": "a",
                "semantic_id": "things",
                "label": "thing",
                "parent_instance_id": "ROOT",
                "parent_semantic_id": "ROOT",
                "segmentation": {"size": [2, 2], "counts": [0, 1, 3]},
            },
            {
                "instance_id": "b",
                "semantic_id": "things",
                "label": "thing",
                "parent_instance_id": "ROOT",
                "parent_semantic_id": "ROOT",
                "segmentation": {"size": [2, 2], "counts": [3, 1]},
            },
        ],
    }
    gt = parse_prediction_payload(base)
    pred_payload = {
        **base,
        "nodes": [
            {**node, "instance_id": f"pred_{node['instance_id']}"}
            for node in base["nodes"]
        ],
    }
    pred = parse_prediction_payload(pred_payload)
    result = evaluate(gt, pred, protocol)
    assert result["num_gt_masks"] == 2
    assert result["num_gt_nodes"] == 1
    assert result["hpq"] == pytest.approx(1.0)
    assert result["otq"] == pytest.approx(1.0)


def test_others_mask_excluded_but_hpq_bucket_retained(protocol: dict) -> None:
    payload = {
        "schema": "cocotree_prediction_v1",
        "image_id": 9,
        "nodes": [
            {
                "instance_id": "o",
                "semantic_id": "others_sem",
                "label": "others",
                "parent_instance_id": "ROOT",
                "parent_semantic_id": "ROOT",
                "segmentation": {"size": [2, 2], "counts": [0, 4]},
            },
            {
                "instance_id": "c",
                "semantic_id": "child_sem",
                "label": "car",
                "parent_instance_id": "o",
                "parent_semantic_id": "others_sem",
                "segmentation": {"size": [2, 2], "counts": [0, 2, 2]},
            },
        ],
    }
    gt = parse_prediction_payload(payload)
    pred = parse_prediction_payload(
        {
            **payload,
            "nodes": [
                {
                    **node,
                    "instance_id": f"p_{node['instance_id']}",
                    "parent_instance_id": (
                        "ROOT"
                        if node["parent_instance_id"] == "ROOT"
                        else f"p_{node['parent_instance_id']}"
                    ),
                    "semantic_id": f"p_{node['semantic_id']}",
                    "parent_semantic_id": (
                        "ROOT"
                        if node["parent_semantic_id"] == "ROOT"
                        else f"p_{node['parent_semantic_id']}"
                    ),
                }
                for node in payload["nodes"]
            ],
        }
    )
    result = evaluate(gt, pred, protocol)
    assert result["num_gt_masks"] == 1
    assert result["num_gt_nodes"] == 2
    assert result["otq"] == pytest.approx(1.0)
    assert result["hpq"] == pytest.approx(1.0)


def test_depth_scope_is_selected_before_others_promotion(protocol: dict) -> None:
    tree = parse_prediction_payload(
        {
            "schema": "cocotree_prediction_v1",
            "image_id": 10,
            "nodes": [
                {
                    "instance_id": "others",
                    "semantic_id": "others_sem",
                    "label": "others",
                    "parent_instance_id": "ROOT",
                    "parent_semantic_id": "ROOT",
                    "segmentation": {"size": [2, 2], "counts": [0, 4]},
                },
                {
                    "instance_id": "child",
                    "semantic_id": "child_sem",
                    "label": "car",
                    "parent_instance_id": "others",
                    "parent_semantic_id": "others_sem",
                    "segmentation": {"size": [2, 2], "counts": [0, 2, 2]},
                },
            ],
        }
    )
    shallow = scoped_tree(
        tree,
        1,
        others_policy="keep_top_bucket_promote_direct_children",
    )
    full = scoped_tree(
        tree,
        None,
        others_policy="keep_top_bucket_promote_direct_children",
    )
    assert {node.label for node in shallow.nodes} == {"others"}
    assert {node.label for node in full.nodes} == {"others", "car"}
    assert {node.semantic_depth for node in full.nodes} == {1}


def test_paper_instance_parent_is_derived_from_semantic_ancestors() -> None:
    tree = parse_prediction_payload(
        {
            "schema": "cocotree_prediction_v1",
            "image_id": 11,
            "nodes": [
                {
                    "instance_id": "person",
                    "semantic_id": "person_sem",
                    "label": "person",
                    "parent_instance_id": "ROOT",
                    "parent_semantic_id": "ROOT",
                    "segmentation": {"size": [2, 2], "counts": [0, 4]},
                },
                {
                    "instance_id": "face",
                    "semantic_id": "face_sem",
                    "label": "face",
                    "parent_instance_id": "person",
                    "parent_semantic_id": "person_sem",
                    "segmentation": {"size": [2, 2], "counts": [0, 1, 3]},
                },
                {
                    "instance_id": "eye",
                    "semantic_id": "eye_sem",
                    "label": "eye",
                    "parent_instance_id": "face",
                    "parent_semantic_id": "face_sem",
                    "segmentation": {"size": [2, 2], "counts": [3, 1]},
                },
            ],
        }
    )
    derived = derive_instance_parents(tree)
    parent = {node.instance_id: node.parent_instance_id for node in derived.nodes}
    # The eye does not overlap face, so the search climbs to person.
    assert parent["eye"] == "person"


def _parent_selection_tree(parent_masks: list[np.ndarray]) -> ImageTree:
    child_mask = np.zeros((3, 3), dtype=np.uint8)
    child_mask[1:, 1:] = 1
    nodes = [
        {
            "instance_id": f"parent_{index}",
            "semantic_id": "parent_sem",
            "label": "parent",
            "parent_instance_id": "ROOT",
            "parent_semantic_id": "ROOT",
            "segmentation": encode_mask(mask),
        }
        for index, mask in enumerate(parent_masks, start=1)
    ]
    nodes.append(
        {
            "instance_id": "child",
            "semantic_id": "child_sem",
            "label": "child",
            "parent_instance_id": "parent_1",
            "parent_semantic_id": "parent_sem",
            "segmentation": encode_mask(child_mask),
        }
    )
    return parse_prediction_payload(
        {
            "schema": "cocotree_prediction_v1",
            "image_id": 12,
            "nodes": nodes,
        }
    )


def test_derived_parent_uses_max_iou_at_first_positive_level() -> None:
    weak = np.zeros((3, 3), dtype=np.uint8)
    weak[1, 1] = 1
    strong = np.zeros((3, 3), dtype=np.uint8)
    strong[1:, 1:] = 1
    tree = derive_instance_parents(_parent_selection_tree([weak, strong]))
    parent = {node.instance_id: node.parent_instance_id for node in tree.nodes}
    assert parent["child"] == "parent_2"


def test_derived_parent_tie_breaks_by_stable_instance_id() -> None:
    identical = np.ones((3, 3), dtype=np.uint8)
    tree = derive_instance_parents(
        _parent_selection_tree([identical, identical.copy()])
    )
    parent = {node.instance_id: node.parent_instance_id for node in tree.nodes}
    assert parent["child"] == "parent_1"


def test_full_tree_parent_cache_matches_scope_local_derivation() -> None:
    tree = parse_prediction_payload(
        {
            "schema": "cocotree_prediction_v1",
            "image_id": 13,
            "nodes": [
                {
                    "instance_id": "others",
                    "semantic_id": "others_sem",
                    "label": "others",
                    "parent_instance_id": "ROOT",
                    "parent_semantic_id": "ROOT",
                    "segmentation": {"size": [2, 2], "counts": [0, 4]},
                },
                {
                    "instance_id": "object",
                    "semantic_id": "object_sem",
                    "label": "object",
                    "parent_instance_id": "others",
                    "parent_semantic_id": "others_sem",
                    "segmentation": {"size": [2, 2], "counts": [0, 3, 1]},
                },
                {
                    "instance_id": "part",
                    "semantic_id": "part_sem",
                    "label": "part",
                    "parent_instance_id": "object",
                    "parent_semantic_id": "object_sem",
                    "segmentation": {"size": [2, 2], "counts": [1, 1, 2]},
                },
            ],
        }
    )
    policy = "keep_top_bucket_promote_direct_children"
    cached_full = derive_instance_parents(tree)
    for max_depth in (1, 2, 3, None):
        cached_scope = scoped_tree(
            cached_full,
            max_depth,
            others_policy=policy,
        )
        local_scope = derive_instance_parents(
            scoped_tree(tree, max_depth, others_policy=policy)
        )
        cached_parents = {
            node.instance_id: node.parent_instance_id
            for node in cached_scope.nodes
        }
        local_parents = {
            node.instance_id: node.parent_instance_id
            for node in local_scope.nodes
        }
        assert cached_parents == local_parents
