from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .labels import ExactLabelSimilarity
from .metrics import evaluate_image
from .models import ImageTree, InstanceNode
from .rle import encode_mask
from .validation import parse_prediction_payload


def _rle(mask: list[list[int]]) -> dict[str, Any]:
    return encode_mask(np.asarray(mask, dtype=np.uint8))


def toy_tree(*, prediction: bool = False) -> ImageTree:
    prefix = "p" if prediction else "g"
    return parse_prediction_payload(
        {
            "schema": "cocotree_prediction_v1",
            "image_id": 1,
            "nodes": [
                {
                    "instance_id": f"{prefix}_car",
                    "semantic_id": f"{prefix}_car_sem",
                    "label": "car",
                    "parent_instance_id": "ROOT",
                    "parent_semantic_id": "ROOT",
                    "segmentation": _rle(
                        [
                            [1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 1, 0, 0],
                        ]
                    ),
                },
                {
                    "instance_id": f"{prefix}_wheel",
                    "semantic_id": f"{prefix}_wheel_sem",
                    "label": "wheel",
                    "parent_instance_id": f"{prefix}_car",
                    "parent_semantic_id": f"{prefix}_car_sem",
                    "segmentation": _rle(
                        [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [1, 0, 0, 0],
                        ]
                    ),
                },
            ],
        }
    )


def run_quality_invariants(protocol: dict[str, Any]) -> dict[str, Any]:
    exact_protocol = dict(protocol)
    exact_protocol["label_similarity"] = {"backend": "exact", "template": "{label}"}
    scorer = ExactLabelSimilarity()
    gt = toy_tree(prediction=False)
    perfect = toy_tree(prediction=True)
    perfect_result, _ = evaluate_image(
        gt,
        perfect,
        label_scorer=scorer,
        protocol=exact_protocol,
    )

    rewired_nodes = [
        (
            replace(
                node,
                parent_instance_id="ROOT",
                parent_semantic_id="ROOT",
                semantic_depth=1,
            )
            if node.label == "wheel"
            else node
        )
        for node in perfect.nodes
    ]
    rewired = ImageTree(image_id=1, nodes=rewired_nodes)
    rewired_result, _ = evaluate_image(
        gt,
        rewired,
        label_scorer=scorer,
        protocol=exact_protocol,
    )

    wrong_label = ImageTree(
        image_id=1,
        nodes=[
            replace(node, label="tire") if node.label == "wheel" else node
            for node in perfect.nodes
        ],
    )
    wrong_label_result, _ = evaluate_image(
        gt,
        wrong_label,
        label_scorer=scorer,
        protocol=exact_protocol,
    )

    missing = ImageTree(
        image_id=1,
        nodes=[node for node in perfect.nodes if node.label != "wheel"],
    )
    missing_result, _ = evaluate_image(
        gt,
        missing,
        label_scorer=scorer,
        protocol=exact_protocol,
    )
    checks = {
        "identity_all_primary_metrics_are_one": all(
            abs(float(perfect_result[field]) - 1.0) < 1e-9
            for field in ("hpq", "otq", "tq", "bq", "mean_nq", "mq", "lq")
        ),
        "parent_rewire_preserves_mq_lq": (
            abs(rewired_result["mq"] - perfect_result["mq"]) < 1e-9
            and abs(rewired_result["lq"] - perfect_result["lq"]) < 1e-9
        ),
        "parent_rewire_lowers_tree_quality": (
            rewired_result["tq"] < perfect_result["tq"]
        ),
        "wrong_label_preserves_matching_and_tq": (
            wrong_label_result["tp"] == perfect_result["tp"]
            and abs(wrong_label_result["tq"] - perfect_result["tq"]) < 1e-9
        ),
        "wrong_label_lowers_lq_and_otq": (
            wrong_label_result["lq"] < perfect_result["lq"]
            and wrong_label_result["otq"] < perfect_result["otq"]
        ),
        "missing_mask_counts_fn_and_lowers_tq": (
            missing_result["fn"] == 1
            and missing_result["tq"] < perfect_result["tq"]
        ),
    }
    return {
        "schema": "cocotree_quality_invariants_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "results": {
            "perfect": perfect_result,
            "parent_rewire": rewired_result,
            "wrong_label": wrong_label_result,
            "missing_mask": missing_result,
        },
    }

