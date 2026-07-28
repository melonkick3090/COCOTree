from __future__ import annotations

from collections import defaultdict
from typing import Any

from .labels import build_label_scorer
from .metrics import derive_instance_parents, evaluate_image, macro_mean
from .models import ImageTree


QUALITY_FIELDS = ("hpq", "otq", "tq", "bq", "mean_nq", "mq", "lq")


def _depth_moments(
    rows: list[dict[str, Any]],
    prefix: str,
) -> tuple[int, float, float]:
    count = sum(int(row[f"num_{prefix}_nodes"]) for row in rows)
    if count <= 0:
        return 0, 0.0, 0.0
    value_sum = sum(
        float(row[f"{prefix}_mean_depth"]) * int(row[f"num_{prefix}_nodes"])
        for row in rows
    )
    value_square_sum = sum(
        (
            float(row[f"{prefix}_depth_variance"])
            + float(row[f"{prefix}_mean_depth"]) ** 2
        )
        * int(row[f"num_{prefix}_nodes"])
        for row in rows
    )
    mean = value_sum / count
    variance = max(0.0, value_square_sum / count - mean**2)
    return count, float(mean), float(variance)


def aggregate_depth_rows(
    per_image_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        grouped[str(row["depth_scope"])].append(row)
    order = {"<=1": 0, "<=2": 1, "<=3": 2, "all": 3}
    output: list[dict[str, Any]] = []
    for depth_scope, rows in sorted(
        grouped.items(),
        key=lambda item: order.get(item[0], 99),
    ):
        gt_count, gt_mean, gt_variance = _depth_moments(rows, "gt")
        pred_count, pred_mean, pred_variance = _depth_moments(rows, "pred")
        record: dict[str, Any] = {
            "depth_scope": depth_scope,
            "num_images": len(rows),
            "num_gt_masks": sum(int(row["num_gt_masks"]) for row in rows),
            "num_pred_masks": sum(int(row["num_pred_masks"]) for row in rows),
            "num_gt_nodes": gt_count,
            "num_pred_nodes": pred_count,
            "gt_max_depth": max((int(row["gt_max_depth"]) for row in rows), default=0),
            "gt_mean_depth": gt_mean,
            "gt_depth_variance": gt_variance,
            "pred_max_depth": max(
                (int(row["pred_max_depth"]) for row in rows),
                default=0,
            ),
            "pred_mean_depth": pred_mean,
            "pred_depth_variance": pred_variance,
            "otq_tp": sum(int(row["tp"]) for row in rows),
            "otq_fp": sum(int(row["fp"]) for row in rows),
            "otq_fn": sum(int(row["fn"]) for row in rows),
        }
        for field in QUALITY_FIELDS:
            output_name = f"mean_{field}" if field != "mean_nq" else "mean_mnq"
            record[output_name] = macro_mean(rows, field)
        output.append(record)
    return output


def evaluate_dataset(
    ground_truth: dict[int, ImageTree],
    predictions: dict[int, ImageTree],
    manifest_ids: list[int],
    *,
    protocol: dict[str, Any],
    label_device: str = "cpu",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_protocol = dict(protocol)
    if (
        str(protocol.get("instance_parent_policy", "submitted"))
        == "derive_first_positive_semantic_ancestor_max_iou"
    ):
        # Every cumulative depth scope keeps a node's semantic ancestors, so
        # canonical instance parents can be materialized once on the full tree
        # and reused without changing the scoped result.
        ground_truth = {
            image_id: derive_instance_parents(tree)
            for image_id, tree in ground_truth.items()
        }
        predictions = {
            image_id: derive_instance_parents(tree)
            for image_id, tree in predictions.items()
        }
        evaluation_protocol["instance_parent_policy"] = "submitted"
    scorer = build_label_scorer(protocol, device=label_device)
    scorer.prepare(
        [
            node.label
            for image_id in manifest_ids
            for tree in (ground_truth[image_id], predictions[image_id])
            for node in tree.nodes
        ]
    )
    per_image_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for raw_scope in protocol.get("depth_scopes", [1, 2, 3, "all"]):
        max_depth = None if str(raw_scope) == "all" else int(raw_scope)
        for image_id in manifest_ids:
            row, image_matches = evaluate_image(
                ground_truth[image_id],
                predictions[image_id],
                label_scorer=scorer,
                protocol=evaluation_protocol,
                max_depth=max_depth,
            )
            per_image_rows.append(row)
            match_rows.extend(image_matches)
    summary_rows = aggregate_depth_rows(per_image_rows)
    return summary_rows, per_image_rows, match_rows
