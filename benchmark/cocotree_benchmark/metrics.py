from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .labels import normalize_label
from .models import (
    ROOT_INSTANCE_ID,
    ROOT_SEMANTIC_ID,
    ImageTree,
    InstanceNode,
    natural_id_key,
)
from .rle import iou_matrix


@dataclass(frozen=True)
class MaskMatch:
    gt_index: int | None
    pred_index: int | None
    gt_instance_id: str
    pred_instance_id: str
    iou: float
    status: str


@dataclass
class SemanticNode:
    semantic_id: str
    label: str
    parent_semantic_id: str
    instance_rles: list[dict[str, Any]]


def _scored_nodes(
    tree: ImageTree,
    excluded_labels: set[str],
) -> list[InstanceNode]:
    return [
        node
        for node in tree.sorted_nodes()
        if normalize_label(node.label) not in excluded_labels
    ]


def derive_instance_parents(tree: ImageTree) -> ImageTree:
    """Materialize the paper mask-parent rule from the semantic tree.

    For each instance, climb direct semantic ancestors. At the first semantic
    level containing any positive-overlap instance, choose the maximum-IoU
    instance. Ties are resolved by the stable instance-id order.
    """

    by_semantic: dict[str, list[InstanceNode]] = defaultdict(list)
    semantic_parent: dict[str, str] = {}
    for node in tree.sorted_nodes():
        by_semantic[node.semantic_id].append(node)
        semantic_parent.setdefault(node.semantic_id, node.parent_semantic_id)
    output: list[InstanceNode] = []
    for node in tree.sorted_nodes():
        selected_parent = ROOT_INSTANCE_ID
        ancestor = node.parent_semantic_id
        visited: set[str] = set()
        while ancestor != ROOT_SEMANTIC_ID and ancestor not in visited:
            visited.add(ancestor)
            candidates = sorted(
                by_semantic.get(ancestor, []),
                key=lambda candidate: natural_id_key(candidate.instance_id),
            )
            if candidates:
                values = iou_matrix(
                    [node.segmentation],
                    [candidate.segmentation for candidate in candidates],
                )[0]
                best_value = float(np.max(values)) if len(values) else 0.0
                if best_value > 0.0:
                    best_index = next(
                        index
                        for index, value in enumerate(values)
                        if float(value) == best_value
                    )
                    selected_parent = candidates[best_index].instance_id
                    break
            ancestor = semantic_parent.get(ancestor, ROOT_SEMANTIC_ID)
        output.append(replace(node, parent_instance_id=selected_parent))
    return ImageTree(image_id=tree.image_id, nodes=output)


def global_mask_matches(
    gt_nodes: list[InstanceNode],
    pred_nodes: list[InstanceNode],
    *,
    iou_threshold: float,
) -> tuple[list[MaskMatch], list[MaskMatch], np.ndarray]:
    matrix = iou_matrix(
        [node.segmentation for node in gt_nodes],
        [node.segmentation for node in pred_nodes],
    )
    matches: list[MaskMatch] = []
    tp_matches: list[MaskMatch] = []
    assigned_gt: set[int] = set()
    assigned_pred: set[int] = set()

    if gt_nodes and pred_nodes:
        gt_assignment, pred_assignment = linear_sum_assignment(1.0 - matrix)
        for gt_raw, pred_raw in zip(
            gt_assignment,
            pred_assignment,
            strict=False,
        ):
            gt_index = int(gt_raw)
            pred_index = int(pred_raw)
            assigned_gt.add(gt_index)
            assigned_pred.add(pred_index)
            value = float(matrix[gt_index, pred_index])
            status = "tp" if value >= float(iou_threshold) else "below_threshold"
            match = MaskMatch(
                gt_index=gt_index,
                pred_index=pred_index,
                gt_instance_id=gt_nodes[gt_index].instance_id,
                pred_instance_id=pred_nodes[pred_index].instance_id,
                iou=value,
                status=status,
            )
            matches.append(match)
            if status == "tp":
                tp_matches.append(match)

    for gt_index, node in enumerate(gt_nodes):
        if gt_index not in assigned_gt:
            matches.append(
                MaskMatch(
                    gt_index=gt_index,
                    pred_index=None,
                    gt_instance_id=node.instance_id,
                    pred_instance_id="",
                    iou=0.0,
                    status="unmatched_gt",
                )
            )
    for pred_index, node in enumerate(pred_nodes):
        if pred_index not in assigned_pred:
            matches.append(
                MaskMatch(
                    gt_index=None,
                    pred_index=pred_index,
                    gt_instance_id="",
                    pred_instance_id=node.instance_id,
                    iou=0.0,
                    status="unmatched_pred",
                )
            )
    return matches, tp_matches, matrix


def ancestor_chain(
    parent_by_id: dict[str, str],
    node_id: str,
    *,
    root_id: str,
) -> list[str]:
    chain: list[str] = []
    current = str(node_id)
    seen: set[str] = set()
    while current and current not in seen:
        chain.append(current)
        seen.add(current)
        if current == root_id:
            break
        current = str(parent_by_id.get(current, root_id))
    if root_id not in chain:
        chain.append(root_id)
    return chain


def matched_skeleton_parent_map(
    parent_by_id: dict[str, str],
    matched_ids: set[str],
    *,
    root_id: str,
) -> dict[str, str]:
    skeleton = {root_id: ""}
    for node_id in sorted(matched_ids, key=natural_id_key):
        parent = root_id
        for ancestor in ancestor_chain(
            parent_by_id,
            node_id,
            root_id=root_id,
        )[1:]:
            if ancestor in matched_ids or ancestor == root_id:
                parent = ancestor
                break
        skeleton[node_id] = parent
    return skeleton


def skeleton_lca(
    skeleton_parent: dict[str, str],
    left: str,
    right: str,
    *,
    root_id: str,
) -> str:
    left_chain = ancestor_chain(
        skeleton_parent,
        left,
        root_id=root_id,
    )
    right_chain = set(
        ancestor_chain(
            skeleton_parent,
            right,
            root_id=root_id,
        )
    )
    for node_id in left_chain:
        if node_id in right_chain:
            return node_id
    return root_id


def branch_pair_accuracy(
    gt_tree: ImageTree,
    pred_tree: ImageTree,
    tp_matches: list[MaskMatch],
    *,
    empty_value: float,
) -> tuple[float, int, int]:
    pair_count = len(tp_matches) * (len(tp_matches) - 1) // 2
    if pair_count == 0:
        return float(empty_value), 0, 0
    gt_parent = {ROOT_INSTANCE_ID: ""}
    gt_parent.update(
        {node.instance_id: node.parent_instance_id for node in gt_tree.nodes}
    )
    pred_parent = {ROOT_INSTANCE_ID: ""}
    pred_parent.update(
        {node.instance_id: node.parent_instance_id for node in pred_tree.nodes}
    )
    gt_to_pred = {
        match.gt_instance_id: match.pred_instance_id
        for match in tp_matches
    }
    pred_to_gt = {
        match.pred_instance_id: match.gt_instance_id
        for match in tp_matches
    }
    gt_skeleton = matched_skeleton_parent_map(
        gt_parent,
        set(gt_to_pred),
        root_id=ROOT_INSTANCE_ID,
    )
    pred_skeleton = matched_skeleton_parent_map(
        pred_parent,
        set(pred_to_gt),
        root_id=ROOT_INSTANCE_ID,
    )
    correct = 0
    for left_index in range(len(tp_matches)):
        for right_index in range(left_index + 1, len(tp_matches)):
            left = tp_matches[left_index]
            right = tp_matches[right_index]
            gt_lca = skeleton_lca(
                gt_skeleton,
                left.gt_instance_id,
                right.gt_instance_id,
                root_id=ROOT_INSTANCE_ID,
            )
            pred_lca = skeleton_lca(
                pred_skeleton,
                left.pred_instance_id,
                right.pred_instance_id,
                root_id=ROOT_INSTANCE_ID,
            )
            if gt_lca == ROOT_INSTANCE_ID and pred_lca == ROOT_INSTANCE_ID:
                correct += 1
                continue
            if (
                gt_to_pred.get(gt_lca) == pred_lca
                and pred_to_gt.get(pred_lca) == gt_lca
            ):
                correct += 1
    return float(correct / pair_count), int(pair_count), int(correct)


def evaluate_otq(
    gt_tree: ImageTree,
    pred_tree: ImageTree,
    *,
    label_scorer: Any,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parent_policy = str(protocol.get("instance_parent_policy", "submitted"))
    if parent_policy == "derive_first_positive_semantic_ancestor_max_iou":
        gt_tree = derive_instance_parents(gt_tree)
        pred_tree = derive_instance_parents(pred_tree)
    elif parent_policy != "submitted":
        raise ValueError(f"Unsupported instance_parent_policy: {parent_policy}")
    excluded = {
        normalize_label(label)
        for label in protocol.get("otq_excluded_normalized_labels", [])
    }
    gt_nodes = _scored_nodes(gt_tree, excluded)
    pred_nodes = _scored_nodes(pred_tree, excluded)
    label_scorer.prepare(
        [node.label for node in gt_nodes] + [node.label for node in pred_nodes]
    )
    matches, tp_matches, _matrix = global_mask_matches(
        gt_nodes,
        pred_nodes,
        iou_threshold=float(protocol["mask_iou_threshold"]),
    )
    tp = len(tp_matches)
    fp = len(pred_nodes) - tp
    fn = len(gt_nodes) - tp
    denominator = float(tp) + 0.5 * float(fp) + 0.5 * float(fn)
    recovery = float(tp / denominator) if denominator > 0.0 else 0.0

    gt_by_id = {node.instance_id: node for node in gt_nodes}
    pred_by_id = {node.instance_id: node for node in pred_nodes}
    sum_iou = 0.0
    sum_label = 0.0
    sum_node_quality = 0.0
    similarity_by_pair: dict[tuple[str, str], float] = {}
    for match in tp_matches:
        gt_node = gt_by_id[match.gt_instance_id]
        pred_node = pred_by_id[match.pred_instance_id]
        similarity = float(label_scorer.similarity(gt_node.label, pred_node.label))
        similarity_by_pair[(match.gt_instance_id, match.pred_instance_id)] = similarity
        sum_iou += match.iou
        sum_label += similarity
        sum_node_quality += match.iou * similarity
    mq = float(sum_iou / tp) if tp else 0.0
    lq = float(sum_label / tp) if tp else 0.0
    mean_nq = float(sum_node_quality / tp) if tp else 0.0
    bq, branch_pairs, correct_branch_pairs = branch_pair_accuracy(
        gt_tree,
        pred_tree,
        tp_matches,
        empty_value=float(protocol.get("bq_empty_pair_accuracy", 1.0)),
    )
    tq = float(bq * recovery)
    otq = float(tq * mean_nq)

    match_rows: list[dict[str, Any]] = []
    for match in matches:
        similarity = similarity_by_pair.get(
            (match.gt_instance_id, match.pred_instance_id),
            0.0,
        )
        match_rows.append(
            {
                "gt_instance_id": match.gt_instance_id,
                "pred_instance_id": match.pred_instance_id,
                "status": match.status,
                "iou": float(match.iou),
                "label_similarity": float(similarity),
                "node_quality": (
                    float(match.iou * similarity)
                    if match.status == "tp"
                    else 0.0
                ),
                "deducts_fp": int(match.status in {"below_threshold", "unmatched_pred"}),
                "deducts_fn": int(match.status in {"below_threshold", "unmatched_gt"}),
            }
        )
    return {
        "otq": otq,
        "tq": tq,
        "bq": float(bq),
        "mean_nq": mean_nq,
        "mq": mq,
        "lq": lq,
        "mask_recovery": recovery,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "num_gt_masks": len(gt_nodes),
        "num_pred_masks": len(pred_nodes),
        "num_branch_pairs": branch_pairs,
        "num_correct_branch_pairs": correct_branch_pairs,
    }, match_rows


def build_semantic_nodes(
    tree: ImageTree,
    excluded_labels: set[str],
    *,
    others_policy: str,
) -> tuple[dict[str, SemanticNode], dict[str, list[str]]]:
    grouped: dict[str, list[InstanceNode]] = defaultdict(list)
    for node in tree.nodes:
        if normalize_label(node.label) not in excluded_labels:
            grouped[node.semantic_id].append(node)
    all_semantic_parent = {
        node.semantic_id: node.parent_semantic_id
        for node in tree.nodes
    }
    top_level_others = {
        semantic_id
        for semantic_id, instances in grouped.items()
        if normalize_label(instances[0].label) == "others"
        and instances[0].parent_semantic_id == ROOT_SEMANTIC_ID
    }
    nodes: dict[str, SemanticNode] = {}
    for semantic_id, instances in grouped.items():
        first = instances[0]
        parent = first.parent_semantic_id
        if (
            others_policy == "keep_top_bucket_promote_direct_children"
            and parent in top_level_others
        ):
            parent = ROOT_SEMANTIC_ID
        while (
            parent != ROOT_SEMANTIC_ID
            and parent not in grouped
        ):
            parent = all_semantic_parent.get(parent, ROOT_SEMANTIC_ID)
        nodes[semantic_id] = SemanticNode(
            semantic_id=semantic_id,
            label=normalize_label(first.label),
            parent_semantic_id=parent,
            instance_rles=[node.segmentation for node in instances],
        )
    children: dict[str, list[str]] = defaultdict(list)
    children[ROOT_SEMANTIC_ID] = []
    for semantic_id, node in nodes.items():
        children[node.parent_semantic_id].append(semantic_id)
    for parent_id in children:
        children[parent_id].sort(key=natural_id_key)
    return nodes, dict(children)


def semantic_leaf_score(
    gt_node: SemanticNode,
    pred_node: SemanticNode,
) -> float:
    if not gt_node.instance_rles or not pred_node.instance_rles:
        return 0.0
    matrix = iou_matrix(gt_node.instance_rles, pred_node.instance_rles)
    gt_assignment, pred_assignment = linear_sum_assignment(1.0 - matrix)
    if len(gt_assignment) == 0:
        return 0.0
    return float(
        sum(
            float(matrix[int(gt_index), int(pred_index)])
            for gt_index, pred_index in zip(
                gt_assignment,
                pred_assignment,
                strict=False,
            )
        )
        / len(gt_assignment)
    )


def evaluate_hpq(
    gt_tree: ImageTree,
    pred_tree: ImageTree,
    *,
    protocol: dict[str, Any],
) -> float:
    excluded = {
        normalize_label(label)
        for label in protocol.get("hpq_excluded_normalized_labels", [])
    }
    others_policy = str(protocol.get("hpq_others_policy", "preserve"))
    gt_nodes, gt_children = build_semantic_nodes(
        gt_tree,
        excluded,
        others_policy=others_policy,
    )
    pred_nodes, pred_children = build_semantic_nodes(
        pred_tree,
        excluded,
        others_policy=others_policy,
    )
    threshold = float(protocol.get("hpq_match_threshold", 0.5))
    memo: dict[tuple[str, str], float] = {}

    def grouped(
        nodes: dict[str, SemanticNode],
        children: dict[str, list[str]],
        parent_id: str,
    ) -> dict[str, list[str]]:
        by_label: dict[str, list[str]] = defaultdict(list)
        for child_id in children.get(parent_id, []):
            by_label[nodes[child_id].label].append(child_id)
        return dict(by_label)

    def node_score(gt_id: str, pred_id: str) -> float:
        key = (gt_id, pred_id)
        if key in memo:
            return memo[key]
        gt_node = gt_nodes.get(gt_id)
        pred_node = pred_nodes.get(pred_id)
        if (
            gt_node is None
            or pred_node is None
            or gt_node.label != pred_node.label
        ):
            memo[key] = 0.0
            return 0.0
        gt_child_ids = gt_children.get(gt_id, [])
        pred_child_ids = pred_children.get(pred_id, [])
        if not gt_child_ids and not pred_child_ids:
            score = semantic_leaf_score(gt_node, pred_node)
        else:
            score = parent_score(gt_id, pred_id)
        memo[key] = float(score)
        return float(score)

    def parent_score(gt_parent: str, pred_parent: str) -> float:
        gt_by_label = grouped(gt_nodes, gt_children, gt_parent)
        pred_by_label = grouped(pred_nodes, pred_children, pred_parent)
        active_labels = sorted(set(gt_by_label) | set(pred_by_label))
        if not active_labels:
            return 0.0
        class_scores: list[float] = []
        for label in active_labels:
            gt_ids = gt_by_label.get(label, [])
            pred_ids = pred_by_label.get(label, [])
            if not gt_ids or not pred_ids:
                class_scores.append(0.0)
                continue
            score_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
            for gt_index, gt_id in enumerate(gt_ids):
                for pred_index, pred_id in enumerate(pred_ids):
                    score_matrix[gt_index, pred_index] = node_score(gt_id, pred_id)
            gt_assignment, pred_assignment = linear_sum_assignment(
                1.0 - score_matrix
            )
            tp_scores = [
                float(score_matrix[int(gt_index), int(pred_index)])
                for gt_index, pred_index in zip(
                    gt_assignment,
                    pred_assignment,
                    strict=False,
                )
                if float(score_matrix[int(gt_index), int(pred_index)]) >= threshold
            ]
            tp = len(tp_scores)
            fp = len(pred_ids) - tp
            fn = len(gt_ids) - tp
            denominator = float(tp) + 0.5 * float(fp) + 0.5 * float(fn)
            class_scores.append(
                float(sum(tp_scores) / denominator)
                if denominator > 0.0
                else 0.0
            )
        return float(sum(class_scores) / len(class_scores))

    return parent_score(ROOT_SEMANTIC_ID, ROOT_SEMANTIC_ID)


def effective_semantic_depths(
    tree: ImageTree,
    *,
    others_policy: str,
) -> dict[str, int]:
    label_by_semantic: dict[str, str] = {}
    parent_by_semantic: dict[str, str] = {}
    for node in tree.nodes:
        label_by_semantic.setdefault(node.semantic_id, normalize_label(node.label))
        parent_by_semantic.setdefault(node.semantic_id, node.parent_semantic_id)
    if others_policy == "keep_top_bucket_promote_direct_children":
        top_level_others = {
            semantic_id
            for semantic_id, label in label_by_semantic.items()
            if label == "others"
            and parent_by_semantic.get(semantic_id) == ROOT_SEMANTIC_ID
        }
        parent_by_semantic = {
            semantic_id: (
                ROOT_SEMANTIC_ID
                if parent_id in top_level_others
                else parent_id
            )
            for semantic_id, parent_id in parent_by_semantic.items()
        }
    depths = {ROOT_SEMANTIC_ID: 0}

    def depth(semantic_id: str) -> int:
        if semantic_id in depths:
            return depths[semantic_id]
        parent_id = parent_by_semantic.get(semantic_id, ROOT_SEMANTIC_ID)
        value = depth(parent_id) + 1
        depths[semantic_id] = value
        return value

    for semantic_id in parent_by_semantic:
        depth(semantic_id)
    return depths


def scoped_tree(
    tree: ImageTree,
    max_depth: int | None,
    *,
    others_policy: str,
) -> ImageTree:
    # Canonical depth scopes are selected from raw/direct semantic depth first.
    # Legacy top-level "others" promotion is applied only to that selected view.
    selected = ImageTree(
        image_id=tree.image_id,
        nodes=[
            node
            for node in tree.nodes
            if max_depth is None
            or 1 <= node.semantic_depth <= int(max_depth)
        ],
    )
    effective_depth = effective_semantic_depths(
        selected,
        others_policy=others_policy,
    )
    return ImageTree(
        image_id=selected.image_id,
        nodes=[
            replace(node, semantic_depth=effective_depth[node.semantic_id])
            for node in selected.nodes
        ],
    )


def semantic_depth_stats(
    tree: ImageTree,
    *,
    excluded_labels: set[str],
) -> dict[str, float | int]:
    depth_by_semantic: dict[str, int] = {}
    for node in tree.nodes:
        if normalize_label(node.label) in excluded_labels:
            continue
        depth_by_semantic.setdefault(node.semantic_id, node.semantic_depth)
    values = list(depth_by_semantic.values())
    if not values:
        return {"count": 0, "max": 0, "mean": 0.0, "variance": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "max": int(max(values)),
        "mean": float(array.mean()),
        "variance": float(array.var()),
    }


def evaluate_image(
    gt_tree: ImageTree,
    pred_tree: ImageTree,
    *,
    label_scorer: Any,
    protocol: dict[str, Any],
    max_depth: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    others_policy = str(protocol.get("hpq_others_policy", "preserve"))
    gt_scope = scoped_tree(
        gt_tree,
        max_depth,
        others_policy=others_policy,
    )
    pred_scope = scoped_tree(
        pred_tree,
        max_depth,
        others_policy=others_policy,
    )
    otq, match_rows = evaluate_otq(
        gt_scope,
        pred_scope,
        label_scorer=label_scorer,
        protocol=protocol,
    )
    hpq = evaluate_hpq(gt_scope, pred_scope, protocol=protocol)
    hpq_excluded = {
        normalize_label(label)
        for label in protocol.get("hpq_excluded_normalized_labels", [])
    }
    gt_depth = semantic_depth_stats(gt_scope, excluded_labels=hpq_excluded)
    pred_depth = semantic_depth_stats(pred_scope, excluded_labels=hpq_excluded)
    result = {
        "image_id": gt_tree.image_id,
        "depth_scope": "all" if max_depth is None else f"<={int(max_depth)}",
        "hpq": float(hpq),
        **otq,
        "num_gt_nodes": int(gt_depth["count"]),
        "num_pred_nodes": int(pred_depth["count"]),
        "gt_max_depth": int(gt_depth["max"]),
        "gt_mean_depth": float(gt_depth["mean"]),
        "gt_depth_variance": float(gt_depth["variance"]),
        "pred_max_depth": int(pred_depth["max"]),
        "pred_mean_depth": float(pred_depth["mean"]),
        "pred_depth_variance": float(pred_depth["variance"]),
    }
    for row in match_rows:
        row["image_id"] = gt_tree.image_id
        row["depth_scope"] = result["depth_scope"]
    return result, match_rows


def macro_mean(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(sum(values) / len(values)) if values else 0.0
