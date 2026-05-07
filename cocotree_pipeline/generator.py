from __future__ import annotations

import gc
import os
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from . import mask_utils as U
from . import prompts as prompts
from .image_utils import (
    apply_l_clahe_pil,
    image_to_base64_jpeg,
    prepare_llm_crop,
    render_masks_only_canvas,
    resize_longest_side,
    save_candidates_score_overlay_full,
    save_subcrop_debug,
    union_fill_minus_boundary,
    visualize_union_on_image,
)
from .io_utils import (
    bbox_to_dict,
    build_basic_provenance,
    mask_to_coco_rle,
    node_dir_from_path,
    render_tree_text,
    reset_dir,
    sanitize_filename,
    save_json,
    save_json_gz,
    save_jsonl_gz,
    save_mask_png,
    save_node_masks,
    save_tree_summary_provenance,
    sha256_file,
    utc_now_str,
    write_done_marker,
)
from .llm_openrouter import (
    dedupe_texts,
    parse_llm_batch_mapping,
    parse_llm_single_prompts,
    parse_llm_tool_call,
    send_openrouter_with_retry,
)
from .mask_postprocess import (
    compute_pp_params_from_area,
    postprocess_mask_u8,
    split_parent_into_subcrops,
)
from .sam3_backend import Sam3Backend
from .types import GeneratorConfig, NodeInfo, RunResult


NODE_ID_WIDTH = 5

def _semantic_path(path: List[str]) -> List[str]:
    if len(path) >= 2 and path[1] == "others":
        return [path[0]] + path[2:]
    return path


def _raw_depth(path: List[str]) -> int:
    return max(0, len(path) - 1)


def _semantic_depth(path: List[str]) -> int:
    return max(0, len(_semantic_path(path)) - 1)


def _is_root_others_node(path: List[str], label: str) -> bool:
    return label == "others" and len(path) == 2 and path[0] == "root" and path[1] == "others"


def _align_scores(scores: Any, n_masks: int) -> List[Optional[float]]:
    if n_masks <= 0:
        return []
    if isinstance(scores, list):
        arr = list(scores)
    elif isinstance(scores, tuple):
        arr = list(scores)
    elif scores is None:
        arr = []
    else:
        arr = [scores]

    if len(arr) < n_masks:
        arr += [None] * (n_masks - len(arr))
    else:
        arr = arr[:n_masks]

    out: List[Optional[float]] = []
    for sc in arr:
        if sc is None:
            out.append(None)
            continue
        try:
            out.append(float(sc))
        except Exception:
            out.append(None)
    return out


class COCOTreeGenerator:
    """
    COCOTree generator.

    Policy:
    - SAM capture threshold is loose: 0.3
    - SAM internal mask threshold stays inside SAM processor/backend (assumed 0.5)
    - final selection is per-candidate
    - root/init selection threshold: 0.5
    - child/subcrop selection threshold:
        * processed candidate area <= 5% of full image area -> 0.4
        * otherwise -> 0.5
    """

    def __init__(
        self,
        cfg: Optional[GeneratorConfig] = None,
        *,
        models_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        sam_capture_threshold: float = 0.3,
        root_selection_threshold: float = 0.5,
        child_selection_threshold_default: float = 0.5,
        child_selection_threshold_small: float = 0.4,
        child_small_area_pct_of_full_image: float = 5.0,
    ) -> None:
        self.cfg = cfg or GeneratorConfig()

        # Keep the released pipeline policy fixed unless explicitly changed.
        self.sam_capture_threshold = float(
            getattr(self.cfg, "sam_capture_threshold", sam_capture_threshold)
        )
        self.root_selection_threshold = float(
            getattr(self.cfg, "root_selection_threshold", root_selection_threshold)
        )
        self.child_selection_threshold_default = float(
            getattr(self.cfg, "child_selection_threshold_default", child_selection_threshold_default)
        )
        self.child_selection_threshold_small = float(
            getattr(self.cfg, "child_selection_threshold_small", child_selection_threshold_small)
        )
        self.child_small_area_pct_of_full_image = float(
            getattr(
                self.cfg,
                "child_small_area_pct_of_full_image",
                child_small_area_pct_of_full_image,
            )
        )
        self.child_small_area_ratio_of_full_image = (
            self.child_small_area_pct_of_full_image / 100.0
        )

        # self.cfg.selection_mode = "per_candidate"

        self.backend = Sam3Backend(models_dir=models_dir, checkpoint_path=checkpoint_path)
        self._sam3_processor = None
        self._last_capture_threshold: Optional[float] = None
        self._package_dir = os.path.dirname(os.path.abspath(__file__))
        self._reset_run_state()

    def _reset_run_state(self) -> None:
        self.t_llm = 0.0
        self.t_sam = 0.0
        self.t_io = 0.0

        self.llm_stage_counts: Counter[str] = Counter()
        self.sam_call_count = 0

        self.llm_batch_parse_fail = 0
        self.llm_missing_nodes = 0
        self.llm_single_fallback_nodes = 0
        self.llm_malformed_nodes = 0

        self.node_meta: Dict[str, Dict[str, Any]] = {}
        self.llm_call_records: List[Dict[str, Any]] = []
        self.prompt_decision_rows: List[Dict[str, Any]] = []
        self.root_initial_audit: Dict[str, Any] = {}
        self._llm_call_id = 0

    def _next_llm_call_id(self) -> int:
        self._llm_call_id += 1
        return int(self._llm_call_id)

    def _load_processor(self):
        need_reload = (
            self._sam3_processor is None
            or self._last_capture_threshold != float(self.sam_capture_threshold)
        )
        if need_reload:
            self._sam3_processor = self.backend.load(
                confidence_threshold=float(self.sam_capture_threshold),
            )
            self._last_capture_threshold = float(self.sam_capture_threshold)
        return self._sam3_processor

    def _call_llm(
        self,
        *,
        api_key: str,
        model_name: str,
        messages: List[Dict[str, Any]],
        tag: str,
        stage: str,
        temperature: float,
        max_retry: int,
        node_ids: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int], int]:
        t0 = time.perf_counter()
        self.llm_stage_counts[stage] += 1
        call_id = self._next_llm_call_id()
        resp, usage = send_openrouter_with_retry(
            api_key,
            model_name,
            messages,
            max_retry=max_retry,
            tag=tag,
            temperature=temperature,
        )
        self.t_llm += time.perf_counter() - t0
        record = {
            "call_id": int(call_id),
            "stage": stage,
            "tag": tag,
            "temperature": float(temperature),
            "node_ids": list(node_ids or []),
            "usage": usage,
            "created_at_utc": utc_now_str(),
        }
        if extra:
            record.update(extra)
        if self.cfg.save_llm_raw_responses:
            record["raw_response"] = resp
        self.llm_call_records.append(record)
        return resp or "", usage or {}, call_id

    def _run_sam_on_crop(
        self,
        processor,
        pil_img: Image.Image,
        prompts_in: List[str],
    ) -> Dict[str, Any]:
        prompts_uniq = dedupe_texts(prompts_in)
        if not prompts_uniq:
            return {
                "orig_img_h": int(pil_img.height),
                "orig_img_w": int(pil_img.width),
                "label_results": {},
            }
        t0 = time.perf_counter()
        self.sam_call_count += 1
        out = self.backend.run_on_crop(processor, pil_img, prompts_uniq)
        self.t_sam += time.perf_counter() - t0
        return out

    def _build_depth_batch_messages(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ids = [it["node_id"] for it in items]
        ids_csv = ", ".join(ids)
        header = prompts.LOCAL_DECOMPOSITION_BATCH_MESSAGE.format(
            max_children=int(self.cfg.max_children)
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": header}]
        content.append(
            {
                "type": "text",
                "text": f"Provided ids (MUST include all exactly once): {ids_csv}\n",
            }
        )

        for idx, item in enumerate(items, start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"\nITEM {idx}\n"
                        f"node_id: {item['node_id']}\n"
                        f"path: {item['path_text']}\n"
                        f"current_label: {item['label']}\n"
                    ),
                }
            )
            content.append({"type": "image_url", "image_url": {"url": item["image_b64"]}})

        return [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _llm_single_suggest(
        self,
        *,
        api_key: str,
        model_name: str,
        local_b64: str,
        path_text: str,
        focus_label: str,
        node_id: str,
    ) -> Tuple[List[str], List[int], bool]:
        if self.cfg.max_children <= 0:
            return [], [], False

        local_text = prompts.LOCAL_DECOMPOSITION_USER_MESSAGE.format(
            path_text=path_text,
            max_children=int(self.cfg.max_children),
            current_label=focus_label,
        )
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": local_b64}},
                    {"type": "text", "text": local_text},
                ],
            },
        ]

        resp, _usage, call_id = self._call_llm(
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            tag=f"single_focus={focus_label}",
            stage="single",
            temperature=float(self.cfg.single_temperature),
            max_retry=int(self.cfg.llm_max_retry),
            node_ids=[node_id],
            extra={"path_text": path_text, "focus_label": focus_label},
        )
        prompts_out = parse_llm_single_prompts(resp)
        fix_used = False
        call_ids = [int(call_id)]

        if not prompts_out:
            fix_text = (
                "FORMAT FIX: Return ONLY one <tool>...</tool> block.\n"
                "Inside <tool>, output STRICT JSON with schema:\n"
                "<tool>{\"name\":\"propose_prompts\",\"parameters\":{\"text_prompts\":[]}}</tool>\n"
                "No other text, no markdown."
            )
            fix_messages = list(messages) + [
                {"role": "user", "content": [{"type": "text", "text": fix_text}]}
            ]
            resp2, _usage2, call_id2 = self._call_llm(
                api_key=api_key,
                model_name=model_name,
                messages=fix_messages,
                tag=f"single_focus={focus_label}_fix",
                stage="fix",
                temperature=float(self.cfg.fix_temperature),
                max_retry=int(self.cfg.llm_max_retry),
                node_ids=[node_id],
                extra={
                    "path_text": path_text,
                    "focus_label": focus_label,
                    "fix_for": int(call_id),
                },
            )
            call_ids.append(int(call_id2))
            prompts_out = parse_llm_single_prompts(resp2)
            fix_used = True

        return dedupe_texts(prompts_out, max_items=int(self.cfg.max_children)), call_ids, fix_used

    def _batch_request(
        self,
        *,
        api_key: str,
        model_name: str,
        items: List[Dict[str, Any]],
        tag_base: str,
        stage: str,
    ) -> Tuple[Dict[str, List[str]], List[int], bool]:
        messages = self._build_depth_batch_messages(items)
        resp, _usage, call_id = self._call_llm(
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            tag=tag_base,
            stage=stage,
            temperature=float(
                self.cfg.batch_temperature if stage != "fix" else self.cfg.fix_temperature
            ),
            max_retry=int(self.cfg.llm_max_retry),
            node_ids=[it["node_id"] for it in items],
            extra={"path_texts": [it["path_text"] for it in items]},
        )
        mapping = parse_llm_batch_mapping(resp)
        call_ids = [int(call_id)]
        fix_used = False

        if not mapping:
            ids_csv = ", ".join(it["node_id"] for it in items)
            fix_text = (
                "FORMAT FIX: Your previous response violated the tool contract.\n"
                "Return ONLY one <tool>...</tool> block.\n"
                "Inside <tool>, output STRICT JSON with schema:\n"
                "<tool>{\"name\":\"propose_prompts\",\"parameters\":{\"items\":[{\"id\":\"n00001\",\"text_prompts\":[\"p1\",\"p2\"]}]}}</tool>\n"
                f"You MUST include ALL ids exactly once: {ids_csv}\n"
                "No other text, no markdown."
            )
            fix_messages = list(messages) + [
                {"role": "user", "content": [{"type": "text", "text": fix_text}]}
            ]
            resp2, _usage2, call_id2 = self._call_llm(
                api_key=api_key,
                model_name=model_name,
                messages=fix_messages,
                tag=f"{tag_base}_fix",
                stage="fix",
                temperature=float(self.cfg.fix_temperature),
                max_retry=int(self.cfg.llm_max_retry),
                node_ids=[it["node_id"] for it in items],
                extra={
                    "path_texts": [it["path_text"] for it in items],
                    "fix_for": int(call_id),
                },
            )
            call_ids.append(int(call_id2))
            mapping = parse_llm_batch_mapping(resp2)
            fix_used = True

        if not mapping:
            self.llm_batch_parse_fail += 1
        return mapping or {}, call_ids, fix_used

    def _get_mapping_resilient(
        self,
        *,
        api_key: str,
        model_name: str,
        items: List[Dict[str, Any]],
        tag_base: str,
        level: int = 0,
        stage: str = "batch",
    ) -> Dict[str, Dict[str, Any]]:
        expected_ids = [it["node_id"] for it in items]
        expected_set = set(expected_ids)

        mapping_raw, call_ids, fix_used = self._batch_request(
            api_key=api_key,
            model_name=model_name,
            items=items,
            tag_base=tag_base,
            stage=stage,
        )
        mapping = {
            k: dedupe_texts(v, max_items=int(self.cfg.max_children))
            for k, v in mapping_raw.items()
            if k in expected_set
        }
        out: Dict[str, Dict[str, Any]] = {}
        for nid, prompts_out in mapping.items():
            out[nid] = {
                "prompts": prompts_out,
                "llm_mode": stage,
                "call_ids": list(call_ids),
                "fix_used": bool(fix_used),
                "missing_from_batch": False,
            }

        if mapping and all(nid in mapping for nid in expected_ids):
            return out

        if len(items) <= 1:
            return out

        if mapping:
            missing_items = [it for it in items if it["node_id"] not in mapping]
            fill = self._get_mapping_resilient(
                api_key=api_key,
                model_name=model_name,
                items=missing_items,
                tag_base=f"{tag_base}_miss{level}",
                level=level + 1,
                stage="mini",
            )
            for nid, meta in fill.items():
                meta = dict(meta)
                meta["missing_from_batch"] = True
                out[nid] = meta
            return {k: v for k, v in out.items() if k in expected_set}

        mid = len(items) // 2
        left = self._get_mapping_resilient(
            api_key=api_key,
            model_name=model_name,
            items=items[:mid],
            tag_base=f"{tag_base}_A{level}",
            level=level + 1,
            stage="mini",
        )
        right = self._get_mapping_resilient(
            api_key=api_key,
            model_name=model_name,
            items=items[mid:],
            tag_base=f"{tag_base}_B{level}",
            level=level + 1,
            stage="mini",
        )
        left.update(right)
        return {k: v for k, v in left.items() if k in expected_set}

    @staticmethod
    def _pop_semantic_batch(
        queue: Deque[str],
        nodes: Dict[str, NodeInfo],
        batch_size: int,
    ) -> Tuple[Optional[int], List[str]]:
        valid_depths = [_semantic_depth(nodes[nid].path) for nid in queue if nid in nodes]
        if not valid_depths:
            return None, []
        target_depth = min(valid_depths)
        kept = deque()
        batch: List[str] = []
        while queue:
            nid = queue.popleft()
            node = nodes.get(nid)
            if node is None:
                continue
            depth = _semantic_depth(node.path)
            if depth == target_depth and len(batch) < int(batch_size):
                batch.append(nid)
            else:
                kept.append(nid)
        queue.extend(kept)
        return target_depth, batch

    @staticmethod
    def _dedupe_by_containment(
        items: List[Dict[str, Any]],
        *,
        contain_thr: float,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, float]]]:
        if not items:
            return [], []

        valid_idxs: List[int] = []
        areas: List[int] = []
        for i, item in enumerate(items):
            mask = item.get("merged_u8")
            # area = int((mask > 0).sum()) if mask is not None else 0
            area = int(np.count_nonzero(mask)) if mask is not None else 0
            if area > 0:
                valid_idxs.append(i)
                areas.append(area)
        if not valid_idxs:
            return [], []

        order = sorted(range(len(valid_idxs)), key=lambda k: areas[k], reverse=True)
        kept_orig_idxs: List[int] = []
        dropped: List[Tuple[int, int, float]] = []

        for ord_i in order:
            i = valid_idxs[ord_i]
            small_u8 = U.ensure_u8(items[i]["merged_u8"])
            is_contained = False
            for j in kept_orig_idxs:
                big_u8 = U.ensure_u8(items[j]["merged_u8"])
                ratio = float(U.containment_ratio(small_u8, big_u8))
                if ratio >= float(contain_thr):
                    dropped.append((i, j, ratio))
                    is_contained = True
                    break
            if not is_contained:
                kept_orig_idxs.append(i)

        return [items[i] for i in kept_orig_idxs], dropped

    def _selection_threshold_for_candidate(
        self,
        *,
        source: str,
        processed_area_px: int,
        full_area_px: Optional[int],
    ) -> float:
        source_norm = str(source or "").strip().lower()
        if source_norm == "init":
            return float(self.root_selection_threshold)

        if source_norm == "child":
            if full_area_px is not None and int(full_area_px) > 0:
                area_ratio = float(processed_area_px) / float(full_area_px)
                if area_ratio <= float(self.child_small_area_ratio_of_full_image):
                    return float(self.child_selection_threshold_small)
            return float(self.child_selection_threshold_default)

        return float(self.root_selection_threshold)

    def _select_candidates_by_policy(
        self,
        *,
        candidates: List[Tuple[Optional[float], np.ndarray, U.BBox]],
        prompt_record: Dict[str, Any],
    ) -> List[Tuple[Optional[float], np.ndarray, U.BBox]]:
        mode = str(self.cfg.selection_mode).strip().lower()
        if mode != "per_candidate":
            raise ValueError(
                f"COCOTree pipeline expects per_candidate selection_mode, got: {self.cfg.selection_mode}"
            )

        source = str(prompt_record.get("source", "")).strip().lower()
        full_area_px = prompt_record.get("full_area_px", None)
        if full_area_px is not None:
            try:
                full_area_px = int(full_area_px)
            except Exception:
                full_area_px = None

        scores_all: List[float] = []
        for score, _mask, _sb in candidates:
            if score is not None:
                scores_all.append(float(score))

        prompt_record["cand_scored"] = len(scores_all)
        prompt_record["median_score"] = (
            None if not scores_all
            else float(np.median(np.asarray(scores_all, dtype=np.float32)))
        )
        prompt_record["sam_capture_threshold"] = float(self.sam_capture_threshold)
        prompt_record["root_selection_threshold"] = float(self.root_selection_threshold)
        prompt_record["child_selection_threshold_default"] = float(
            self.child_selection_threshold_default
        )
        prompt_record["child_selection_threshold_small"] = float(
            self.child_selection_threshold_small
        )
        prompt_record["child_small_area_pct_of_full_image"] = float(
            self.child_small_area_pct_of_full_image
        )

        selected: List[Tuple[Optional[float], np.ndarray, U.BBox]] = []
        thresholds_used: List[float] = []

        for score, mb_pp, sb in candidates:
            # area_px = int((mb_pp > 0).sum()) if mb_pp is not None else 0
            area_px = int(np.count_nonzero(mb_pp)) if mb_pp is not None else 0
            thr = self._selection_threshold_for_candidate(
                source=source,
                processed_area_px=area_px,
                full_area_px=full_area_px,
            )
            thresholds_used.append(float(thr))
            if score is None:
                continue
            if float(score) >= float(thr):
                selected.append((float(score), mb_pp, sb))

        prompt_record["selection_thresholds_used"] = sorted(
            {float(t) for t in thresholds_used}
        )
        prompt_record["passes_selection_threshold"] = bool(len(selected) > 0)
        prompt_record["passes_confidence_threshold"] = bool(len(selected) > 0)
        return selected

    def _record_prompt_summary(
        self,
        parent_node_id: Optional[str],
        parent_label: str,
        parent_path: List[str],
        prompt_record: Dict[str, Any],
    ) -> None:
        row = {
            "parent_node_id": parent_node_id,
            "parent_label": parent_label,
            "parent_path": list(parent_path),
            "prompt": prompt_record.get("prompt"),
            "source": prompt_record.get("source"),
            "llm_mode": prompt_record.get("llm_mode"),
            "status": prompt_record.get("status"),
            "reject_reason": prompt_record.get("reject_reason"),
            "median_score": prompt_record.get("median_score"),
            "cand_total_raw": prompt_record.get("candidate_total_raw"),
            "cand_total_processed": prompt_record.get("candidate_total_processed"),
            "cand_scored": prompt_record.get("cand_scored"),
            "merged_area_px": prompt_record.get("merged_area_px"),
            "parent_child_iou": prompt_record.get("parent_child_iou"),
            "passes_confidence_threshold": prompt_record.get("passes_confidence_threshold"),
            "passes_selection_threshold": prompt_record.get("passes_selection_threshold"),
            "sibling_contained_by": prompt_record.get("sibling_contained_by"),
            "sam_capture_threshold": prompt_record.get("sam_capture_threshold"),
            "root_selection_threshold": prompt_record.get("root_selection_threshold"),
            "child_selection_threshold_default": prompt_record.get(
                "child_selection_threshold_default"
            ),
            "child_selection_threshold_small": prompt_record.get(
                "child_selection_threshold_small"
            ),
            "child_small_area_pct_of_full_image": prompt_record.get(
                "child_small_area_pct_of_full_image"
            ),
            "selection_thresholds_used": prompt_record.get("selection_thresholds_used"),
            "selection_mode": prompt_record.get("selection_mode"),
        }
        self.prompt_decision_rows.append(row)

    def _materialize_items_to_nodes(
        self,
        *,
        items: List[Dict[str, Any]],
        pil_full: Image.Image,
        processor,
        min_area_px: int,
        pad: int,
        nodes: Dict[str, NodeInfo],
        root_children: List[str],
        queue: Deque[str],
        new_node_id,
        max_total_nodes: int,
        parent_id: Optional[str],
        image_out_dir: str,
    ) -> Tuple[List[str], List[str]]:
        added_labels: List[str] = []
        added_node_ids: List[str] = []
        w_full, h_full = pil_full.width, pil_full.height

        for item in items:
            if len(nodes) >= max_total_nodes:
                break
            label = str(item.get("label", "")).strip()
            if not label:
                continue

            child_path = item.get("child_path")
            child_dir = node_dir_from_path(image_out_dir, child_path)
            item["child_dir"] = child_dir
            os.makedirs(child_dir, exist_ok=True)

            if self.cfg.save_candidate_png and item.get("candidates_for_overlay"):
                safe_label = sanitize_filename(label)
                overlay_path = os.path.join(child_dir, f"{safe_label}__candidates.png")
                save_candidates_score_overlay_full(
                    out_png_path=overlay_path,
                    pil_full=pil_full,
                    crop_bbox=(0, 0, w_full, h_full),
                    candidates=item["candidates_for_overlay"],
                    score_thr=0.0,
                )
                item.setdefault("prompt_record", {})["candidate_overlay_png"] = overlay_path

            t0 = time.perf_counter()
            merged_path, mask_rgb_path, bbox_rgb_path, instance_paths, bbox_pad, bundle_path = save_node_masks(
                pil_full=pil_full,
                node_dir=child_dir,
                prompt_label=label,
                instances_u8=item.get("instances_u8", []),
                merged_u8=item.get("merged_u8"),
                min_area_px=min_area_px,
                pad=pad,
            )
            self.t_io += time.perf_counter()-t0

            if not merged_path or not mask_rgb_path or not bbox_rgb_path or not bbox_pad or not bundle_path:
                item.setdefault("prompt_record", {})["status"] = "rejected"
                item.setdefault("prompt_record", {})["reject_reason"] = "min_area"
                continue

            node_id = new_node_id()
            node = NodeInfo(
                node_id=node_id,
                label=label,
                path=list(child_path),
                parent_id=parent_id,
                merged_mask=item["merged_u8"],
                crop_bbox=bbox_pad,
                pad=pad,
                node_dir=child_dir,
                merged_mask_path=merged_path,
                mask_rgb_path=mask_rgb_path,
                bbox_rgb_path=bbox_rgb_path,
                mask_bundle_path=bundle_path,
                instance_mask_paths=instance_paths,
            )
            nodes[node_id] = node
            queue.append(node_id)
            if parent_id is None:
                root_children.append(node_id)

            added_labels.append(label)
            added_node_ids.append(node_id)

            prompt_record = item.get("prompt_record")
            if isinstance(prompt_record, dict):
                prompt_record["materialized_node_id"] = node_id
                prompt_record["node_dir"] = child_dir
                self.node_meta[node_id] = {
                    "source": prompt_record.get("source"),
                    "median_score": prompt_record.get("median_score"),
                    "cand_total": prompt_record.get("candidate_total_processed"),
                    "cand_scored": prompt_record.get("cand_scored"),
                    "passes_selection_threshold": prompt_record.get("passes_selection_threshold"),
                    "sam_capture_threshold": prompt_record.get("sam_capture_threshold"),
                    "root_selection_threshold": prompt_record.get("root_selection_threshold"),
                    "child_selection_threshold_default": prompt_record.get(
                        "child_selection_threshold_default"
                    ),
                    "child_selection_threshold_small": prompt_record.get(
                        "child_selection_threshold_small"
                    ),
                }

        return added_labels, added_node_ids

    def _expand_node_with_child_prompts(
        self,
        *,
        api_key: str,
        model_name: str,
        nid: str,
        node: NodeInfo,
        llm_meta: Dict[str, Any],
        child_prompts: List[str],
        nodes: Dict[str, NodeInfo],
        root_children: List[str],
        queue: Deque[str],
        new_node_id,
        image_out_dir: str,
        pil_full: Image.Image,
        processor,
        min_area_px: int,
        full_area: int,
    ) -> None:
        child_prompts = dedupe_texts(child_prompts, max_items=int(self.cfg.max_children))
        node_audit: Dict[str, Any] = {
            "node_id": nid,
            "label": node.label,
            "path": list(node.path),
            "llm": dict(llm_meta),
            "prompts": [],
        }

        if not child_prompts:
            audit_path = save_json_gz(os.path.join(node.node_dir, "node.audit.json.gz"), node_audit)
            self.node_meta.setdefault(nid, {})["audit_path"] = audit_path
            self.node_meta.setdefault(nid, {})["llm_mode"] = llm_meta.get("llm_mode")
            self.node_meta.setdefault(nid, {})["single_fallback_used"] = bool(
                llm_meta.get("single_fallback_used")
            )
            return

        strict_full_u8 = node.merged_mask
        # parent_area_strict = int((strict_full_u8 > 0).sum())
        parent_area_strict = int(np.count_nonzero(strict_full_u8))
        if parent_area_strict <= 0:
            audit_path = save_json_gz(os.path.join(node.node_dir, "node.audit.json.gz"), node_audit)
            self.node_meta.setdefault(nid, {})["audit_path"] = audit_path
            return

        h_full = pil_full.height
        w_full = pil_full.width
        bbox_strict = U.bbox_from_mask(strict_full_u8)
        if bbox_strict is None:
            audit_path = save_json_gz(os.path.join(node.node_dir, "node.audit.json.gz"), node_audit)
            self.node_meta.setdefault(nid, {})["audit_path"] = audit_path
            return

        parent_bbox_pad = U.pad_bbox(
            bbox_strict,
            pad=int(self.cfg.pad),
            w=int(w_full),
            h=int(h_full),
        )
        print(f"[SUBCROP] node={node.label} path={'/'.join(node.path)}")
        sub_bboxes = split_parent_into_subcrops(
            parent_mask_full_u8=strict_full_u8,
            parent_bbox_pad=parent_bbox_pad,
            pad=int(self.cfg.pad),
            w_full=int(w_full),
            h_full=int(h_full),
            split_fill_ratio=float(self.cfg.subcrop_split_fill_ratio),
            min_cc_area_frac=float(self.cfg.subcrop_min_cc_area_frac),
            min_cc_area_floor=int(self.cfg.subcrop_min_cc_area_floor),
            max_subcrops=self.cfg.max_subcrops,
        )

        per_label_candidates: Dict[str, List[Tuple[Optional[float], np.ndarray, U.BBox]]] = {}
        prompt_records: Dict[str, Dict[str, Any]] = {}

        for rank, child_label in enumerate(child_prompts, start=1):
            prompt_records[child_label] = {
                "prompt": child_label,
                "source": "child",
                "rank": int(rank),
                "llm_mode": llm_meta.get("llm_mode"),
                "llm_call_ids": list(llm_meta.get("llm_call_ids", [])),
                "missing_from_batch": bool(llm_meta.get("missing_from_batch", False)),
                "single_fallback_used": bool(llm_meta.get("single_fallback_used", False)),
                "parse_fix_used": bool(llm_meta.get("parse_fix_used", False)),
                "status": "pending",
                "reject_reason": None,
                "selection_mode": "per_candidate",
                "passes_confidence_threshold": False,
                "passes_selection_threshold": False,
                "candidate_total_raw": 0,
                "candidate_total_processed": 0,
                "cand_scored": 0,
                "median_score": None,
                "merged_area_px": 0,
                "parent_child_iou": None,
                "subcrop_count": len(sub_bboxes),
                "full_area_px": int(full_area),
                "sam_capture_threshold": float(self.sam_capture_threshold),
                "root_selection_threshold": float(self.root_selection_threshold),
                "child_selection_threshold_default": float(
                    self.child_selection_threshold_default
                ),
                "child_selection_threshold_small": float(
                    self.child_selection_threshold_small
                ),
                "child_small_area_pct_of_full_image": float(
                    self.child_small_area_pct_of_full_image
                ),
                "candidates": [],
            }

        sub_debug_dir = (
            os.path.join(node.node_dir, "__subcrops")
            if (self.cfg.debug_save and self.cfg.save_subcrop_debug)
            else None
        )

        for sb_idx, (sb, cc_mask01) in enumerate(sub_bboxes):
            sx1, sy1, sx2, sy2 = sb
            crop_rgb = np.asarray(pil_full.crop(sb)).copy()
            crop_rgb[cc_mask01 == 0] = 0
            sub_parent_pil = Image.fromarray(crop_rgb)
            sub_parent_mask = cc_mask01.astype(np.uint8)

            if sub_debug_dir is not None:
                t0 = time.perf_counter()
                cc_full_u8 = np.zeros((h_full, w_full), dtype=np.uint8)
                cc_full_u8[sy1:sy2, sx1:sx2] = cc_mask01.astype(np.uint8) * 255
                save_subcrop_debug(
                    out_dir=sub_debug_dir,
                    pil_full=pil_full,
                    mask_full_u8=cc_full_u8,
                    sub_bbox=sb,
                    idx=sb_idx,
                    tag=f"focus_{sanitize_filename(node.label)}",
                )
                self.t_io += time.perf_counter() - t0

            sam_out_sub = self._run_sam_on_crop(
                processor=processor,
                pil_img=sub_parent_pil,
                prompts_in=child_prompts,
            )
            Hs = int(sam_out_sub.get("orig_img_h", sy2 - sy1))
            Ws = int(sam_out_sub.get("orig_img_w", sx2 - sx1))
            label_results_sub = sam_out_sub.get("label_results", {}) or {}
            parent_area_sub = int(sub_parent_mask.sum())
            pp_sub = compute_pp_params_from_area(parent_area_sub)

            for child_label in child_prompts:
                prompt_record = prompt_records[child_label]
                lr = label_results_sub.get(child_label, {}) or {}
                rles = lr.get("rles", []) or []
                scores = _align_scores(lr.get("scores", []) or [], len(rles))
                if not rles:
                    continue

                masks_sub = self.backend.decode_rle_masks(rles, h=Hs, w=Ws)
                scores = _align_scores(scores, len(masks_sub))

                for cand_idx, (mask01, score) in enumerate(zip(masks_sub, scores)):
                    mb = (mask01 > 0).astype(np.uint8)
                    crop_h, crop_w = sub_parent_mask.shape
                    if mb.shape != (crop_h, crop_w):
                        mb = cv2.resize(mb, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    mb = (mb & sub_parent_mask).astype(np.uint8)
                    raw_area = int(mb.sum())

                    cand_record: Dict[str, Any] = {
                        "candidate_id": f"sb{sb_idx:02d}_cand{cand_idx:03d}",
                        "subcrop_idx": int(sb_idx),
                        "subcrop_bbox": bbox_to_dict(sb),
                        "mask_space": "subcrop_local",
                        "mask_size": {"width": int(crop_w), "height": int(crop_h)},
                        "score": None if score is None else float(score),
                        "raw_area_px": int(raw_area),
                        "processed_area_px": 0,
                        "status": "pending",
                        "drop_reason": None,
                    }

                    if raw_area <= 0:
                        cand_record["status"] = "dropped"
                        cand_record["drop_reason"] = "empty_after_parent_clip"
                        prompt_record["candidates"].append(cand_record)
                        continue

                    prompt_record["candidate_total_raw"] += 1
                    if self.cfg.save_candidate_cache and self.cfg.save_raw_candidate_masks:
                        cand_record["raw_rle"] = mask_to_coco_rle(
                            (mb > 0).astype(np.uint8) * 255
                        )

                    mb_pp = postprocess_mask_u8(
                        mb,
                        min_comp_area=int(pp_sub["min_comp_area"]),
                        close_k=int(pp_sub["close_k"]),
                        open_k=int(pp_sub["open_k"]),
                        fill_holes=bool(pp_sub["fill_holes"]),
                    )
                    processed_area = int(mb_pp.sum())
                    cand_record["processed_area_px"] = int(processed_area)

                    if processed_area <= 0:
                        cand_record["status"] = "dropped"
                        cand_record["drop_reason"] = "empty_after_postprocess"
                        prompt_record["candidates"].append(cand_record)
                        continue

                    cover = processed_area / max(1, int(sub_parent_mask.sum()))
                    if cover >= float(self.cfg.candidate_cover_parent_max):
                        cand_record["status"] = "dropped"
                        cand_record["drop_reason"] = "cover_parent"
                        cand_record["cover_parent"] = float(cover)
                        if (
                            self.cfg.save_candidate_cache
                            and self.cfg.save_processed_candidate_masks
                        ):
                            cand_record["processed_rle"] = mask_to_coco_rle(
                                (mb_pp > 0).astype(np.uint8) * 255
                            )
                        prompt_record["candidates"].append(cand_record)
                        continue

                    sel_thr = self._selection_threshold_for_candidate(
                        source="child",
                        processed_area_px=processed_area,
                        full_area_px=int(full_area),
                    )
                    area_ratio_to_full = (
                        float(processed_area) / float(full_area) if int(full_area) > 0 else 0.0
                    )

                    cand_record["status"] = "kept_processed"
                    cand_record["cover_parent"] = float(cover)
                    cand_record["selection_threshold_used"] = float(sel_thr)
                    cand_record["area_ratio_to_full"] = float(area_ratio_to_full)
                    cand_record["passes_selection_threshold"] = (
                        None if score is None else bool(float(score) >= float(sel_thr))
                    )
                    cand_record["passes_confidence_threshold"] = cand_record[
                        "passes_selection_threshold"
                    ]

                    if self.cfg.save_candidate_cache and self.cfg.save_processed_candidate_masks:
                        cand_record["processed_rle"] = mask_to_coco_rle(
                            (mb_pp > 0).astype(np.uint8) * 255
                        )

                    prompt_record["candidate_total_processed"] += 1
                    prompt_record["candidates"].append(cand_record)
                    per_label_candidates.setdefault(child_label, []).append((score, mb_pp, sb))

        accepted_items: List[Dict[str, Any]] = []
        for child_label in child_prompts:
            if len(nodes) >= int(self.cfg.max_total_nodes):
                break

            prompt_record = prompt_records[child_label]
            cand_list_all = per_label_candidates.get(child_label, []) or []
            if not cand_list_all:
                prompt_record["status"] = "rejected"
                prompt_record["reject_reason"] = "sam_not_found"
                node_audit["prompts"].append(prompt_record)
                self._record_prompt_summary(nid, node.label, node.path, prompt_record)
                continue

            cand_list = self._select_candidates_by_policy(
                candidates=cand_list_all,
                prompt_record=prompt_record,
            )

            if not cand_list:
                prompt_record["status"] = "rejected"
                prompt_record["reject_reason"] = "no_candidate_passed_selection_threshold"
                node_audit["prompts"].append(prompt_record)
                self._record_prompt_summary(nid, node.label, node.path, prompt_record)
                continue

            merged_u8 = np.zeros((h_full, w_full), dtype=np.uint8)
            instances_u8: List[np.ndarray] = []
            overlay_candidates: List[Tuple[Optional[float], np.ndarray]] = []

            for score, mb_sub01, sb in cand_list:
                sx1, sy1, sx2, sy2 = sb
                full_mask = np.zeros((h_full, w_full), dtype=np.uint8)
                full_mask[sy1:sy2, sx1:sx2] = (mb_sub01 > 0).astype(np.uint8) * 255
                merged_u8[sy1:sy2, sx1:sx2] = np.maximum(
                    merged_u8[sy1:sy2, sx1:sx2],
                    full_mask[sy1:sy2, sx1:sx2],
                )
                instances_u8.append(full_mask)
                overlay_candidates.append((score, full_mask > 0))

            # prompt_record["merged_area_px"] = int((merged_u8 > 0).sum())
            prompt_record["merged_area_px"] = int(np.count_nonzero(merged_u8))

            parent_iou = U.mask_iou_u8(U.ensure_u8(merged_u8), U.ensure_u8(node.merged_mask))
            prompt_record["parent_child_iou"] = float(parent_iou)
            if parent_iou >= float(self.cfg.parent_child_iou_max):
                prompt_record["status"] = "rejected"
                prompt_record["reject_reason"] = "parent_child_iou"
                node_audit["prompts"].append(prompt_record)
                self._record_prompt_summary(nid, node.label, node.path, prompt_record)
                continue

            item = {
                "label": child_label,
                "child_path": node.path + [child_label],
                "merged_u8": merged_u8,
                "instances_u8": instances_u8,
                "prompt_record": prompt_record,
                "candidates_for_overlay": overlay_candidates,
            }
            accepted_items.append(item)

        kept_items, dropped_infos = self._dedupe_by_containment(
            accepted_items,
            contain_thr=float(self.cfg.sibling_containment_thr),
        )
        dropped_small_idxs = {
            small_idx: (big_idx, ratio) for small_idx, big_idx, ratio in dropped_infos
        }

        for idx, item in enumerate(accepted_items):
            prompt_record = item["prompt_record"]
            if idx in dropped_small_idxs:
                big_idx, ratio = dropped_small_idxs[idx]
                big_label = accepted_items[big_idx]["label"]
                prompt_record["status"] = "rejected"
                prompt_record["reject_reason"] = "sibling_containment"
                prompt_record["sibling_contained_by"] = big_label
                prompt_record["containment_ratio"] = float(ratio)
            else:
                prompt_record["status"] = "accepted"

        _added_labels, added_node_ids = self._materialize_items_to_nodes(
            items=kept_items,
            pil_full=pil_full,
            processor=processor,
            min_area_px=min_area_px,
            pad=int(self.cfg.pad),
            nodes=nodes,
            root_children=root_children,
            queue=queue,
            new_node_id=new_node_id,
            max_total_nodes=int(self.cfg.max_total_nodes),
            parent_id=nid,
            image_out_dir=image_out_dir,
        )
        node.children.extend(added_node_ids)

        for item in accepted_items:
            prompt_record = item["prompt_record"]
            node_audit["prompts"].append(prompt_record)
            self._record_prompt_summary(nid, node.label, node.path, prompt_record)

        audit_path = save_json_gz(os.path.join(node.node_dir, "node.audit.json.gz"), node_audit)
        self.node_meta.setdefault(nid, {})["audit_path"] = audit_path
        self.node_meta.setdefault(nid, {})["llm_mode"] = llm_meta.get("llm_mode")
        self.node_meta.setdefault(nid, {})["single_fallback_used"] = bool(
            llm_meta.get("single_fallback_used")
        )
        self.node_meta.setdefault(nid, {})["parse_fix_used"] = bool(
            llm_meta.get("parse_fix_used")
        )
        self.node_meta.setdefault(nid, {})["llm_suggested_count"] = len(child_prompts)

    def _build_nodes_json(
        self,
        nodes: Dict[str, NodeInfo],
        *,
        keep_set: Optional[Sequence[str]] = None,
        child_overrides: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        keep = None if keep_set is None else set(keep_set)
        nodes_json_out: Dict[str, Any] = {}

        def semantic_parent_id_of(nid: str) -> Optional[str]:
            node = nodes.get(nid)
            if not node or node.parent_id is None:
                return None
            parent = nodes.get(node.parent_id)
            if parent and _is_root_others_node(parent.path, parent.label):
                return None
            return node.parent_id

        for nid, node in nodes.items():
            if keep is not None and nid not in keep:
                continue
            payload = node.to_json()
            payload["raw_depth"] = int(_raw_depth(node.path))
            payload["semantic_depth"] = int(_semantic_depth(node.path))
            payload["semantic_path"] = _semantic_path(node.path)
            payload["semantic_parent_id"] = semantic_parent_id_of(nid)
            payload["under_others"] = bool(len(node.path) >= 2 and node.path[1] == "others")
            if child_overrides is not None:
                payload["children"] = [
                    cid
                    for cid in child_overrides.get(nid, [])
                    if cid in nodes and (keep is None or cid in keep)
                ]
            meta = self.node_meta.get(nid, {})
            for key, value in meta.items():
                payload[key] = value
            nodes_json_out[nid] = payload
        return nodes_json_out

    def _build_hmaps(
        self,
        nodes: Dict[str, NodeInfo],
        keep_set: Sequence[str],
        image_out_dir: str,
    ) -> Tuple[Optional[str], Optional[str], int]:
        if not self.cfg.save_raw_hmaps:
            return None, None, 0
        keep = set(keep_set)
        if not keep:
            return None, None, 0

        sample = next(iter(keep))
        h_full, w_full = nodes[sample].merged_mask.shape[:2]

        depth_max = np.zeros((h_full, w_full), dtype=np.uint16)
        depth_cnt = np.zeros((h_full, w_full), dtype=np.uint16)
        node_id_map = np.zeros((h_full, w_full), dtype=np.uint16)

        node_items: List[Tuple[int, float, int, str]] = []
        for nid in keep:
            node = nodes[nid]
            if _is_root_others_node(node.path, node.label):
                continue
            sd = _semantic_depth(node.path)
            if sd <= 0:
                continue
            # area = int((node.merged_mask > 0).sum())
            area = int(np.count_nonzero(node.merged_mask))
            if area <= 0:
                continue
            ms = self.node_meta.get(nid, {}).get("median_score", None)
            ms = float(ms) if ms is not None else -1.0
            node_items.append((int(sd), float(ms), -int(area), nid))

        node_items.sort(key=lambda t: (t[0], t[1], t[2]))
        idx_to_node_id: List[str] = []
        max_seen_sem_depth = 0

        for local_idx, (sd, _ms, _neg_area, nid) in enumerate(node_items, start=1):
            node = nodes[nid]
            mask = node.merged_mask > 0
            if not np.any(mask):
                continue
            max_seen_sem_depth = max(max_seen_sem_depth, int(sd))
            vals = depth_cnt[mask].astype(np.uint32) + 1
            depth_cnt[mask] = np.minimum(vals, np.iinfo(np.uint16).max).astype(np.uint16)
            upd = mask & (sd >= depth_max)
            depth_max[upd] = np.uint16(sd)
            node_id_map[upd] = np.uint16(local_idx)
            idx_to_node_id.append(nid)

        hmaps_npz = os.path.join(image_out_dir, "h_maps_semantic.npz")
        node_index_json = os.path.join(image_out_dir, "node_index.json")
        np.savez_compressed(
            hmaps_npz,
            depth_max=depth_max,
            node_id_map=node_id_map,
            depth_cnt=depth_cnt,
            max_seen_sem_depth=np.array([max_seen_sem_depth], dtype=np.uint16),
        )
        save_json(node_index_json, {"idx_to_node_id": idx_to_node_id})

        if self.cfg.save_hmaps_vis:
            if max_seen_sem_depth > 0:
                dvis = (
                    depth_max.astype(np.float32)
                    / float(max_seen_sem_depth)
                    * 255.0
                ).clip(0, 255).astype(np.uint8)
            else:
                dvis = np.zeros((h_full, w_full), dtype=np.uint8)

            depth_color_bgr = cv2.applyColorMap(dvis, cv2.COLORMAP_VIRIDIS)
            depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
            depth_color_rgb[depth_cnt == 0] = 0
            Image.fromarray(depth_color_rgb).save(
                os.path.join(image_out_dir, "depth_max_viridis.png")
            )

            c = depth_cnt.astype(np.float32)
            cden = max(1.0, float(c.max()))
            cvis = (c / cden * 255.0).clip(0, 255).astype(np.uint8)
            cnt_color_bgr = cv2.applyColorMap(cvis, cv2.COLORMAP_INFERNO)
            Image.fromarray(cv2.cvtColor(cnt_color_bgr, cv2.COLOR_BGR2RGB)).save(
                os.path.join(image_out_dir, "depth_cnt_inferno.png")
            )

        return hmaps_npz, node_index_json, int(max_seen_sem_depth)

    def generate(
        self,
        *,
        image: Image.Image,
        image_out_dir: str,
        openrouter_api_key: str,
        model_name: str,
        title: str = "",
        description: str = "",
        filename: str = "image",
        image_source_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> RunResult:
        if not openrouter_api_key or openrouter_api_key.strip().lower() in {"replace-me", "your-openrouter-api-key"}:
            raise ValueError("Please provide a valid OpenRouter API key.")

        if overwrite:
            reset_dir(image_out_dir)
        else:
            os.makedirs(image_out_dir, exist_ok=True)

        self._reset_run_state()
        processor = self._load_processor()

        pil_in = (
            image.convert("RGB")
            if isinstance(image, Image.Image)
            else Image.fromarray(np.asarray(image)).convert("RGB")
        )
        w_in, h_in = pil_in.size
        pil_full = resize_longest_side(
            pil_in,
            max_longest_side=int(self.cfg.input_resize_longest_side),
        )

        image_root_path = os.path.join(image_out_dir, "image.root.jpg")
        pil_full.save(image_root_path, format="JPEG", quality=95, subsampling=0)

        if self.cfg.use_clahe:
            pil_full = apply_l_clahe_pil(
                pil_full,
                clip_limit=float(self.cfg.clahe_clip_limit),
                tile_grid=int(self.cfg.clahe_tile_grid),
            )
            pil_full.save(
                os.path.join(image_out_dir, "image.clahe.jpg"),
                format="JPEG",
                quality=95,
                subsampling=0,
            )

        h_full = pil_full.height
        w_full = pil_full.width
        hw_full = max(h_full, w_full)
        full_area = h_full * w_full
        min_area_px = max(int(full_area * (float(self.cfg.min_area_pct) / 100.0)), 1)

        prompt_payload = {
            "SYSTEM_PROMPT": prompts.SYSTEM_PROMPT,
            "INITIAL_DISCOVERY_USER_MESSAGE": prompts.INITIAL_DISCOVERY_USER_MESSAGE,
            "LOCAL_DECOMPOSITION_USER_MESSAGE": prompts.LOCAL_DECOMPOSITION_USER_MESSAGE,
            "LOCAL_DECOMPOSITION_BATCH_MESSAGE": prompts.LOCAL_DECOMPOSITION_BATCH_MESSAGE,
        }
        image_sha = (
            sha256_file(image_source_path)
            if image_source_path and os.path.isfile(image_source_path)
            else None
        )
        provenance = build_basic_provenance(
            cfg=self.cfg,
            prompt_payload=prompt_payload,
            image_source_path=image_source_path,
            image_sha256=image_sha,
            package_dir=self._package_dir,
            model_name=model_name,
        )
        provenance["processed_root_image_sha256"] = sha256_file(image_root_path)
        provenance["sam_capture_threshold"] = float(self.sam_capture_threshold)
        provenance["root_selection_threshold"] = float(self.root_selection_threshold)
        provenance["child_selection_threshold_default"] = float(
            self.child_selection_threshold_default
        )
        provenance["child_selection_threshold_small"] = float(
            self.child_selection_threshold_small
        )
        provenance["child_small_area_pct_of_full_image"] = float(
            self.child_small_area_pct_of_full_image
        )

        full_b64 = image_to_base64_jpeg(pil_full, quality=85)
        init_text = prompts.INITIAL_DISCOVERY_USER_MESSAGE.format(
            title=title or "",
            description=description or "",
        )
        init_messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": full_b64}},
                    {"type": "text", "text": init_text},
                ],
            },
        ]

        init_labels: List[str] = []
        init_call_ids: List[int] = []
        tool = None
        for attempt in range(3):
            resp, _usage, call_id = self._call_llm(
                api_key=openrouter_api_key,
                model_name=model_name,
                messages=init_messages,
                tag=f"initial_{attempt}",
                stage="initial",
                temperature=float(self.cfg.initial_temperature),
                max_retry=int(self.cfg.initial_max_retry),
                extra={"title": title, "description": description},
            )
            init_call_ids.append(int(call_id))
            tool = parse_llm_tool_call(resp or "")
            if tool and tool.get("name") == "propose_prompts":
                break

        if tool and tool.get("name") == "propose_prompts":
            init_labels = tool.get("parameters", {}).get("text_prompts", []) or []
            if isinstance(init_labels, str):
                init_labels = [init_labels]
            init_labels = dedupe_texts(init_labels)

        sam_out = self._run_sam_on_crop(
            processor=processor,
            pil_img=pil_full,
            prompts_in=init_labels,
        )
        label_results = sam_out.get("label_results", {}) or {}
        Hc = int(sam_out.get("orig_img_h", h_full))
        Wc = int(sam_out.get("orig_img_w", w_full))

        nodes: Dict[str, NodeInfo] = {}
        root_children: List[str] = []
        queue: Deque[str] = deque()
        node_counter = 0

        def new_node_id() -> str:
            nonlocal node_counter
            node_counter += 1
            return f"n{node_counter:0{NODE_ID_WIDTH}d}"

        root_prompt_records: List[Dict[str, Any]] = []
        root_items: List[Dict[str, Any]] = []

        for rank, label in enumerate(init_labels, start=1):
            lr = label_results.get(label, {}) or {}
            rles = lr.get("rles", []) or []
            scores = _align_scores(lr.get("scores", []) or [], len(rles))

            record: Dict[str, Any] = {
                "prompt": label,
                "source": "init",
                "rank": int(rank),
                "llm_call_ids": list(init_call_ids),
                "status": "pending",
                "reject_reason": None,
                "median_score": None,
                "selection_mode": "per_candidate",
                "passes_confidence_threshold": False,
                "passes_selection_threshold": False,
                "candidate_total_raw": 0,
                "candidate_total_processed": 0,
                "cand_scored": 0,
                "merged_area_px": 0,
                "full_area_px": int(full_area),
                "sam_capture_threshold": float(self.sam_capture_threshold),
                "root_selection_threshold": float(self.root_selection_threshold),
                "child_selection_threshold_default": float(
                    self.child_selection_threshold_default
                ),
                "child_selection_threshold_small": float(
                    self.child_selection_threshold_small
                ),
                "child_small_area_pct_of_full_image": float(
                    self.child_small_area_pct_of_full_image
                ),
                "candidates": [],
            }

            if not rles:
                record["status"] = "rejected"
                record["reject_reason"] = "sam_not_found"
                root_prompt_records.append(record)
                continue

            masks_full = self.backend.decode_rle_masks(rles, h=Hc, w=Wc)
            scores = _align_scores(scores, len(masks_full))

            processed_candidates: List[Tuple[Optional[float], np.ndarray, U.BBox]] = []

            for idx, (mask01, score) in enumerate(zip(masks_full, scores)):
                mb = (mask01 > 0).astype(np.uint8)
                raw_area = int(mb.sum())
                cand = {
                    "candidate_id": f"root_cand_{idx:03d}",
                    "score": None if score is None else float(score),
                    "raw_area_px": int(raw_area),
                    "status": "pending",
                    "drop_reason": None,
                    "mask_space": "processed_root",
                    "mask_size": {"width": int(Wc), "height": int(Hc)},
                }

                if raw_area <= 0:
                    cand["status"] = "dropped"
                    cand["drop_reason"] = "empty"
                    record["candidates"].append(cand)
                    continue

                if (raw_area / float(full_area)) > float(self.cfg.root_max_mask_area_frac):
                    cand["status"] = "dropped"
                    cand["drop_reason"] = "root_mask_too_large"
                    if self.cfg.save_candidate_cache and self.cfg.save_raw_candidate_masks:
                        cand["raw_rle"] = mask_to_coco_rle(
                            (mask01 > 0).astype(np.uint8) * 255
                        )
                    record["candidates"].append(cand)
                    continue

                pp_root = compute_pp_params_from_area(raw_area)
                mb_pp = postprocess_mask_u8(
                    mb,
                    min_comp_area=int(pp_root["min_comp_area"]),
                    close_k=int(pp_root["close_k"]),
                    open_k=int(pp_root["open_k"]),
                    fill_holes=True,
                )

                processed_area = int(mb_pp.sum())
                cand["processed_area_px"] = int(processed_area)
                if processed_area <= 0:
                    cand["status"] = "dropped"
                    cand["drop_reason"] = "empty_after_postprocess"
                    record["candidates"].append(cand)
                    continue

                sel_thr = self._selection_threshold_for_candidate(
                    source="init",
                    processed_area_px=processed_area,
                    full_area_px=int(full_area),
                )

                cand["status"] = "kept_processed"
                cand["selection_threshold_used"] = float(sel_thr)
                cand["passes_selection_threshold"] = (
                    None if score is None else bool(float(score) >= float(sel_thr))
                )
                cand["passes_confidence_threshold"] = cand["passes_selection_threshold"]

                if self.cfg.save_candidate_cache and self.cfg.save_raw_candidate_masks:
                    cand["raw_rle"] = mask_to_coco_rle((mask01 > 0).astype(np.uint8) * 255)

                processed_candidates.append((score, mb_pp, (0, 0, Wc, Hc)))
                record["candidate_total_raw"] += 1
                record["candidate_total_processed"] += 1
                record["candidates"].append(cand)

            if not processed_candidates:
                record["status"] = "rejected"
                record["reject_reason"] = "no_valid_root_mask"
                root_prompt_records.append(record)
                continue

            selected_candidates = self._select_candidates_by_policy(
                candidates=processed_candidates,
                prompt_record=record,
            )

            if not selected_candidates:
                record["status"] = "rejected"
                record["reject_reason"] = "no_candidate_passed_selection_threshold"
                root_prompt_records.append(record)
                continue

            valid_masks_01 = [mb_pp for _score, mb_pp, _sb in selected_candidates]
            instances_u8 = [(m.astype(np.uint8) * 255) for m in valid_masks_01]
            merged_u8 = U.mask_union(valid_masks_01).astype(np.uint8) * 255
            overlay_candidates = [(_score, _mb_pp) for _score, _mb_pp, _sb in selected_candidates]
            # record["merged_area_px"] = int((merged_u8 > 0).sum())
            record["merged_area_px"] = int(np.count_nonzero(merged_u8))

            item = {
                "label": label,
                "child_path": ["root", label],
                "merged_u8": merged_u8,
                "instances_u8": instances_u8,
                "prompt_record": record,
                "candidates_for_overlay": overlay_candidates,
            }
            root_items.append(item)
            root_prompt_records.append(record)

        kept_root_items, dropped_root_infos = self._dedupe_by_containment(
            root_items,
            contain_thr=float(self.cfg.sibling_containment_thr),
        )
        dropped_root_small_idxs = {
            small_idx: (big_idx, ratio) for small_idx, big_idx, ratio in dropped_root_infos
        }

        for idx, item in enumerate(root_items):
            record = item["prompt_record"]
            if idx in dropped_root_small_idxs:
                big_idx, ratio = dropped_root_small_idxs[idx]
                record["status"] = "rejected"
                record["reject_reason"] = "sibling_containment"
                record["sibling_contained_by"] = root_items[big_idx]["label"]
                record["containment_ratio"] = float(ratio)
            elif record.get("status") == "pending":
                record["status"] = "accepted_root"

        _added_labels_init, added_init_ids = self._materialize_items_to_nodes(
            items=kept_root_items,
            pil_full=pil_full,
            processor=processor,
            min_area_px=min_area_px,
            pad=int(self.cfg.pad),
            nodes=nodes,
            root_children=root_children,
            queue=queue,
            new_node_id=new_node_id,
            max_total_nodes=int(self.cfg.max_total_nodes),
            parent_id=None,
            image_out_dir=image_out_dir,
        )
        for nid in added_init_ids:
            self.node_meta.setdefault(nid, {})["source"] = "init"

        for record in root_prompt_records:
            self._record_prompt_summary(None, "root", ["root"], record)

        self.root_initial_audit = {
            "llm_call_ids": init_call_ids,
            "init_labels": init_labels,
            "prompts": root_prompt_records,
        }
        initial_audit_path = save_json_gz(
            os.path.join(image_out_dir, "root.initial.audit.json.gz"),
            self.root_initial_audit,
        )

        union_u8 = np.zeros((h_full, w_full), dtype=np.uint8)
        for nid in added_init_ids:
            node = nodes.get(nid)
            if node is not None:
                union_u8[:] = np.maximum(union_u8, node.merged_mask)

        others_raw_u8 = ((union_u8 == 0).astype(np.uint8) * 255)
        # others_area_raw = int((others_raw_u8 > 0).sum())
        others_area_raw = int(np.count_nonzero(others_raw_u8))
        min_comp_area_others = int(
            others_area_raw * float(self.cfg.others_min_component_area_frac)
        )

        others_pp_u8 = others_raw_u8.copy()
        k_oc = U.odd(int(hw_full * 0.01))
        others_pp_u8 = U.morph_open_close(others_pp_u8, k_oc)
        dilate_px = max(1, int(hw_full * 0.001))
        others_pp_u8 = U.morph_dilate(others_pp_u8, 2 * dilate_px + 1, iters=1)

        others01 = U.to_mb01(others_pp_u8)
        others01 = U.fill_holes01(others01)
        others_pp_u8 = (others01 * 255).astype(np.uint8)

        others_pp_u8 = np.where(union_u8 == 0, others_pp_u8, 0).astype(np.uint8)
        num, labels, stats, _ = U.connected_components(others_pp_u8, connectivity=8)
        cleaned = np.zeros_like(others_pp_u8, dtype=np.uint8)
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_comp_area_others:
                cleaned[labels == i] = 255
        others_pp_u8 = cleaned
        # others_area_pp = int((others_pp_u8 > 0).sum())
        others_area_pp = int(np.count_nonzero(others_pp_u8))
        others_ratio = (others_area_pp / float(full_area)) if full_area > 0 else 0.0

        others_dir = node_dir_from_path(image_out_dir, ["root", "others"])
        os.makedirs(others_dir, exist_ok=True)
        save_mask_png(os.path.join(others_dir, "others.raw.mask.png"), others_raw_u8)

        if others_area_pp >= min_area_px:
            merged_path, mask_rgb_path, bbox_rgb_path, instance_paths, bbox_pad, bundle_path = save_node_masks(
                pil_full=pil_full,
                node_dir=others_dir,
                prompt_label="others",
                instances_u8=[],
                merged_u8=others_pp_u8,
                min_area_px=min_area_px,
                pad=int(self.cfg.pad),
            )
            if merged_path and mask_rgb_path and bbox_rgb_path and bbox_pad and bundle_path:
                nid = new_node_id()
                others_node = NodeInfo(
                    node_id=nid,
                    label="others",
                    path=["root", "others"],
                    parent_id=None,
                    merged_mask=others_pp_u8,
                    crop_bbox=bbox_pad,
                    pad=int(self.cfg.pad),
                    node_dir=others_dir,
                    merged_mask_path=merged_path,
                    mask_rgb_path=mask_rgb_path,
                    bbox_rgb_path=bbox_rgb_path,
                    mask_bundle_path=bundle_path,
                    instance_mask_paths=instance_paths,
                )
                nodes[nid] = others_node
                root_children.append(nid)
                queue.append(nid)
                self.node_meta[nid] = {
                    "source": "others",
                    "median_score": None,
                    "sam_capture_threshold": float(self.sam_capture_threshold),
                }

        step = 0
        batch_idx = 0
        while queue and len(nodes) < int(self.cfg.max_total_nodes):
            depth0, batch_nids = self._pop_semantic_batch(
                queue,
                nodes,
                int(self.cfg.llm_batch_size),
            )
            if depth0 is None or not batch_nids:
                break
            batch_idx += 1

            llm_items: List[Dict[str, Any]] = []
            llm_crop_meta: Dict[str, Dict[str, Any]] = {}

            for nid in batch_nids:
                node = nodes.get(nid)
                if node is None:
                    continue
                step += 1
                sem_depth = _semantic_depth(node.path)
                # area = int((node.merged_mask > 0).sum()) if node.merged_mask is not None else -1
                area = int(np.count_nonzero(node.merged_mask)) if node.merged_mask is not None else -1

                if sem_depth >= int(self.cfg.max_depth):
                    self.node_meta.setdefault(nid, {})["stop_reason"] = "max_depth"
                    continue
                if area >= 0 and area < min_area_px:
                    self.node_meta.setdefault(nid, {})["stop_reason"] = "min_area"
                    continue

                _local_img_full, local_img_llm, local_b64 = prepare_llm_crop(
                    pil_full,
                    node.merged_mask,
                    node.crop_bbox,
                    max_longest_side=int(self.cfg.llm_crop_max_side),
                    quality=85,
                )
                llm_items.append(
                    {
                        "node_id": nid,
                        "path_text": " -> ".join(node.path),
                        "label": node.label,
                        "image_b64": local_b64,
                    }
                )
                llm_crop_meta[nid] = {
                    "path_text": " -> ".join(node.path),
                    "label": node.label,
                    "local_b64": local_b64,
                    "llm_crop_size": list(local_img_llm.size),
                }

            if not llm_items:
                continue

            mapping_meta = self._get_mapping_resilient(
                api_key=openrouter_api_key,
                model_name=model_name,
                items=llm_items,
                tag_base=f"sem_depth={depth0}_batch={batch_idx}",
                stage="batch",
            )
            expected_ids = [it["node_id"] for it in llm_items]
            missing = [nid for nid in expected_ids if nid not in mapping_meta]
            if missing:
                self.llm_missing_nodes += len(missing)

            for item in llm_items:
                nid = item["node_id"]
                node = nodes.get(nid)
                if node is None or len(nodes) >= int(self.cfg.max_total_nodes):
                    continue

                meta = mapping_meta.get(nid)
                llm_meta = {
                    "llm_mode": None,
                    "llm_call_ids": [],
                    "missing_from_batch": False,
                    "single_fallback_used": False,
                    "parse_fix_used": False,
                }

                if meta is None:
                    self.llm_single_fallback_nodes += 1
                    prompts_out, call_ids, fix_used = self._llm_single_suggest(
                        api_key=openrouter_api_key,
                        model_name=model_name,
                        local_b64=llm_crop_meta[nid]["local_b64"],
                        path_text=llm_crop_meta[nid]["path_text"],
                        focus_label=llm_crop_meta[nid]["label"],
                        node_id=nid,
                    )
                    llm_meta.update(
                        {
                            "llm_mode": "single",
                            "llm_call_ids": call_ids,
                            "missing_from_batch": True,
                            "single_fallback_used": True,
                            "parse_fix_used": bool(fix_used),
                        }
                    )
                    child_prompts = prompts_out
                else:
                    child_prompts = meta.get("prompts", [])
                    if not isinstance(child_prompts, list):
                        self.llm_malformed_nodes += 1
                        child_prompts = []
                    llm_meta.update(
                        {
                            "llm_mode": meta.get("llm_mode"),
                            "llm_call_ids": list(meta.get("call_ids", [])),
                            "missing_from_batch": bool(meta.get("missing_from_batch", False)),
                            "single_fallback_used": False,
                            "parse_fix_used": bool(meta.get("fix_used", False)),
                        }
                    )

                self._expand_node_with_child_prompts(
                    api_key=openrouter_api_key,
                    model_name=model_name,
                    nid=nid,
                    node=node,
                    llm_meta=llm_meta,
                    child_prompts=child_prompts,
                    nodes=nodes,
                    root_children=root_children,
                    queue=queue,
                    new_node_id=new_node_id,
                    image_out_dir=image_out_dir,
                    pil_full=pil_full,
                    processor=processor,
                    min_area_px=min_area_px,
                    full_area=full_area,
                )

            if step % 50 == 0:
                gc.collect()

        final_keep_set = set(nodes.keys())
        final_root_children = list(root_children)

        nodes_json = self._build_nodes_json(nodes)
        tree_str = render_tree_text(nodes, final_root_children)

        hmaps_npz, node_index_json_path, max_seen_sem_depth = self._build_hmaps(
            nodes,
            sorted(final_keep_set),
            image_out_dir,
        )

        merged_list_final: List[Tuple[str, np.ndarray]] = []
        for nid, node in nodes.items():
            if _is_root_others_node(node.path, node.label):
                continue
            merged_list_final.append((f"{nid}__{node.label}", node.merged_mask))

        final_u8 = union_fill_minus_boundary(
            [m for _, m in merged_list_final],
            boundary_px=1,
        )
        if final_u8 is None:
            final_u8 = np.zeros((h_full, w_full), dtype=np.uint8)

        final_mask_path = save_mask_png(os.path.join(image_out_dir, "final_mask.png"), final_u8)

        union_viz = visualize_union_on_image(
            pil_full=pil_full,
            merged_masks_full=merged_list_final,
        )
        union_viz.save(os.path.join(image_out_dir, "final_union_on_image.png"))

        render_masks_only_canvas(
            h=h_full,
            w=w_full,
            merged_masks_full=[m for _, m in merged_list_final],
            background="black",
        ).save(os.path.join(image_out_dir, "final_union_masks_only.png"))

        llm_calls_path = save_json_gz(
            os.path.join(image_out_dir, "llm_calls.json.gz"),
            self.llm_call_records,
        )
        prompt_decisions_path = save_jsonl_gz(
            os.path.join(image_out_dir, "prompt_decisions.jsonl.gz"),
            self.prompt_decision_rows,
        )

        tree_payload = {
            "schema": self.cfg.generator_schema_version,
            "created_at": utc_now_str(),
            "root_children": final_root_children,
            "nodes": nodes_json,
        }

        summary = {
            "schema": self.cfg.generator_schema_version,
            "output_dir": image_out_dir,
            "filename": filename,
            "model_name": model_name,
            "config": self.cfg.to_dict(),
            "config_hash": provenance["config_hash"],
            "prompt_hash": provenance["prompt_hash"],
            "package_source_hash": provenance["package_source_hash"],
            "git_commit": provenance["git_commit"],
            "image_size_original": [int(w_in), int(h_in)],
            "image_size_processed": [int(w_full), int(h_full)],
            "image_source_path": image_source_path,
            "image_source_sha256": image_sha,
            "processed_root_image_sha256": provenance["processed_root_image_sha256"],
            "initial_audit_path": initial_audit_path,
            "llm_calls_path": llm_calls_path,
            "prompt_decisions_path": prompt_decisions_path,
            "hmaps_path": hmaps_npz,
            "node_index_path": node_index_json_path,
            "max_seen_sem_depth": int(max_seen_sem_depth),
            "others_area_raw_px": int(others_area_raw),
            "others_area_processed_px": int(others_area_pp),
            "others_area_ratio": float(others_ratio),
            "llm_calls_total": int(sum(self.llm_stage_counts.values())),
            "llm_calls_initial": int(self.llm_stage_counts.get("initial", 0)),
            "llm_calls_batch": int(self.llm_stage_counts.get("batch", 0)),
            "llm_calls_fix": int(self.llm_stage_counts.get("fix", 0)),
            "llm_calls_mini": int(self.llm_stage_counts.get("mini", 0)),
            "llm_calls_single": int(self.llm_stage_counts.get("single", 0)),
            "llm_batch_parse_fail": int(self.llm_batch_parse_fail),
            "llm_missing_nodes": int(self.llm_missing_nodes),
            "llm_single_fallback_nodes": int(self.llm_single_fallback_nodes),
            "llm_malformed_nodes": int(self.llm_malformed_nodes),
            "sam_calls": int(self.sam_call_count),
            "sam_capture_threshold": float(self.sam_capture_threshold),
            "root_selection_threshold": float(self.root_selection_threshold),
            "child_selection_threshold_default": float(
                self.child_selection_threshold_default
            ),
            "child_selection_threshold_small": float(
                self.child_selection_threshold_small
            ),
            "child_small_area_pct_of_full_image": float(
                self.child_small_area_pct_of_full_image
            ),
            "selection_mode": "per_candidate",
            "time_llm": float(self.t_llm),
            "time_sam": float(self.t_sam),
            "time_io": float(self.t_io),
            "time_total_partial": float(self.t_llm + self.t_sam + self.t_io),
            "final_mask_path": final_mask_path,
        }

        if isinstance(summary.get("config"), dict):
            summary["config"]["sam_capture_threshold"] = float(self.sam_capture_threshold)
            summary["config"]["root_selection_threshold"] = float(
                self.root_selection_threshold
            )
            summary["config"]["child_selection_threshold_default"] = float(
                self.child_selection_threshold_default
            )
            summary["config"]["child_selection_threshold_small"] = float(
                self.child_selection_threshold_small
            )
            summary["config"]["child_small_area_pct_of_full_image"] = float(
                self.child_small_area_pct_of_full_image
            )
            summary["config"]["selection_mode"] = "per_candidate"

        tree_json_path, tree_txt_path, summary_path, provenance_path = save_tree_summary_provenance(
            image_out_dir=image_out_dir,
            tree_payload=tree_payload,
            tree_text=tree_str,
            summary=summary,
            provenance=provenance,
            tree_filename="tree.json",
            tree_text_filename="tree.txt",
            summary_filename="run_summary.json",
            provenance_filename="provenance.json",
        )

        done_payload = {
            "status": "done",
            "created_at_utc": utc_now_str(),
            "summary_path": summary_path,
            "tree_path": tree_json_path,
        }
        write_done_marker(image_out_dir, done_payload)

        return RunResult(
            output_dir=image_out_dir,
            summary=summary,
            tree_str=tree_str,
            final_mask_path=final_mask_path,
            tree_path=tree_json_path,
        )
