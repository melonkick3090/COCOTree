from __future__ import annotations

import os
import argparse
import gc
import glob
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import COCOTreeGenerator
    from .types import GeneratorConfig

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def sanitize_filename(name: str, max_len: int = 200) -> str:
    s = str(name).strip()
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = s.strip("._")
    if not s:
        s = "item"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def list_images_in_dir(image_dir: str, recursive: bool = False) -> List[str]:
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(image_dir)
    patterns = [os.path.join(image_dir, "**", f"*{ext}") if recursive else os.path.join(image_dir, f"*{ext}") for ext in IMG_EXTS]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=recursive))
    return sorted(set(p for p in files if os.path.isfile(p)))


def apply_slice(paths: List[str], start_idx: int, end_idx: int) -> List[str]:
    n = len(paths)
    s = max(0, int(start_idx))
    e = int(end_idx)
    if e < 0:
        e = n - 1
    e = min(e, n - 1)
    if s > e:
        return []
    return paths[s : e + 1]


def apply_shard(paths: List[str], shard_id: int, num_shards: int) -> List[str]:
    if num_shards <= 1:
        return paths
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards - 1}]")
    return [p for i, p in enumerate(paths) if (i % num_shards) == shard_id]


def output_subdir(out_root: str, run_name: str, image_path: str) -> Tuple[str, str]:
    stem = sanitize_filename(os.path.splitext(os.path.basename(image_path))[0])
    base = os.path.join(out_root, run_name) if run_name else out_root
    return os.path.join(base, stem), stem


def is_done(out_subdir: str) -> bool:
    return os.path.isfile(os.path.join(out_subdir, "_DONE.json"))


def write_error(out_subdir: str, image_path: str, err: str) -> None:
    from .io_utils import ensure_dir, save_json, utc_now_str

    ensure_dir(out_subdir)
    payload = {
        "image": image_path,
        "error": err,
        "time_utc": utc_now_str(),
    }
    save_json(os.path.join(out_subdir, "error.json"), payload)
    with open(os.path.join(os.path.dirname(out_subdir), "_errors.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@dataclass
class RunConfig:
    image_dir: str
    out_dir: str
    run_name: str
    recursive: bool
    openrouter_api_key: str
    model_name: str
    generator_cfg: "GeneratorConfig"
    start_idx: int
    end_idx: int
    shard_id: int
    num_shards: int
    skip_existing: bool
    overwrite: bool
    gc_every: int
    sleep_sec: float
    models_dir: str | None = None
    checkpoint_path: str | None = None


def build_manifest(paths: List[str], cfg: RunConfig) -> Dict[str, Any]:
    from .io_utils import sha256_jsonable, utc_now_str

    manifest = {
        "created_at_utc": utc_now_str(),
        "image_dir": cfg.image_dir,
        "out_dir": cfg.out_dir,
        "run_name": cfg.run_name,
        "recursive": cfg.recursive,
        "model_name": cfg.model_name,
        "generator_cfg": cfg.generator_cfg.to_dict(),
        "start_idx": cfg.start_idx,
        "end_idx": cfg.end_idx,
        "shard_id": cfg.shard_id,
        "num_shards": cfg.num_shards,
        "skip_existing": cfg.skip_existing,
        "overwrite": cfg.overwrite,
        "gc_every": cfg.gc_every,
        "sleep_sec": cfg.sleep_sec,
        "models_dir": cfg.models_dir,
        "checkpoint_path": cfg.checkpoint_path,
        "inputs": paths,
    }
    manifest["manifest_hash"] = sha256_jsonable(manifest)
    return manifest


def run_one(generator: "COCOTreeGenerator", cfg: RunConfig, image_path: str) -> Dict[str, Any]:
    from PIL import Image

    out_sub, stem = output_subdir(cfg.out_dir, cfg.run_name, image_path)
    if cfg.skip_existing and (not cfg.overwrite) and is_done(out_sub):
        return {"status": "skipped", "image": image_path, "out": out_sub}

    pil_img = Image.open(image_path).convert("RGB")
    result = generator.generate(
        image=pil_img,
        image_out_dir=out_sub,
        openrouter_api_key=cfg.openrouter_api_key,
        model_name=cfg.model_name,
        title="",
        description="",
        filename=stem,
        image_source_path=image_path,
        overwrite=bool(cfg.overwrite),
    )
    return {"status": "done", "image": image_path, "out": out_sub, "summary": result.summary}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the COCOTree teacher pipeline")
    
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)

    ap.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--openrouter_api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--model_name", type=str, default="google/gemini-3-flash-preview")
    ap.add_argument("--models_dir", type=str, default=os.environ.get("SAM3_MODELS_DIR") or None)
    ap.add_argument("--checkpoint_path", type=str, default=None)

    ap.add_argument("--input_resize_longest_side", type=int, default=2048)
    ap.add_argument("--use_clahe", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--clahe_clip_limit", type=float, default=2.0)
    ap.add_argument("--clahe_tile_grid", type=int, default=8)

    ap.add_argument("--sam_capture_threshold", type=float, default=0.4)
    ap.add_argument("--root_selection_threshold", type=float, default=0.5)
    ap.add_argument("--child_selection_threshold_default", type=float, default=0.5)
    ap.add_argument("--child_selection_threshold_small", type=float, default=0.4)
    ap.add_argument("--child_small_area_pct_of_full_image", type=float, default=5.0)
    ap.add_argument("--selection_mode", type=str, default="per_candidate", choices=["per_candidate", "median"])

    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--min_area_pct", type=float, default=0.0)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_children", type=int, default=20)
    ap.add_argument("--max_total_nodes", type=int, default=1000)
    ap.add_argument("--llm_batch_size", type=int, default=32)
    ap.add_argument("--llm_crop_max_side", type=int, default=1500)
    
    ap.add_argument("--root_max_mask_area_frac", type=float, default=0.7)
    ap.add_argument("--candidate_cover_parent_max", type=float, default=0.9)
    ap.add_argument("--parent_child_iou_max", type=float, default=0.9)
    ap.add_argument("--sibling_containment_thr", type=float, default=0.9)

    ap.add_argument("--subcrop_split_fill_ratio", type=float, default=4.0)
    ap.add_argument("--subcrop_min_cc_area_frac", type=float, default=0.02)
    ap.add_argument("--subcrop_min_cc_area_floor", type=int, default=50)
    ap.add_argument("--max_subcrops", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--others_min_component_area_frac", type=float, default=0.01)

    ap.add_argument("--debug_save", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_raw_hmaps", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_hmaps_vis", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_candidate_png", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_subcrop_debug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_llm_raw_responses", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_candidate_cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_raw_candidate_masks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_processed_candidate_masks", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--initial_temperature", type=float, default=0.0)
    ap.add_argument("--batch_temperature", type=float, default=0.0)
    ap.add_argument("--fix_temperature", type=float, default=0.0)
    ap.add_argument("--single_temperature", type=float, default=0.0)
    ap.add_argument("--initial_max_retry", type=int, default=5)
    ap.add_argument("--llm_max_retry", type=int, default=3)

    ap.add_argument("--gc_every", type=int, default=10)
    ap.add_argument("--sleep_sec", type=float, default=0.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not str(args.openrouter_api_key).strip():
        raise RuntimeError("Set OPENROUTER_API_KEY or pass --openrouter_api_key.")

    image_dir = os.path.abspath(args.image_dir)
    out_root = os.path.abspath(args.out_dir)
    run_name = sanitize_filename(args.run_name) if args.run_name else ""

    paths = list_images_in_dir(image_dir, recursive=bool(args.recursive))
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    
    paths = apply_slice(paths, args.start_idx, args.end_idx)
    paths = apply_shard(paths, args.shard_id, args.num_shards)
    if not paths:
        print("[DONE] nothing to process after slice/shard")
        return

    run_root = os.path.join(out_root, run_name) if run_name else out_root

    from .io_utils import ensure_dir, save_json
    from .types import GeneratorConfig

    ensure_dir(run_root)

    generator_cfg = GeneratorConfig(
        input_resize_longest_side=int(args.input_resize_longest_side),
        use_clahe=bool(args.use_clahe),
        clahe_clip_limit=float(args.clahe_clip_limit),
        clahe_tile_grid=int(args.clahe_tile_grid),
        sam_capture_threshold=float(args.sam_capture_threshold),
        root_selection_threshold=float(args.root_selection_threshold),
        child_selection_threshold_default=float(args.child_selection_threshold_default),
        child_selection_threshold_small=float(args.child_selection_threshold_small),
        child_small_area_pct_of_full_image=float(args.child_small_area_pct_of_full_image),
        selection_mode=str(args.selection_mode),
        pad=int(args.pad),
        min_area_pct=float(args.min_area_pct),
        max_depth=int(args.max_depth),
        max_children=int(args.max_children),
        max_total_nodes=int(args.max_total_nodes),
        llm_batch_size=int(args.llm_batch_size),
        llm_crop_max_side=int(args.llm_crop_max_side),
        root_max_mask_area_frac=float(args.root_max_mask_area_frac),
        candidate_cover_parent_max=float(args.candidate_cover_parent_max),
        parent_child_iou_max=float(args.parent_child_iou_max),
        sibling_containment_thr=float(args.sibling_containment_thr),
        subcrop_split_fill_ratio=float(args.subcrop_split_fill_ratio),
        subcrop_min_cc_area_frac=float(args.subcrop_min_cc_area_frac),
        subcrop_min_cc_area_floor=int(args.subcrop_min_cc_area_floor),
        max_subcrops=None if int(args.max_subcrops) <= 0 else int(args.max_subcrops),
        others_min_component_area_frac=float(args.others_min_component_area_frac),
        debug_save=bool(args.debug_save),
        save_raw_hmaps=bool(args.save_raw_hmaps),
        save_hmaps_vis=bool(args.save_hmaps_vis),
        save_candidate_png=bool(args.save_candidate_png),
        save_subcrop_debug=bool(args.save_subcrop_debug),
        save_llm_raw_responses=bool(args.save_llm_raw_responses),
        save_candidate_cache=bool(args.save_candidate_cache),
        save_raw_candidate_masks=bool(args.save_raw_candidate_masks),
        save_processed_candidate_masks=bool(args.save_processed_candidate_masks),
        initial_temperature=float(args.initial_temperature),
        batch_temperature=float(args.batch_temperature),
        fix_temperature=float(args.fix_temperature),
        single_temperature=float(args.single_temperature),
        initial_max_retry=int(args.initial_max_retry),
        llm_max_retry=int(args.llm_max_retry),
    )


    cfg = RunConfig(
        image_dir=image_dir,
        out_dir=out_root,
        run_name=run_name,
        recursive=bool(args.recursive),
        openrouter_api_key=args.openrouter_api_key,
        model_name=args.model_name,
        generator_cfg=generator_cfg,
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
        skip_existing=bool(args.skip_existing),
        overwrite=bool(args.overwrite),
        gc_every=max(1, int(args.gc_every)),
        sleep_sec=max(0.0, float(args.sleep_sec)),
        models_dir=args.models_dir,
        checkpoint_path=args.checkpoint_path,
    )


    manifest = build_manifest(paths, cfg)
    save_json(os.path.join(run_root, "run_manifest.json"), manifest)

    from .generator import COCOTreeGenerator

    generator = COCOTreeGenerator(
        cfg=generator_cfg,
        models_dir=cfg.models_dir,
        checkpoint_path=cfg.checkpoint_path,
        sam_capture_threshold=float(args.sam_capture_threshold),
        root_selection_threshold=float(args.root_selection_threshold),
        child_selection_threshold_default=float(args.child_selection_threshold_default),
        child_selection_threshold_small=float(args.child_selection_threshold_small),
        child_small_area_pct_of_full_image=float(args.child_small_area_pct_of_full_image),
    )

    print(f"[INFO] image_dir = {image_dir}")
    print(f"[INFO] out_root  = {out_root} / run_name='{run_name}'")
    print(f"[INFO] total_inputs(after slice/shard) = {len(paths)}")
    print(f"[INFO] shard = {cfg.shard_id}/{cfg.num_shards}")
    print(f"[INFO] idx_range = [{cfg.start_idx}, {cfg.end_idx}] (inclusive; end=-1 means last)")

    t0 = time.perf_counter()
    done = skipped = failed = 0
    for i, img_path in enumerate(paths, start=1):
        out_sub, _stem = output_subdir(cfg.out_dir, cfg.run_name, img_path)
        print(f"\n[RUN] {i}/{len(paths)} :: {img_path}")
        try:
            res = run_one(generator, cfg, img_path)
            if res["status"] == "done":
                done += 1
                print(f"[DONE] out={res['out']}")
            else:
                skipped += 1
                print(f"[SKIP] out={res['out']}")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {repr(exc)}")
            write_error(out_sub, img_path, repr(exc))
        finally:
            if i % cfg.gc_every == 0:
                gc.collect()
            if cfg.sleep_sec > 0:
                time.sleep(cfg.sleep_sec)

    dt = time.perf_counter() - t0
    print(f"\n[ALL DONE] done={done} skipped={skipped} failed={failed} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
