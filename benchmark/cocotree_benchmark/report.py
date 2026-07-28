from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .protocol import file_sha256, payload_sha256


EVALUATION_OUTPUT_FILES = {
    "matches.jsonl",
    "metrics_by_depth.csv",
    "metrics_per_image.csv",
    "metrics_per_image.jsonl",
    "metrics_summary.json",
    "run_manifest.json",
    "validation_report.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def path_digest(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return file_sha256(target)
    digest = hashlib.sha256()
    for file_path in sorted(
        (item for item in target.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    ):
        relative = file_path.relative_to(target).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(file_path)))
    return digest.hexdigest()


def scored_ground_truth_artifact(path: str | Path) -> Path:
    target = Path(path)
    release_rows = target / "annotations" / "instance_nodes.jsonl"
    return release_rows if release_rows.is_file() else target


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "pycocotools",
        "sentence-transformers",
        "transformers",
        "huggingface-hub",
        "torch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def prepare_evaluation_output(
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> None:
    root = Path(output_dir)
    if not root.exists():
        return
    if not root.is_dir():
        raise FileExistsError(f"Evaluation output is not a directory: {root}")
    entries = list(root.iterdir())
    if not entries:
        return
    unknown = sorted(
        entry.name
        for entry in entries
        if entry.name not in EVALUATION_OUTPUT_FILES or not entry.is_file()
    )
    if unknown:
        raise FileExistsError(
            f"Output directory contains non-evaluator entries; refusing to "
            f"modify it: {root}. First unknown entry: {unknown[0]}"
        )
    if not overwrite:
        raise FileExistsError(
            f"Output directory already contains evaluation files: {root}. "
            "Choose a new directory or pass --overwrite."
        )
    for entry in entries:
        entry.unlink()


def build_run_manifest(
    *,
    protocol: dict[str, Any],
    protocol_path: str | Path,
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    output_dir: str | Path,
    image_count: int,
    label_device: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    ground_truth_artifact = scored_ground_truth_artifact(ground_truth_path)
    return {
        "schema": "cocotree_benchmark_run_manifest_v1",
        "created_at_utc": utc_now(),
        "benchmark_version": __version__,
        "protocol_id": protocol["protocol_id"],
        "protocol": protocol,
        "protocol_sha256": payload_sha256(protocol),
        "protocol_file": str(Path(protocol_path).resolve()),
        "protocol_file_sha256": file_sha256(protocol_path),
        "split_manifest": str(Path(manifest_path).resolve()),
        "split_manifest_sha256": file_sha256(manifest_path),
        "ground_truth": str(Path(ground_truth_path).resolve()),
        "ground_truth_scored_artifact": str(ground_truth_artifact.resolve()),
        "ground_truth_sha256": path_digest(ground_truth_artifact),
        "predictions": str(Path(prediction_path).resolve()),
        "predictions_sha256": path_digest(prediction_path),
        "output_dir": str(Path(output_dir).resolve()),
        "num_manifest_images": int(image_count),
        "label_device": str(label_device),
        "elapsed_seconds": float(elapsed_seconds),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
    }


def write_evaluation_outputs(
    output_dir: str | Path,
    *,
    summary_rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    validation_report: dict[str, Any],
    run_manifest: dict[str, Any],
    write_matches: bool = True,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "metrics_summary.json",
        {
            "schema": "cocotree_metrics_summary_v1",
            "protocol_id": run_manifest["protocol_id"],
            "metrics_by_depth": summary_rows,
        },
    )
    write_csv(root / "metrics_by_depth.csv", summary_rows)
    write_csv(root / "metrics_per_image.csv", per_image_rows)
    write_jsonl(root / "metrics_per_image.jsonl", per_image_rows)
    if write_matches:
        write_jsonl(root / "matches.jsonl", match_rows)
    write_json(root / "validation_report.json", validation_report)
    write_json(root / "run_manifest.json", run_manifest)
