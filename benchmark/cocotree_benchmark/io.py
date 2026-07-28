from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import ImageTree
from .protocol import file_sha256
from .validation import (
    PredictionValidationError,
    normalize_image_id,
    parse_prediction_payload,
    release_rows_to_image_tree,
)


def _open_text(path: str | Path):
    source = Path(path)
    if source.suffix.lower() == ".gz":
        return gzip.open(source, "rt", encoding="utf-8")
    return source.open("r", encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, payload


def read_manifest(path: str | Path) -> list[int]:
    manifest_path = Path(path)
    image_ids: list[int] = []
    seen: set[int] = set()
    with _open_text(manifest_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                try:
                    value = json.loads(text).get("image_id")
                except (json.JSONDecodeError, AttributeError) as exc:
                    raise ValueError(
                        f"{manifest_path}:{line_number}: invalid manifest JSON"
                    ) from exc
            else:
                value = text
            image_id = normalize_image_id(value)
            if image_id in seen:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate image_id {image_id}"
                )
            seen.add(image_id)
            image_ids.append(image_id)
    if not image_ids:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return image_ids


def prediction_path_candidates(root: Path, image_id: int) -> list[Path]:
    padded = f"{int(image_id):012d}"
    plain = str(int(image_id))
    return [
        root / padded / "prediction.json",
        root / padded / "tree.json",
        root / f"{padded}.json",
        root / plain / "prediction.json",
        root / plain / "tree.json",
        root / f"{plain}.json",
    ]


def discover_directory_image_ids(root: Path) -> set[int]:
    found: set[int] = set()
    for entry in root.iterdir():
        stem = entry.stem if entry.is_file() else entry.name
        if stem.isdigit():
            found.add(int(stem))
    return found


def load_prediction_source(
    source: str | Path,
    manifest_ids: list[int],
    *,
    missing_policy: str = "empty",
    extra_policy: str = "error",
) -> tuple[dict[int, ImageTree], dict[str, Any]]:
    source_path = Path(source)
    requested = set(manifest_ids)
    trees: dict[int, ImageTree] = {}
    invalid: list[dict[str, Any]] = []
    source_ids: set[int] = set()
    seen_source_ids: set[int] = set()

    def accept(payload: dict[str, Any], location: str) -> None:
        try:
            tree = parse_prediction_payload(payload)
        except Exception as exc:
            invalid.append({"location": location, "error": str(exc)})
            return
        source_ids.add(tree.image_id)
        if tree.image_id in seen_source_ids:
            invalid.append(
                {
                    "location": location,
                    "error": f"duplicate prediction for image {tree.image_id}",
                }
            )
            return
        seen_source_ids.add(tree.image_id)
        if tree.image_id in requested:
            trees[tree.image_id] = tree

    if source_path.is_file():
        for line_number, payload in read_jsonl(source_path):
            accept(payload, f"{source_path}:{line_number}")
    elif source_path.is_dir():
        source_ids = discover_directory_image_ids(source_path)
        for image_id in sorted(requested & source_ids):
            candidates = prediction_path_candidates(source_path, image_id)
            existing_candidates = list(
                {
                    path.resolve(): path
                    for path in candidates
                    if path.is_file()
                }.values()
            )
            if not existing_candidates:
                invalid.append(
                    {
                        "location": str(source_path),
                        "error": (
                            f"numeric image entry exists for {image_id}, but no "
                            "prediction.json or tree.json was found"
                        ),
                    }
                )
                continue
            if len(existing_candidates) > 1:
                invalid.append(
                    {
                        "location": str(source_path),
                        "error": (
                            f"multiple candidate prediction files exist for "
                            f"image {image_id}: "
                            + ", ".join(
                                str(path)
                                for path in existing_candidates
                            )
                        ),
                    }
                )
                continue
            prediction_path = existing_candidates[0]
            try:
                payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            except Exception as exc:
                invalid.append({"location": str(prediction_path), "error": str(exc)})
                continue
            accept(payload, str(prediction_path))
    else:
        raise FileNotFoundError(f"Prediction source does not exist: {source_path}")

    extras = sorted(source_ids - requested)
    missing = sorted(requested - set(trees))
    if invalid:
        first = invalid[0]
        raise PredictionValidationError(
            f"{len(invalid)} invalid prediction(s); first: "
            f"{first['location']}: {first['error']}"
        )
    if extras and extra_policy == "error":
        raise PredictionValidationError(
            f"Predictions contain {len(extras)} image(s) outside the manifest; "
            f"first={extras[0]}"
        )
    if missing and missing_policy != "empty":
        raise PredictionValidationError(
            f"Predictions are missing {len(missing)} manifest image(s)"
        )
    for image_id in missing:
        trees[image_id] = ImageTree(image_id=image_id, nodes=[])
    return trees, {
        "requested_images": len(manifest_ids),
        "provided_images": len(requested & source_ids),
        "valid_nonempty_images": sum(bool(trees[i].nodes) for i in manifest_ids),
        "empty_or_missing_images": sum(not trees[i].nodes for i in manifest_ids),
        "missing_images": missing,
        "extra_images": extras,
        "invalid_predictions": invalid,
        "source": str(source_path.resolve()),
        "source_sha256": file_sha256(source_path) if source_path.is_file() else None,
    }


def load_release_ground_truth(
    release_root: str | Path,
    manifest_ids: list[int],
) -> dict[int, ImageTree]:
    root = Path(release_root)
    instance_path = root / "annotations" / "instance_nodes.jsonl"
    if not instance_path.is_file():
        raise FileNotFoundError(
            "COCOTree release must contain annotations/instance_nodes.jsonl: "
            f"{root}"
        )
    requested = set(manifest_ids)
    rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line_number, row in read_jsonl(instance_path):
        try:
            image_id = normalize_image_id(row.get("image_id"))
        except Exception as exc:
            raise ValueError(f"{instance_path}:{line_number}: {exc}") from exc
        if image_id in requested:
            rows_by_image[image_id].append(row)
    missing = [image_id for image_id in manifest_ids if not rows_by_image[image_id]]
    if missing:
        raise ValueError(
            f"Ground truth is missing {len(missing)} manifest image(s); "
            f"first={missing[0]}"
        )
    return {
        image_id: release_rows_to_image_tree(image_id, rows_by_image[image_id])
        for image_id in manifest_ids
    }


def load_ground_truth_source(
    source: str | Path,
    manifest_ids: list[int],
) -> dict[int, ImageTree]:
    source_path = Path(source)
    if (source_path / "annotations" / "instance_nodes.jsonl").is_file():
        return load_release_ground_truth(source_path, manifest_ids)
    trees, report = load_prediction_source(
        source_path,
        manifest_ids,
        missing_policy="error",
        extra_policy="ignore",
    )
    if report["empty_or_missing_images"]:
        raise ValueError("Ground-truth source contains empty image trees")
    return trees
