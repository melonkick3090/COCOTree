from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cocotree_benchmark.io import (
    load_ground_truth_source,
    load_prediction_source,
    read_manifest,
)
from cocotree_benchmark.report import (
    prepare_evaluation_output,
    scored_ground_truth_artifact,
)
from cocotree_benchmark.adapters.pipeline_tree import convert_pipeline_image
from cocotree_benchmark.validation import PredictionValidationError


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"


def test_manifest_and_missing_prediction_policy(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image_id": 1}\n{"image_id": 2}\n', encoding="utf-8")
    image_ids = read_manifest(manifest)
    trees, report = load_prediction_source(
        TOY / "predictions" / "perfect",
        image_ids,
        missing_policy="empty",
        extra_policy="ignore",
    )
    assert list(image_ids) == [1, 2]
    assert trees[1].nodes
    assert trees[2].nodes == []
    assert report["empty_or_missing_images"] == 1
    assert report["missing_images"] == [2]


def test_extra_prediction_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image_id": 2}\n', encoding="utf-8")
    with pytest.raises(PredictionValidationError, match="outside the manifest"):
        load_prediction_source(
            TOY / "predictions" / "perfect",
            read_manifest(manifest),
        )


def test_duplicate_extra_jsonl_id_rejected_even_when_extras_allowed(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"image_id": 2}\n', encoding="utf-8")
    payload = json.loads(
        (
            TOY
            / "predictions"
            / "perfect"
            / "000000000001"
            / "prediction.json"
        ).read_text(encoding="utf-8")
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(payload) + "\n" + json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PredictionValidationError, match="duplicate prediction"):
        load_prediction_source(
            predictions,
            read_manifest(manifest),
            extra_policy="ignore",
        )


def test_multiple_directory_candidates_for_one_image_rejected(
    tmp_path: Path,
) -> None:
    source = tmp_path / "predictions"
    padded = source / "000000000001"
    plain = source / "1"
    padded.mkdir(parents=True)
    plain.mkdir(parents=True)
    fixture = (
        TOY
        / "predictions"
        / "perfect"
        / "000000000001"
        / "prediction.json"
    ).read_text(encoding="utf-8")
    (padded / "prediction.json").write_text(fixture, encoding="utf-8")
    (plain / "prediction.json").write_text(fixture, encoding="utf-8")
    with pytest.raises(
        PredictionValidationError,
        match="multiple candidate prediction files",
    ):
        load_prediction_source(source, [1])


def test_cli_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "result"
    command = [
        sys.executable,
        "-m",
        "cocotree_benchmark",
        "evaluate",
        "--ground-truth",
        str(TOY / "ground_truth"),
        "--predictions",
        str(TOY / "predictions" / "perfect"),
        "--manifest",
        str(TOY / "manifest.jsonl"),
        "--protocol",
        str(ROOT / "configs" / "toy_exact_v1.json"),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "metrics_summary.json").read_text(encoding="utf-8"))
    all_depth = next(
        row
        for row in summary["metrics_by_depth"]
        if row["depth_scope"] == "all"
    )
    assert all_depth["mean_otq"] == pytest.approx(1.0)
    for required in (
        "metrics_by_depth.csv",
        "metrics_per_image.csv",
        "metrics_per_image.jsonl",
        "matches.jsonl",
        "validation_report.json",
        "run_manifest.json",
    ):
        assert (output / required).is_file()

    refused = subprocess.run(
        [*command, "--no-matches"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "already contains evaluation files" in refused.stderr

    replaced = subprocess.run(
        [*command, "--no-matches", "--overwrite"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert replaced.returncode == 0, replaced.stderr
    assert not (output / "matches.jsonl").exists()


def test_pipeline_adapter_materializes_instance_parent(tmp_path: Path) -> None:
    image_dir = tmp_path / "000000000123"
    car_dir = image_dir / "root__car"
    wheel_dir = image_dir / "root__car__wheel"
    car_dir.mkdir(parents=True)
    wheel_dir.mkdir(parents=True)
    car_bundle = car_dir / "car.masks.json"
    wheel_bundle = wheel_dir / "wheel.masks.json"
    car_bundle.write_text(
        json.dumps(
            {
                "instances": [
                    {"rle": {"size": [2, 2], "counts": [0, 4]}}
                ]
            }
        ),
        encoding="utf-8",
    )
    wheel_bundle.write_text(
        json.dumps(
            {
                "instances": [
                    {"rle": {"size": [2, 2], "counts": [3, 1]}}
                ]
            }
        ),
        encoding="utf-8",
    )
    (image_dir / "tree.json").write_text(
        json.dumps(
            {
                "schema": "cocotree_pipeline",
                "root_children": ["n1"],
                "nodes": {
                    "n1": {
                        "node_id": "n1",
                        "label": "car",
                        "parent_id": None,
                        "files": {"mask_bundle": str(car_bundle)},
                    },
                    "n2": {
                        "node_id": "n2",
                        "label": "wheel",
                        "parent_id": "n1",
                        "files": {"mask_bundle": str(wheel_bundle)},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    converted, warnings = convert_pipeline_image(image_dir)
    assert not warnings
    assert converted["schema"] == "cocotree_prediction_v1"
    parents = {
        node["instance_id"]: node["parent_instance_id"]
        for node in converted["nodes"]
    }
    assert parents["n2#1"] == "n1#1"


def test_public_release_instance_rows_and_root_aliases(tmp_path: Path) -> None:
    release = tmp_path / "release"
    annotations = release / "annotations"
    annotations.mkdir(parents=True)
    rows = [
        {
            "image_id": 55,
            "instance_node_id": "n1#1",
            "semantic_node_id": "n1",
            "final_label": "person",
            "parent_instance_node_id": "n00000#1",
            "parent_semantic_node_id": "n00000",
            "segmentation": {"size": [2, 2], "counts": [0, 4]},
        },
        {
            "image_id": 55,
            "instance_node_id": "n2#1",
            "semantic_node_id": "n2",
            "final_label": "face",
            "parent_instance_node_id": "n1#1",
            "parent_instance_semantic_node_id": "n1",
            "segmentation": {"size": [2, 2], "counts": [0, 1, 3]},
        },
    ]
    (annotations / "instance_nodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    trees = load_ground_truth_source(release, [55])
    tree = trees[55]
    assert len(tree.nodes) == 2
    by_id = {node.instance_id: node for node in tree.nodes}
    assert by_id["n1#1"].parent_instance_id == "ROOT"
    assert by_id["n1#1"].parent_semantic_id == "ROOT"
    assert by_id["n2#1"].parent_semantic_id == "n1"
    assert by_id["n2#1"].semantic_depth == 2
    assert scored_ground_truth_artifact(release) == (
        annotations / "instance_nodes.jsonl"
    )


def test_output_overwrite_never_removes_unknown_files(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    user_file = output / "notes.txt"
    user_file.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-evaluator entries"):
        prepare_evaluation_output(output, overwrite=True)
    assert user_file.read_text(encoding="utf-8") == "keep me"
