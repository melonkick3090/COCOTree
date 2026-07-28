from __future__ import annotations

import json
from pathlib import Path

from cocotree_benchmark.cli import DEFAULT_PROTOCOL
from cocotree_benchmark.io import read_jsonl, read_manifest
from cocotree_benchmark.protocol import file_sha256, payload_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_default_protocol_matches_public_config() -> None:
    packaged = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    public = json.loads(
        (ROOT / "configs" / "paper_v1.json").read_text(encoding="utf-8")
    )
    assert packaged == public


def test_frozen_1k_manifest_identity() -> None:
    manifest = ROOT / "splits" / "cocotree_1k_v1.jsonl"
    assert len(read_manifest(manifest)) == 1000
    assert file_sha256(manifest) == (
        "49cc2f0ab1c54a01a02be27e48987fde"
        "7487946bc49b5069f10ad600b27466ef"
    )


def test_frozen_1k_reference_shard_identity() -> None:
    reference = ROOT / "reference" / "cocotree_1k_v1.jsonl.gz"
    assert file_sha256(reference) == (
        "94c36e9e2f25e9c38d1684e73b5f3645"
        "280a2e55168ed23e182fd4745ccf26a9"
    )
    rows = list(read_jsonl(reference))
    assert len(rows) == 1000
    assert sum(len(payload["nodes"]) for _, payload in rows) == 84117


def test_committed_1k_identity_regression() -> None:
    payload = json.loads(
        (
            ROOT
            / "regression"
            / "strict1k_identity_paper_v1"
            / "metrics_summary.json"
        ).read_text(encoding="utf-8")
    )
    all_depth = next(
        row
        for row in payload["metrics_by_depth"]
        if row["depth_scope"] == "all"
    )
    assert all_depth["num_images"] == 1000
    assert all_depth["num_gt_masks"] == 83120
    assert all_depth["num_gt_nodes"] == 26121
    assert all_depth["otq_tp"] == 83120
    assert all_depth["otq_fp"] == 0
    assert all_depth["otq_fn"] == 0
    for field in (
        "mean_hpq",
        "mean_otq",
        "mean_tq",
        "mean_bq",
        "mean_mnq",
        "mean_mq",
        "mean_lq",
    ):
        assert all_depth[field] == 1.0


def test_committed_regression_records_the_shipped_protocol() -> None:
    config_path = ROOT / "configs" / "paper_v1.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(
        (
            ROOT
            / "regression"
            / "strict1k_identity_paper_v1"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert run_manifest["protocol_file_sha256"] == file_sha256(config_path)
    assert run_manifest["protocol_sha256"] == payload_sha256(protocol)
