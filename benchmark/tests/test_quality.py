from __future__ import annotations

from pathlib import Path

from cocotree_benchmark.protocol import load_protocol
from cocotree_benchmark.quality import run_quality_invariants


ROOT = Path(__file__).resolve().parents[1]


def test_controlled_quality_invariants() -> None:
    report = run_quality_invariants(
        load_protocol(ROOT / "configs" / "toy_exact_v1.json")
    )
    assert report["passed"]
    assert all(report["checks"].values())

