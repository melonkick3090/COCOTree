from __future__ import annotations

import pytest

from cocotree_benchmark.labels import SentenceTransformerLabelSimilarity


@pytest.mark.paper
def test_pinned_minilm_nonidentical_label_golden() -> None:
    pytest.importorskip("sentence_transformers")
    scorer = SentenceTransformerLabelSimilarity(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        template="visual segmentation label: {label}",
        device="cpu",
    )
    assert scorer.similarity("fire-hydrant", "fire_hydrant") == 1.0
    assert scorer.similarity("car", "automobile") == pytest.approx(
        0.9622997045516968,
        abs=2e-6,
    )
    assert scorer.similarity("car", "banana") == pytest.approx(
        0.6691741943359375,
        abs=2e-6,
    )
