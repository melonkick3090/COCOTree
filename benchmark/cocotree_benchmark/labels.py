from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def normalize_label(value: str | None) -> str:
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(text.split()).lower()


class ExactLabelSimilarity:
    backend = "exact"

    def prepare(self, labels: Iterable[str]) -> None:
        del labels

    def similarity(self, left: str | None, right: str | None) -> float:
        left_norm = normalize_label(left)
        right_norm = normalize_label(right)
        return float(bool(left_norm) and left_norm == right_norm)


class SentenceTransformerLabelSimilarity:
    backend = "sentence_transformer_cosine"

    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        template: str,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The paper label backend requires the optional dependency. "
                "Install with: pip install -e '.[paper]'"
            ) from exc
        self.model_name = str(model_name)
        self.revision = str(revision)
        self.template = str(template)
        self.model = SentenceTransformer(
            self.model_name,
            revision=self.revision,
            device=device,
        )
        self.embeddings: dict[str, np.ndarray] = {}
        self.cache: dict[tuple[str, str], float] = {}

    def _format(self, normalized: str) -> str:
        try:
            return self.template.format(label=normalized)
        except Exception:
            return normalized

    def prepare(self, labels: Iterable[str]) -> None:
        missing = sorted(
            {
                normalize_label(label)
                for label in labels
                if normalize_label(label) and normalize_label(label) not in self.embeddings
            }
        )
        if not missing:
            return
        vectors = self.model.encode(
            [self._format(label) for label in missing],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for label, vector in zip(missing, np.asarray(vectors), strict=True):
            self.embeddings[label] = np.asarray(vector, dtype=np.float32).reshape(-1)

    def similarity(self, left: str | None, right: str | None) -> float:
        left_norm = normalize_label(left)
        right_norm = normalize_label(right)
        if not left_norm or not right_norm:
            return 0.0
        if left_norm == right_norm:
            return 1.0
        key = tuple(sorted((left_norm, right_norm)))
        if key in self.cache:
            return self.cache[key]
        self.prepare([left_norm, right_norm])
        left_vector = self.embeddings[left_norm]
        right_vector = self.embeddings[right_norm]
        denominator = max(
            float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector)),
            1e-12,
        )
        cosine = float(np.dot(left_vector, right_vector) / denominator)
        value = float(max(0.0, min(cosine, 1.0)))
        self.cache[key] = value
        return value


def build_label_scorer(
    protocol: dict[str, Any],
    *,
    device: str = "cpu",
) -> ExactLabelSimilarity | SentenceTransformerLabelSimilarity:
    config = dict(protocol.get("label_similarity", {}))
    backend = str(config.get("backend", "exact"))
    if backend == "exact":
        return ExactLabelSimilarity()
    if backend == "sentence_transformer_cosine":
        return SentenceTransformerLabelSimilarity(
            model_name=str(config["model"]),
            revision=str(config["revision"]),
            template=str(config.get("template", "{label}")),
            device=device,
        )
    raise ValueError(f"Unsupported label similarity backend: {backend}")

