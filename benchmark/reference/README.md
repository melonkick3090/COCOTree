# Frozen 1K reference shard

`cocotree_1k_v1.jsonl.gz` is the exact annotation shard used for the frozen
1,000-image paper/rebuttal evaluation. It contains masks and tree metadata
only; original COCO pixels are not required to run the evaluator.

Provenance and integrity:

| Item | Value |
|---|---|
| Images | 1,000 |
| All mask-instance nodes | 84,117 |
| Normalized `others` mask nodes | 997 |
| OTQ-scored mask instances | 83,120 |
| Semantic nodes | 26,121 |
| Split manifest SHA-256 | `49cc2f0ab1c54a01a02be27e48987fde7487946bc49b5069f10ad600b27466ef` |
| Uncompressed JSONL SHA-256 | `a790b0c8fc738783f0d00dd955dc6d6698ec6600add0667042c7f1568c6be9d6` |
| Compressed file SHA-256 | `94c36e9e2f25e9c38d1684e73b5f3645280a2e55168ed23e182fd4745ccf26a9` |

The shard was deterministically materialized on 2026-07-28 from the archived
canonical per-image evaluation inputs used for the reported 1K results.
`scripts/build_reference_shard.py` records the serialization procedure.

This shard is intentionally versioned separately from the current public
annotation-only Hugging Face release. At revision
`ff59a3ed0fdecb04004e5ef34b047e6f81d2a602`, that release does not contain 27
IDs in the pre-existing frozen 1K manifest and does not store the legacy
top-level `others` rows. Silently evaluating the remaining 973 images would
change the benchmark denominator and would not reproduce the reported result.

The reference annotations are distributed under CC BY 4.0, independently of
the MIT-licensed evaluator code. See `THIRD_PARTY_NOTICES.md`.
