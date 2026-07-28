# Frozen split manifests

`cocotree_1k_v1.jsonl` is the exact 1,000-image manifest used for the rebuttal
baseline evaluation. The physical internal pilot directory contains additional
images, so evaluation must use this manifest rather than directory discovery.

This is the pre-existing human-validation/test subset, frozen before baseline
scoring. It was not reconstructed as the first 1,000 directory entries and was
not reselected according to benchmark results. The file preserves its original
order so every method is evaluated on the same population.

The source manifest SHA-256 recorded by the generation runs is:

```text
49cc2f0ab1c54a01a02be27e48987fde7487946bc49b5069f10ad600b27466ef
```

Each line preserves the image ID and source-image provenance used to freeze the
run. Paths are provenance strings only; the evaluator uses `image_id`.
The original absolute `source_path` strings are deliberately preserved
byte-for-byte so the historical manifest checksum remains verifiable. They are
not expected to exist on another machine and are never opened by the evaluator.

Use this manifest with `reference/cocotree_1k_v1.jsonl.gz`. The current public
annotation-only Hugging Face revision is a distinct release and lacks 27 of
these IDs; do not silently reduce this split to 973 images.
