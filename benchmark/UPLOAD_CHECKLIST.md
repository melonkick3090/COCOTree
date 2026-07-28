# Maintainer upload checklist

Upload this directory as `benchmark/` in the public COCOTree repository. The
compressed 1K reference is about 9.9 MB, below GitHub's individual-file limit;
Git LFS is not required.

Before committing:

```bash
cd benchmark
python -m pip install -e ".[paper,test]"
cocotree-benchmark doctor --paper
python -m pytest
cocotree-benchmark quality-test \
  --protocol configs/paper_v1.json \
  --output /tmp/cocotree_quality_invariants.json
```

Then run the exact 1K identity check:

```bash
cocotree-benchmark evaluate \
  --ground-truth reference/cocotree_1k_v1.jsonl.gz \
  --predictions reference/cocotree_1k_v1.jsonl.gz \
  --manifest splits/cocotree_1k_v1.jsonl \
  --protocol configs/paper_v1.json \
  --output-dir /tmp/cocotree_identity_1k \
  --no-matches
```

The `all` row must contain 1,000 images, 83,120 scored masks, 26,121 semantic
nodes, TP=83,120, FP=0, FN=0, and all seven primary quality values equal to
1.0.

Add a visible section to the public repository's root `README.md`:

```markdown
## Benchmark evaluation

The standalone OTQ/HPQ evaluator, prediction schema, frozen 1K split, exact
reference shard, and reproducibility instructions are in
[`benchmark/`](benchmark/README.md).
```

Finally:

- inspect `git status --short` so no benchmark file is omitted;
- do not include caches, local result directories, model weights, or secrets;
- preserve `reference/README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`;
- replace the rebuttal draft's placeholder with the final public URL; and
- tag or record the uploaded commit used for reported results.
