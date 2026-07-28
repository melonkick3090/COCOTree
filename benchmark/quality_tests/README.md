# Controlled metric checks

The self-contained public quality check is:

```bash
cocotree-benchmark quality-test \
  --protocol configs/paper_v1.json \
  --output results/quality_invariants.json
```

It runs four deterministic synthetic cases through the released metric
implementation: identity, parent rewiring, wrong-label replacement, and a
missing mask. The checks verify component isolation and expected score
direction without requiring the dataset.

These synthetic checks are software tests, not image-only baselines and not a
reproduction of the paper's dataset-scale controlled-degradation table. The
latter used separate semantic-tree and instance-tree perturbation artifacts;
combining them into one public prediction file would misstate their provenance.
