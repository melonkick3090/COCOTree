# Frozen regression evidence

The strict 1K adapter and identity checks were run on 2026-07-28 in the
research environment without modifying the source dataset.

Verified adapter counts:

| Item | Expected |
| --- | ---: |
| Manifest images | 1,000 |
| All loaded instance masks | 84,117 |
| Plain `others` masks | 997 |
| OTQ-scored masks | 83,120 |
| HPQ semantic nodes under the paper legacy policy | 26,121 |

`strict1k_identity_paper_v1/` records a GT-vs-GT run through the public CLI.
At `all` depth, HPQ, OTQ, TQ, BQ, meanNQ, MQ, and LQ are all exactly 1.0.
At `<=1`, one image contains no OTQ-scored mask after the explicit `others`
exclusion, so the macro OTQ component means are 0.999 by the frozen empty-set
convention; this is expected rather than a failed match.

The exact compressed canonical input is committed as
`reference/cocotree_1k_v1.jsonl.gz`; original image pixels are not needed.
Per-image generator outputs can also be converted with `convert-pipeline` when
native `tree.json` files and their referenced mask bundles are available. The
annotation-only Hugging Face release is instead read directly by the evaluator.
