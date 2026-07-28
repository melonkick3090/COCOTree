# COCOTree Benchmark

This directory contains the standalone evaluation package for COCOTree. It is
the code path for benchmarking a predicted open-vocabulary instance tree
against the released COCOTree reference. It does not run the annotation
pipeline and does not require access to annotation-time prompts or model
outputs.

The evaluator reports the paper metrics:

- **OTQ**, with its TQ, BQ, meanNQ, MQ, and LQ components;
- **HPQ**, the recursive, exact-label hierarchical metric; and
- transparent coverage counts and per-image results.

COCOTree is a **canonical silver reference**, not a claim that every image has
one uniquely correct decomposition. The reported scores measure agreement with
the released reference under the published protocol. They should be
interpreted together with coverage and component metrics, especially when a
method deliberately chooses a different grouping, intermediate level, or
stopping depth.

## Installation

Python 3.11-3.12 is supported by the frozen dependency set.

```bash
cd benchmark
python -m pip install -e ".[paper]"
```

For development and tests:

```bash
python -m pip install -e ".[paper,test]"
python -m pytest
```

Installing the core package without `[paper]` is sufficient for validation and
the offline exact-label toy profile. The paper label backend downloads
`sentence-transformers/all-MiniLM-L6-v2` on first use unless that model is
already present in the local Hugging Face cache. Use a pinned local model
directory for an offline or archival run.

## One-minute offline smoke test

The committed toy fixture needs only the core dependencies and uses exact label
matching:

```bash
cocotree-benchmark validate \
  --predictions examples/toy/predictions/perfect \
  --manifest examples/toy/manifest.jsonl

cocotree-benchmark evaluate \
  --ground-truth examples/toy/ground_truth \
  --predictions examples/toy/predictions/perfect \
  --manifest examples/toy/manifest.jsonl \
  --protocol configs/toy_exact_v1.json \
  --output-dir results/toy_perfect
```

The perfect fixture must report 1.0 for HPQ, OTQ, TQ, BQ, meanNQ, MQ, and LQ
at every available depth scope. The exact-label toy protocol is a software
smoke test only; do not report it as the paper setting.

## Core workflow

Both command forms below are equivalent:

```bash
cocotree-benchmark --help
python -m cocotree_benchmark --help
```

### 1. Check the environment

```bash
cocotree-benchmark doctor --paper
```

`doctor` checks the dependencies and frozen versions needed to decode masks,
perform Hungarian matching, and run the paper label backend. Omit `--paper`
for the core-only exact-label toy profile. PyTorch is reported but not pinned
because its install target depends on the selected CPU/CUDA platform; its exact
resolved version is recorded in every run manifest.

### 2. Validate predictions

```bash
cocotree-benchmark validate \
  --predictions /path/to/my_predictions \
  --manifest /path/to/cocotree_test.jsonl \
  --output results/my_method_validation.json
```

Validation catches malformed RLE, duplicate instance identifiers, unknown
parents, semantic-parent inconsistencies, cycles, and image identifiers outside
the frozen manifest before metric computation begins.

### 3. Evaluate

```bash
cocotree-benchmark evaluate \
  --ground-truth /path/to/COCOTree \
  --predictions /path/to/my_predictions \
  --manifest /path/to/cocotree_test.jsonl \
  --output-dir results/my_method
```

`--ground-truth` points to a COCOTree release root containing
`annotations/instance_nodes.jsonl`. A canonical
`cocotree_prediction_v1` directory or JSONL can also serve as ground truth for
small fixtures.

For the committed frozen 1K split, the complete command is:

```bash
cocotree-benchmark evaluate \
  --ground-truth reference/cocotree_1k_v1.jsonl.gz \
  --predictions /path/to/my_predictions \
  --manifest splits/cocotree_1k_v1.jsonl \
  --protocol configs/paper_v1.json \
  --output-dir results/my_method_1k
```

The included compressed reference is the exact paper/rebuttal 1K evaluation
shard: 84,117 mask rows in total, of which 997 normalized `others` rows are
excluded from OTQ, leaving 83,120 scored instances and 26,121 semantic nodes.
Its provenance and hashes are in [reference/README.md](reference/README.md).
Original image pixels are not needed for metric evaluation.
Methods that need the RGB inputs to generate predictions should obtain the
corresponding COCO 2017 `train2017` images separately from the
[official COCO download page](https://cocodataset.org/#download); image files
are not redistributed in this benchmark package.

### Evaluate the current full public release

The annotation-only dataset is available at
[melonkick/COCOTree](https://huggingface.co/datasets/melonkick/COCOTree).
Pin the exact revision instead of downloading a moving branch:

```bash
python -m pip install -U huggingface_hub

hf download melonkick/COCOTree \
  --repo-type dataset \
  --revision ff59a3ed0fdecb04004e5ef34b047e6f81d2a602 \
  --local-dir data/COCOTree_ff59a3ed

cocotree-benchmark evaluate \
  --ground-truth data/COCOTree_ff59a3ed \
  --predictions /path/to/full_release_predictions \
  --manifest data/COCOTree_ff59a3ed/annotations/image_dual_tree_summary.jsonl \
  --protocol configs/release_v4.json \
  --output-dir results/my_method_release_v4
```

Keep `annotations/instance_nodes.jsonl` in its released location. The
`release_v4` profile is intentionally distinct from `paper_v1`: the current
release omits the legacy top-level `others` rows and lacks 27 IDs from the
pre-existing frozen 1K subset. A full-release result must therefore be labeled
`release_v4` and must not be presented as direct reproduction of a paper-1K
row. We provide the exact 1K shard rather than silently changing the paper
denominator to 973 images.

The default evaluation profile is the paper protocol:

- global Hungarian mask assignment;
- mask matching at IoU >= 0.5;
- `sentence-transformers/all-MiniLM-L6-v2` for LQ;
- pinned model revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
- label template `visual segmentation label: {label}`;
- paper-profile OTQ instance parents derived from the nearest positive-overlap
  semantic ancestor level;
- legacy paper label handling: exclude normalized `others` mask nodes from OTQ,
  while HPQ keeps the top-level bucket and promotes its direct children;
- macro averaging over the frozen image manifest; and
- a missing prediction for a manifest image treated as an empty prediction,
  never silently removed from the denominator.

Run `cocotree-benchmark evaluate --help` for the explicit protocol, device,
match-log, and extra-ID options.

### 4. Reproduce metric-quality checks

```bash
cocotree-benchmark quality-test \
  --protocol configs/paper_v1.json \
  --output results/quality_invariants.json
```

`quality-test` runs deterministic synthetic invariants through the same metric
implementation: identity, parent rewiring, wrong-label replacement, and a
missing mask. It verifies component isolation and score direction; it is not a
model baseline or a replacement for the paper's dataset-scale experiments. The
invariants intentionally substitute exact label matching so they run offline;
paper evaluation still uses the pinned MiniLM profile.

### Optional: convert public pipeline outputs

Outputs from the public COCOTree generator can be materialized into the
standalone prediction schema before validation:

```bash
cocotree-benchmark convert-pipeline \
  --input-root /path/to/generator_outputs \
  --manifest /path/to/cocotree_test.jsonl \
  --output-dir converted/my_method
```

The adapter reads each `<image_id>/tree.json` and its referenced mask bundles,
emits one instance node per bundled RLE, derives auditable instance parents,
and writes `conversion_report.json`. Missing image directories make the
command fail after writing the report. Existing outputs are never replaced
unless `--overwrite` is passed.

## Required prediction layout

The recommended directory layout is:

```text
my_predictions/
|-- 000000013290/
|   `-- prediction.json
|-- 000000013297/
|   `-- prediction.json
`-- ...
```

Each file contains one `cocotree_prediction_v1` image object:

```json
{
  "schema": "cocotree_prediction_v1",
  "image_id": 13290,
  "method": {
    "name": "example-method",
    "version": "1.0"
  },
  "nodes": [
    {
      "instance_id": "person:0",
      "semantic_id": "person",
      "label": "person",
      "parent_instance_id": "ROOT",
      "parent_semantic_id": "ROOT",
      "segmentation": {
        "size": [480, 640],
        "counts": "compressed COCO RLE string"
      }
    }
  ]
}
```

Prediction-local identifiers do not need to match reference identifiers. The
synthetic root is the literal `ROOT`; it is referenced by top-level nodes but
is never emitted as a node or mask. See
[Prediction format](docs/PREDICTION_FORMAT.md) for the complete contract,
including JSONL input and uncompressed RLE.

## Frozen manifest behavior

The manifest defines the evaluation population and its order. It is UTF-8
JSONL with one object per line:

```json
{"image_id": 13290}
{"image_id": 13297}
```

A plain text file containing one image ID per line is also accepted. IDs may be
integers or zero-padded digit strings. Duplicate IDs are rejected.

Predictions are resolved only for manifest images. If a manifest image is
absent from the prediction source, the evaluator inserts an empty prediction;
all reference instances for that image therefore remain false negatives.
This prevents accidental score inflation from skipped failures.

## What to publish with a benchmark result

For a result that another group can reproduce, publish:

1. the exact manifest;
2. the unmodified prediction JSON files or prediction JSONL;
3. the evaluator version or Git commit;
4. the complete evaluation command;
5. the generated summary, per-image table, and protocol metadata;
6. the label model name and pinned revision or local artifact checksum; and
7. any departure from the paper profile.

Do not tune post-processing thresholds on the test manifest. If multiple
prediction variants are evaluated, report the selection rule and retain all
variant results.

## Evaluation outputs

`evaluate --output-dir DIR` writes:

```text
DIR/
|-- metrics_summary.json
|-- metrics_by_depth.csv
|-- metrics_per_image.csv
|-- metrics_per_image.jsonl
|-- matches.jsonl
|-- validation_report.json
`-- run_manifest.json
```

Use `--no-matches` only when the potentially large per-match audit file is not
needed. The run manifest records checksums for the protocol, frozen split,
ground truth, and prediction source together with software versions and the
label device.

The evaluator refuses a nonempty output directory by default. Pass
`--overwrite` only to replace files produced by an earlier evaluator run; the
command still refuses to remove unknown files or subdirectories. This prevents
an old `matches.jsonl` from being mixed with a newer `--no-matches` result.

## Committed regression evidence

`regression/` records the verified public-format adapter counts and the compact
outputs of a frozen 1K GT-vs-GT identity run. `reference/` contains the exact
compressed input needed to repeat that run. The all-depth result contains
83,120 scored masks and 26,121 semantic nodes, with HPQ, OTQ, TQ, BQ, meanNQ,
MQ, and LQ all equal to 1.0. `convert-pipeline` can generate the same canonical
schema from native per-image generator outputs (`tree.json` plus mask bundles).
It is not a converter for the annotation-only Hugging Face release; that
release is loaded directly through `annotations/instance_nodes.jsonl`.

These artifacts are evaluator regression evidence, not a model baseline and
not evidence that the silver reference is the only valid decomposition.

## Metric interpretation

OTQ separates three questions:

- **Were the reference instances recovered?** The PQ-style recovery term.
- **Was their matched hierarchy preserved?** BQ.
- **Were matched masks and labels accurate?** meanNQ, composed from MQ and LQ.

Consequently, a method can expose whether a low score comes primarily from
coverage, hierarchy, masks, or labels. HPQ provides a complementary,
stricter recursive score that uses exact semantic labels.

The submitted instance hierarchy remains required and is validated. For paper
parity, however, BQ uses a deterministic effective instance parent derived
from the submitted semantic hierarchy and masks; see
[Metric definitions](docs/METRIC_DEFINITIONS.md#paper-profile-instance-parent-policy).

The synthetic root does not contribute a mask. Aggregate TP/FP/FN totals are
provided for auditing, but the reported means are macro-averaged over images;
in general, a macro mean cannot be reconstructed from pooled totals alone.
See [Metric definitions](docs/METRIC_DEFINITIONS.md).

## Repository map

```text
benchmark/
|-- cocotree_benchmark/   # evaluator and CLI
|-- configs/              # frozen paper and toy protocols
|-- docs/                 # schema, metrics, and reproduction protocol
|-- examples/             # minimal runnable prediction fixture
|-- quality_tests/        # scope and synthetic quality-check instructions
|-- reference/            # exact compressed frozen-1K silver reference
|-- regression/           # frozen adapter and 1K identity evidence
|-- schemas/              # machine-readable prediction contract
|-- scripts/              # deterministic release-maintenance utilities
|-- splits/               # frozen evaluation manifests
|-- tests/                # schema, matching, metric, and CLI tests
|-- LICENSE
|-- README.md
|-- THIRD_PARTY_NOTICES.md
|-- UPLOAD_CHECKLIST.md
`-- pyproject.toml
```

## Reproducibility and licensing

The exact protocol and reporting checklist are in
[Reproducibility](docs/REPRODUCIBILITY.md).

The benchmark code is released under the MIT License. The COCOTree dataset is
released under CC BY 4.0 and must be cited and attributed independently of the
code. Dependency and model notices are listed in
[Third-party notices](THIRD_PARTY_NOTICES.md).

The dataset-construction repository is
[melonkick3090/COCOTree](https://github.com/melonkick3090/COCOTree).
