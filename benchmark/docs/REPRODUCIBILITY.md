# Reproducibility Protocol

This checklist defines a paper-comparable COCOTree benchmark run. It is designed
to make failures visible and to prevent silent changes in the evaluated image
population.

## 1. Freeze the evaluation population

Create or obtain an immutable manifest before generating predictions:

```json
{"image_id": 13290}
{"image_id": 13297}
```

Record:

- manifest file checksum;
- COCOTree release identifier and checksum;
- number of manifest images; and
- whether the run is full-set, public subset, or a development split.

Do not discover the evaluation set by listing successful prediction folders.
The manifest, not the available outputs, defines the denominator.

## 2. Preserve the model output

Write one `cocotree_prediction_v1` object per image. Preserve the exact artifact
that is evaluated, including empty `nodes` arrays. Do not replace a failed
image with a later output without recording the retry policy.

Recommended provenance in the optional `method` object:

```json
{
  "name": "my-method",
  "version": "1.0",
  "checkpoint": "model-or-checkpoint-revision",
  "postprocess": "declared-policy-name"
}
```

This metadata does not affect the score. It helps distinguish model inference
from deterministic post-processing and later repairs.

If starting from the public COCOTree generator format, preserve the native
output and convert additively:

```bash
cocotree-benchmark convert-pipeline \
  --input-root /path/to/generator_outputs \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/converted_predictions
```

Archive `conversion_report.json` with the evaluated predictions. Review every
merged-mask fallback warning and keep the original generator artifact available
for audit.

## 3. Record the software environment

From the benchmark directory:

```bash
cocotree-benchmark doctor --paper
python --version
python -m pip freeze > environment.txt
git rev-parse HEAD > benchmark_commit.txt
```

For the paper LQ profile, also record:

- `sentence-transformers/all-MiniLM-L6-v2`;
- pinned Hugging Face revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, plus the checksum of a locally
  archived model directory when applicable;
- the label template `visual segmentation label: {label}`;
- compute device; and
- relevant PyTorch, Transformers, and Sentence Transformers versions.

Using the same model name without pinning its resolved files is insufficient
for long-term archival reproduction.

## 4. Validate before scoring

```bash
cocotree-benchmark validate \
  --predictions /path/to/predictions \
  --manifest /path/to/manifest.jsonl \
  --output validation_report.json
```

Resolve all validation errors. Keep warnings, including extra prediction IDs,
in the run record. Never delete missing IDs from the manifest merely to make
validation look complete.

## 5. Run the paper protocol

```bash
cocotree-benchmark evaluate \
  --ground-truth /path/to/COCOTree \
  --predictions /path/to/predictions \
  --manifest /path/to/manifest.jsonl \
  --output-dir results/my_method
```

The paper-comparable profile uses:

| Setting | Value |
|---|---|
| Assignment | Global one-to-one Hungarian matching |
| Match threshold | Mask IoU >= 0.5 |
| LQ model | `sentence-transformers/all-MiniLM-L6-v2` |
| LQ revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| LQ template | `visual segmentation label: {label}` |
| Image aggregation | Macro mean over manifest images |
| Missing manifest prediction | Empty predicted tree |
| Synthetic root | Structural only; no root mask |
| OTQ instance-parent policy | Nearest positive-overlap semantic ancestor, maximum IoU |
| OTQ `others` policy | Exclude normalized `others` mask nodes |
| HPQ `others` policy | Keep top bucket and promote its direct semantic children |

If any setting differs, label the result as an ablation or alternate protocol,
not a directly comparable paper-profile result.

The label device may change runtime but must not change the mathematical
protocol. For strict archival reproduction, rerun a small fixture on CPU and
the selected accelerator and compare outputs within the reported numeric
tolerance.

## 6. Reproduce metric-quality checks

Synthetic invariants evaluate metric behavior, not model ability:

```bash
cocotree-benchmark quality-test \
  --protocol configs/paper_v1.json \
  --output results/quality_invariants.json
```

The command evaluates a committed synthetic tree and checks:

- an identity prediction gives one for all primary metrics;
- parent rewiring preserves MQ/LQ and lowers TQ;
- wrong-label replacement preserves matching/TQ and lowers LQ/OTQ; and
- removing a mask adds an FN and lowers TQ.

To keep this regression check small and offline, `quality-test` substitutes an
exact-label scorer. It validates the metric decomposition and code path, not
the numerical behavior of the paper MiniLM checkpoint.

These are implementation invariants, not a reproduction of every
dataset-scale controlled-degradation table. For a dataset-scale perturbation
study, keep the original reference immutable and record the perturbation name,
severity, seed, and affected image/node counts in an isolated output. Do not
select a perturbation or pruning rule because it happens to improve OTQ. Any
derived reference view needs a predeclared, deterministic rule and must be
reported separately from the canonical release.

## 7. Archive the run

Archive these artifacts together:

```text
run/
|-- manifest.jsonl
|-- predictions/ or predictions.jsonl
|-- conversion_report.json (when an adapter was used)
|-- results/
|-- environment.txt
|-- benchmark_commit.txt
`-- command.txt
```

The results archive should preserve:

- summary metrics and counts;
- per-image metrics;
- optional per-match audit rows;
- validation diagnostics;
- protocol/configuration metadata;
- elapsed time and software versions; and
- hashes for manifest and evaluated prediction source.

The exact output filenames are less important than retaining all of these
records in a machine-readable form.

## 8. Reporting checklist

In a paper or model card, state:

- dataset release and split/manifest;
- number of evaluated images;
- prediction source and post-processing policy;
- submitted hierarchy and evaluator instance-parent policy;
- HPQ, OTQ, TQ, BQ, meanNQ, MQ, and LQ;
- TP, FP, FN, predicted nodes, and predicted masks;
- IoU match threshold;
- LQ model, template, and pinned revision;
- handling of inference failures and missing images;
- whether scores are paper profile, an ablation, or an oracle analysis; and
- the canonical silver-reference limitation.

Avoid presenting a self-consistency or oracle experiment as an image-only
baseline. If reference labels, masks, nodes, paths, depths, or evaluator
feedback enter prediction generation, disclose that input and categorize the
result separately.

## Expected deterministic invariants

The following must hold across valid implementations:

- reordering nodes or JSONL rows does not change scores because instances use
  natural-ID ordering before matching;
- renaming prediction-local IDs can change the chosen match or parent only in
  a documented exact-IoU tie, so IDs must remain stable in archived outputs;
- a missing manifest image and an explicit empty prediction score identically;
- adding an out-of-manifest prediction does not improve the result;
- an exact copy of a reference view achieves its declared perfect-score
  identity checks; and
- controlled degradation fixtures do not mutate the source reference.

Run the repository test suite before publishing changes:

```bash
python -m pytest
```

## Silver-reference statement

A reproducible score can still reflect reference ambiguity. COCOTree makes a
particular annotation policy explicit and evaluates agreement with its
canonical silver reference. Multiple visually supported decompositions may
differ in grouping, intermediate levels, or stopping depth. Reproducibility
means that the agreement measurement can be repeated exactly; it does not turn
the reference into the only possible semantic decomposition.
