# Prediction Format

This document defines the public input contract for the COCOTree evaluator.
The format is intentionally independent of the dataset-construction pipeline:
a benchmark method supplies only its predicted labels, masks, and hierarchy.

## Image object

Each image is represented by one JSON object:

```json
{
  "schema": "cocotree_prediction_v1",
  "image_id": 13290,
  "method": {
    "name": "my-method",
    "version": "1.0",
    "checkpoint": "optional-checkpoint-identifier"
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
        "counts": [0, 4, 307196]
      }
    }
  ]
}
```

Required top-level fields:

| Field | Type | Meaning |
|---|---:|---|
| `schema` | string | Must be exactly `cocotree_prediction_v1`. |
| `image_id` | integer or digit string | Dataset image identifier. Zero padding is allowed for strings. |
| `nodes` | array | Zero or more predicted instance nodes. |

`method` is optional metadata. It remains in the submitted source artifact for
provenance but never changes the score.

Unknown optional metadata fields may be retained by a producer, but a method
must not rely on them to define the evaluated tree. The six node fields and
`segmentation` below are normative.

## Instance node

Every item in `nodes` represents exactly one predicted **mask instance**.

| Field | Type | Requirement |
|---|---:|---|
| `instance_id` | string | Non-empty and unique within the image. |
| `semantic_id` | string | Non-empty semantic-node identifier. Multiple instances may share it. |
| `label` | string | Non-empty open-vocabulary label. |
| `parent_instance_id` | string | An existing instance ID or the literal `ROOT`. |
| `parent_semantic_id` | string | An existing semantic ID or the literal `ROOT`. |
| `segmentation` | object | COCO run-length encoded binary mask. |

IDs are prediction-local and do not need to match COCOTree reference IDs.
They are nevertheless used for deterministic natural ordering before
assignment and for parent-candidate tie-breaking. Renaming IDs can therefore
change which pair is selected when two or more assignments have exactly equal
IoU, which can in turn change label or hierarchy components. Use stable IDs and
preserve the submitted artifact for reproducibility.

### Semantic and instance hierarchy

The format carries both levels deliberately:

- `semantic_id` groups instances that instantiate the same predicted semantic
  node.
- `parent_semantic_id` defines the semantic tree.
- `parent_instance_id` records the submitted parent of this particular mask
  instance.

When multiple nodes share a `semantic_id`, they must also share the same
normalized `label` and `parent_semantic_id`. They may have different
`instance_id`, `parent_instance_id`, and masks.

Each non-root parent reference must resolve within the same image. Both the
instance graph and semantic graph must be acyclic. Disconnected nodes, unknown
parents, self-parent edges, and cycles are validation errors.

The submitted instance graph is required so every prediction is explicit and
auditable. Under `paper_v1`, it is not by itself the graph used for OTQ BQ.
The evaluator independently derives each tree's effective instance parents
from its semantic hierarchy and masks:

1. start at the node's direct parent semantic level;
2. among instances at that level with positive mask IoU, choose the
   maximum-IoU instance, with stable instance-ID tie-breaking;
3. if that level has no positive-IoU instance, continue to the next semantic
   ancestor; and
4. if no ancestor level has positive overlap, attach the instance to `ROOT`.

`parent_semantic_id` remains the direct, normative hierarchy used for HPQ and
semantic-depth scopes. An alternate protocol may use submitted instance
parents directly, but its scores must not be labeled paper-profile results.

### Synthetic root

`ROOT` is a reserved literal:

- a top-level instance uses `"parent_instance_id": "ROOT"`;
- a top-level semantic node uses `"parent_semantic_id": "ROOT"`; and
- `ROOT` itself must not appear as a node, instance, label, or mask.

The root covers the full image conceptually but is not evaluated as a predicted
mask.

## COCO RLE masks

`segmentation` must be a COCO RLE object with:

```json
{
  "size": [480, 640],
  "counts": "compressed RLE string"
}
```

or:

```json
{
  "size": [480, 640],
  "counts": [0, 4, 307196]
}
```

Rules:

- `size` is `[height, width]`, not `[width, height]`;
- `counts` may be a standard compressed COCO RLE string or an uncompressed
  non-negative integer list;
- the decoded run lengths must cover exactly `height * width` pixels;
- masks use COCO's column-major/Fortran-order convention; and
- all nodes for one image must use one internally consistent canvas size; at
  evaluation time, that size must match the reference image canvas.

Polygon arrays, PNG paths, boxes, logits, and base64 bitmaps are not accepted as
substitutes. Convert them to RLE before validation so the evaluated artifact is
self-contained.

## Input sources

The evaluator accepts either a directory or JSONL prediction source.

### Directory source

The canonical layout is:

```text
predictions/
|-- 000000013290/
|   `-- prediction.json
`-- 000000013297/
    `-- prediction.json
```

For compatibility, the resolver also accepts:

```text
<12-digit-image-id>/tree.json
<12-digit-image-id>.json
```

Use `prediction.json` for newly released methods. Do not place more than one
candidate file for the same image in a submission.

### JSONL source

A `.jsonl` file may contain one complete `cocotree_prediction_v1` image object
per UTF-8 line. Blank lines are ignored. Duplicate normalized image IDs are
rejected.

JSONL is convenient for upload, while the directory form is easier to inspect
and repair. They are scored identically.

### Public generator adapter

The public COCOTree generator's native `tree.json` plus mask-bundle artifacts
can be converted without changing the original run:

```bash
cocotree-benchmark convert-pipeline \
  --input-root /path/to/generator_outputs \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/converted_predictions
```

For each manifest image, the adapter expects a numeric image directory with
`tree.json`. It follows each semantic node's `files.mask_bundle`, converts
`instances[*].rle` to instance nodes, and falls back to `merged.rle` as one
instance only when an instance list is absent. This fallback is recorded as a
conversion warning. The adapter derives `parent_instance_id` using the paper
positive-overlap ancestor rule and validates the serialized result.

The command writes canonical `<12-digit-id>/prediction.json` files and
`conversion_report.json` in the new output directory. It never modifies the
generator output. Missing images cause a nonzero exit, and existing converted
files require an explicit `--overwrite`.

## Manifest

The manifest is the authoritative image population. Preferred JSONL form:

```json
{"image_id": 13290}
{"image_id": "000000013297"}
```

Plain one-ID-per-line text is accepted as well:

```text
000000013290
000000013297
```

Integer `13290` and digit string `"000000013290"` normalize to the same image
ID. A manifest with duplicates or a non-numeric ID is invalid.

An absent prediction for a manifest image is an **empty prediction**, not an
ignored image. Predictions for images outside the manifest do not enter the
score and are surfaced by validation so accidental split leakage is visible.

## Empty prediction

Two cases are equivalent:

1. no prediction file/object exists for a manifest image; or
2. the image object exists with `"nodes": []`.

The evaluator retains the image in macro averaging and counts all of its
reference instances as false negatives.

## Complete multi-instance example

This example separates two person instances while sharing one semantic node:

```json
{
  "schema": "cocotree_prediction_v1",
  "image_id": 13290,
  "nodes": [
    {
      "instance_id": "person:0",
      "semantic_id": "person",
      "label": "person",
      "parent_instance_id": "ROOT",
      "parent_semantic_id": "ROOT",
      "segmentation": {
        "size": [4, 4],
        "counts": [0, 2, 2, 2, 10]
      }
    },
    {
      "instance_id": "person:1",
      "semantic_id": "person",
      "label": "person",
      "parent_instance_id": "ROOT",
      "parent_semantic_id": "ROOT",
      "segmentation": {
        "size": [4, 4],
        "counts": [8, 2, 2, 2, 2]
      }
    },
    {
      "instance_id": "hat:0",
      "semantic_id": "hat",
      "label": "hat",
      "parent_instance_id": "person:0",
      "parent_semantic_id": "person",
      "segmentation": {
        "size": [4, 4],
        "counts": [0, 1, 15]
      }
    }
  ]
}
```

The two `person` rows share a label and semantic parent, but they remain two
independent mask instances. `hat:0` is attached to the first person instance.

## Validation checklist

Before evaluation, run:

```bash
cocotree-benchmark validate \
  --predictions /path/to/predictions \
  --manifest /path/to/manifest.jsonl \
  --output validation_report.json
```

A valid submission has:

- one unambiguous image object per predicted image;
- unique instance IDs within each image;
- consistent label and parent semantic ID within each semantic group;
- resolvable parent references;
- acyclic instance and semantic trees;
- valid, non-empty COCO RLE on one internally consistent image canvas; and
- no reliance on reference IDs or evaluator feedback.

Successful schema validation does not imply high mask, label, or hierarchy
quality; it only guarantees that the prediction has a well-defined score.
Standalone `validate` does not load the reference, so reference-canvas
compatibility is additionally enforced when `evaluate` decodes the pairwise
masks.
