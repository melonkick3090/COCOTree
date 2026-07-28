# Metric Definitions

COCOTree evaluation compares a predicted open-vocabulary instance tree with the
released COCOTree canonical silver reference. This document describes the
public protocol and how to interpret its outputs. The implementation in
`cocotree_benchmark` is the normative executable specification.

## Evaluation unit and averaging

The primary matching unit is a mask instance. The synthetic `ROOT` is used only
to define topology and never contributes a mask.

Metrics are computed per manifest image and then macro-averaged across images.
TP, FP, and FN are additionally pooled as audit counts. Because the headline
scores are macro means, they generally cannot be reconstructed from pooled
counts alone.

If a manifest image has no prediction, it remains in the evaluation as an
empty predicted tree. The evaluator never silently skips such an image.

## Global mask matching

For each image:

1. Decode every reference and predicted instance mask on the common image
   canvas.
2. Compute the pairwise mask IoU matrix.
3. Solve one global one-to-one Hungarian assignment.
4. Retain assigned pairs with IoU >= 0.5 as true-positive matches.
5. Count all remaining predicted instances as FP and all remaining reference
   instances as FN.

Matching is global rather than parent-local. This prevents a hierarchy error
from hiding an otherwise valid mask match and lets mask/label quality be
reported separately from structural quality. The IoU threshold is fixed at
0.5 in the paper profile.

Before assignment, reference and prediction instances are sorted by natural
instance-ID order. This makes input JSON order irrelevant and provides a
deterministic policy for exactly tied IoU assignments. As a consequence,
renaming prediction-local IDs can change label or hierarchy components in the
rare case of an exact assignment tie.

The protocol field name `stable_input_order` refers to stable assignment on
these natural-ID-canonicalized arrays, not to raw JSON order.

Let `M` be the retained matches and let `TP = |M|`.

## Paper-profile instance-parent policy

The prediction schema requires `parent_instance_id`, and validation checks that
the submitted instance graph is resolvable and consistent with the semantic
ancestry. The paper profile nevertheless derives an **effective** instance
parent graph for OTQ BQ. This reproduces the canonical evaluator's handling of
multiple mask instances under one semantic node.

For every non-root instance, independently in the reference and predicted
trees:

1. inspect instances belonging to the direct parent semantic node;
2. if any have positive mask IoU with the child, select the maximum-IoU
   instance, using stable instance-ID order to break a tie;
3. otherwise climb to the next semantic ancestor and repeat; and
4. attach to the synthetic root if no semantic ancestor contains a
   positive-IoU instance.

The protocol identifier for this rule is
`derive_first_positive_semantic_ancestor_max_iou`. It prevents an unrelated
same-label instance elsewhere in the image from becoming a child's effective
parent merely because it shares a semantic node. The submitted instance parent
is still preserved as auditable method output, but it is not the sole
determinant of paper-profile BQ.

The direct `parent_semantic_id` remains normative for HPQ and semantic-depth
scopes.

## Mask quality: MQ

For a matched pair `i`, mask quality is its intersection-over-union:

```text
MQ_i = IoU(reference_mask_i, predicted_mask_i)
```

`MQ` is the mean of `MQ_i` over true-positive matches. Unmatched nodes affect
the recovery term rather than being assigned an artificial zero IoU inside
this matched-pair mean.

## Label quality: LQ

For a matched pair, LQ is the semantic similarity between its open-vocabulary
labels.

The paper backend is:

```text
model:    sentence-transformers/all-MiniLM-L6-v2
revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
template: visual segmentation label: {label}
```

The same template is applied to reference and predicted labels before
embedding. The paper profile uses cosine similarity between normalized
embeddings, clipped to `[0, 1]`; an exact normalized label match receives 1.
Label normalization lowercases text, replaces hyphens and underscores with
spaces, trims the ends, and collapses repeated whitespace.
The pinned revision above is part of the profile. For an offline archival run,
also record the checksum of the resolved local model artifact.

LQ is averaged over true-positive matches. It gives partial credit to
semantically related strings in a way exact string matching does not; that
similarity score alone does not establish that an alias is objectively valid.

## Node quality: meanNQ

Each true-positive match combines mask and label quality:

```text
NQ_i = MQ_i * LQ_i
meanNQ = mean over i in M of NQ_i
```

`meanNQ` is the mean of products, not necessarily `mean(MQ) * mean(LQ)`.
Therefore, do not attempt to recover it by multiplying two rounded summary
columns.

## Branch quality: BQ

BQ measures whether the hierarchy among the globally matched instances is
preserved.

After applying the paper-profile effective-parent policy, the evaluator
projects the reference and prediction onto their matched-node skeletons and
compares ancestry through their lowest-common-ancestor (LCA) structure for
every unordered pair of retained matches. BQ is the fraction of pairs whose
LCA agrees under the GT-to-prediction match map. This design separates two
effects:

- unmatched or hallucinated content is handled by recovery; and
- parent/child organization among content found by both trees is handled by BQ.

An identical matched skeleton gives BQ = 1. Parent rewiring, misplaced
intermediate nodes, or changed ancestor relationships lower BQ. Nodes without a
mask match do not receive a second penalty inside BQ.

With fewer than two retained matches, the paper profile defines BQ as 1 because
there is no matched-node pair whose branch relation can be wrong. The recovery
term still supplies zero credit when there are no matches. The executable LCA
comparison is authoritative for contracted intermediate levels.

## Tree quality: TQ

TQ combines hierarchy agreement with a PQ-style recovery term:

```text
recovery_PQ = TP / (TP + 0.5 * FP + 0.5 * FN)
TQ = BQ * recovery_PQ
```

The quantities in this equation are per-image quantities. The dataset-level
`mean_tq` is their macro average, so substituting pooled TP/FP/FN into the
equation will not in general reproduce the reported mean.

TQ falls when a method misses reference content, predicts unmatched content, or
changes the matched hierarchy. It does not include label or matched-mask
fidelity; those enter meanNQ.

## Open Tree Quality: OTQ

The primary open-tree metric is:

```text
OTQ = TQ * meanNQ
```

At the image level, OTQ simultaneously requires:

- recovery of reference instances;
- agreement of the matched hierarchy;
- accurate matched masks; and
- semantically compatible open-vocabulary labels.

The dataset result is the macro mean of image-level OTQ, not necessarily
`mean_tq * mean_mean_nq` after aggregation.

## Hierarchical Panoptic Quality: HPQ

HPQ is the complementary recursive semantic-tree metric used in the paper. It
evaluates the hierarchy recursively under its parent context and uses exact
semantic-label matching. Thus, HPQ is intentionally stricter about vocabulary
than OTQ's embedding-based LQ.

At each semantic parent, children are grouped by normalized exact label. Within
each label group, Hungarian assignment selects recursive node-score matches.
Assignments with recursive score below 0.5 are not retained. The per-label
score uses the same `TP + 0.5 FP + 0.5 FN` recovery denominator, weighted by the
retained recursive scores, and the parent score is the unweighted mean over
active labels. At a semantic leaf, the node score is the mean IoU of its
Hungarian-matched mask instances. Recursion begins at the synthetic semantic
root.

The two metrics answer different questions:

- **HPQ:** how well does a prediction reproduce the reference semantic
  hierarchy under exact labels and recursive matching?
- **OTQ:** how well does a prediction recover reference instances while
  separating structural agreement from open-vocabulary mask/label quality?

HPQ should not be substituted for OTQ or described as an OTQ component.

## Paper-profile label policies

The frozen `configs/paper_v1.json` records two label policies inherited from
the paper evaluator:

- OTQ excludes mask nodes whose normalized label is `others`.
- HPQ retains the top-level `others` bucket and promotes its direct semantic
  children to the synthetic root for recursive evaluation.

These policies are part of paper-result reproduction. Changing them creates an
alternate protocol and must be reported as such.

## Output fields

The evaluator reports at least the following concepts:

| Field | Interpretation |
|---|---|
| `num_images` | Number of images in the frozen manifest. |
| `num_gt_masks` | Number of evaluated reference mask instances. |
| `num_pred_masks` | Number of predicted mask instances, including unmatched ones. |
| `num_gt_nodes` | Number of evaluated reference semantic nodes. |
| `num_pred_nodes` | Number of predicted semantic nodes. |
| `mean_hpq` | Macro-averaged recursive exact-label HPQ. |
| `mean_otq` | Macro-averaged image-level OTQ. |
| `mean_tq` | Macro-averaged image-level TQ. |
| `mean_bq` | Macro-averaged matched-skeleton BQ. |
| `mean_mnq` | Macro-averaged meanNQ. |
| `mean_mq` | Macro-averaged mask quality among matches. |
| `mean_lq` | Macro-averaged label quality among matches. |
| `otq_tp` | Pooled retained mask matches. |
| `otq_fp` | Pooled unmatched predicted masks. |
| `otq_fn` | Pooled unmatched reference masks. |

Paper rows `<=1`, `<=2`, and `<=3` are cumulative semantic-depth views; `all`
uses the complete tree. Depth statistics and depth-scoped rows describe the
exact tree view evaluated in that row. Do not combine depth statistics from
one view with metric values from another.

## Edge cases

- **GT present, no prediction:** all reference masks are FN.
- **Prediction present, no GT:** all predicted masks are FP.
- **No retained match:** MQ, LQ, and meanNQ are 0; BQ is 1 under the no-pair
  convention, recovery is 0, and therefore TQ and OTQ are 0.
- **Alternative valid decomposition:** may score lower if it disagrees with
  the canonical reference. This is a benchmark limitation, not evidence that
  the alternative is intrinsically invalid.

## What the score does not claim

COCOTree has no single ontologically inevitable tree for every image. Region
boundaries, grouping, intermediate parents, and stopping depth can admit
multiple reasonable choices. OTQ and HPQ therefore measure agreement with a
published annotation policy and silver reference; they do not prove that one
decomposition is the only valid answer.

For that reason, report component scores, counts, and qualitative alternative
decompositions alongside the headline metric when a method's design
intentionally differs from the reference policy.
