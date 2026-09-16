# Results: sources, definitions and reproducibility limits

## Two historical artifacts

Base inspected: [`a390cac03cdec0c913d79a5d111554277139a345`](https://github.com/yyslyd/U-net/commit/a390cac03cdec0c913d79a5d111554277139a345).

| Metric (%) | Training report | Later leaderboard file | Leaderboard minus report (percentage points) |
| --- | ---: | ---: | ---: |
| mIoU | 85.23 | 85.19 | -0.04 |
| Mean Dice | 91.55 | 91.54 | -0.01 |
| FWIoU | 93.27 | 93.26 | -0.01 |
| Pixel accuracy | 96.45 | Not recorded | Not comparable |

Sources: [`output/training_report.json`](../output/training_report.json) and [`leaderboard/submission.json`](../leaderboard/submission.json).

The April 10, 2026 commit changed only the three leaderboard scores. Its message is `Update metrics in submission.json`; it did not include a new checkpoint, confusion matrix, sample list or evaluation log. There is insufficient evidence to attribute the difference to rounding, a new split or a different model. Both score sets are preserved. The maintenance change only replaces the submission's placeholder repository URL.

[`output/result_provenance.json`](../output/result_provenance.json) records this distinction in machine-readable form. Run:

```bash
python scripts/audit_results.py
```

Success means the files match their **recorded provenance**, not that either benchmark result has been reproduced. An unexplained metric edit, non-finite score or incorrect repository URL fails the audit. New runs should write separate evaluation reports, not mutate these historical artifacts.

## What the historical log establishes

`training_full.log` records a run with 1,380 samples, 1,173 training samples and 207 validation samples; 100 epochs; batch size 8; 15 classes; learning rate 5e-5; image scale 1.0; CUDA and AMP. It records checkpoints through epoch 100 and 506 training-time validation Dice entries. The final logged Dice is approximately 0.92698.

These statements describe the checked-in log. The original images, masks and checkpoint are absent, so neither the training run nor its aggregate scores were rerun for this release. The code that originally prepared/resized the course data is also unavailable.

## Why the logged Dice and report Dice are not interchangeable

The historical training monitor in `evaluate.py`:

- excludes class index 0;
- averages Dice per class/sample, then averages validation batches;
- treats an empty prediction/target pair according to the smoothing rule in `dice_coeff`;
- was supplied a loader with `drop_last=True`, so a 207-sample split with batch size 8 evaluated 200 samples each round.

The historical analysis script instead accumulates a global confusion matrix, includes label 0, and averages per-class scores over classes with ground-truth support. Its validation IDs were sorted before the seeded split; the training dataset did not sort its IDs. A seed of 0 therefore did not establish that both evaluated the same held-out samples. The old JSON key `test_metrics` does not prove use of an independent test set: the analysis code constructed a validation subset.

The current maintenance code retains the training monitor as a monitoring signal, includes the final validation batch, and records the exact split for future checkpoint evaluation. It does not recalculate or relabel the historical scores as corrected measurements.

## Metric protocol for new evaluation reports

Rows of the confusion matrix are ground-truth class indices and columns are predicted indices. For class c:

- IoU = TP / (TP + FP + FN).
- Dice = 2 TP / (2 TP + FP + FN).
- Macro means include classes with nonzero union (TP + FP + FN). A class absent from both target and prediction is undefined and excluded; a class absent from the target but predicted spuriously contributes zero.
- FWIoU weights class IoU by its ground-truth pixel frequency.
- Pixel accuracy = total correct pixels / total pixels.

Scores are percentages. New reports include label 0, as a class, in global evaluation. Class 0 is still excluded by the separate training-time Dice monitor. Unknown raw labels are errors rather than silently remapped to background. These protocol changes mean a future score cannot be described as an improvement over the historical score without evaluating both models on a matched, documented protocol.

The random image-level split is a reproducibility mechanism, not evidence of independent scene-level generalization. If nearby views or frames are correlated, a future benchmark should define scene/sequence-disjoint splits and record them before training.

## Evidence needed for a full benchmark rerun

Obtain the authorized dataset and preprocessing recipe, a matching checkpoint and architecture, exact train/validation IDs, and the chosen metric/ignored-class protocol. Keep their hashes with the evaluation output. The public model and evaluation operate on image pixels.
