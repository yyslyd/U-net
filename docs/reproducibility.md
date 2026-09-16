# Reproduce the maintenance checks

This release verifies runnable code paths using generated data and mathematical fixtures. It does not reproduce the historical AMtown02 benchmark.

## Reference environment

- Python 3.11.
- PyTorch 2.6.0 and torchvision 0.21.0 (CPU wheels for CI).
- NumPy 2.4.3, Pillow 11.3.0, matplotlib 3.10.8, tqdm 4.67.1, pytest 9.0.2.
- Local verification uses Windows with PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124, executing the smoke example on CPU.
- GitHub Actions runs the same regression suite on Windows and Ubuntu using CPU wheels. See the exact status in the repository's Actions tab.

Installation commands are in the [README](../README.md). The legacy `requirements.txt` and Docker image were inherited from upstream and are not the maintenance reference environment. wandb is optional and disabled in the checks.

## Commands and assertions

```bash
python -m pytest tests -q
python scripts/smoke_test.py --output output/smoke-local.json
python scripts/audit_results.py
```

The regression suite covers paired input validation, non-contiguous mask labels, deterministic train/validation IDs and malformed manifests, hand-calculated metric edge cases, historical artifact consistency, and synthetic model execution. An integration test runs the actual trainer for one epoch, evaluates its checkpoint using the saved validation IDs, and confirms that changing a dataset file invalidates the recorded hashes. The smoke test checks:

1. RGB images become a `[2, 3, 32, 32]` tensor and raw masks become contiguous indices.
2. A real U-Net produces `[2, 15, 32, 32]` logits.
3. Focal + Dice loss is finite; backpropagation and SGD update output-layer weights.
4. Saving and reloading model weights plus `mask_values` preserves CPU evaluation output exactly within a single environment.
5. Predictions decode to valid original mask labels.

The smoke example seeds NumPy and PyTorch and uses one CPU thread. A different library/platform may produce slightly different loss values; no cross-platform bitwise accuracy claim is made. It reports its environment and labels all data synthetic. Temporary checkpoints are removed automatically, and no trained benchmark weights are published.

The [recorded Windows smoke result](verification/smoke-windows.json) contains the actual environment and output of this command:

```bash
python scripts/smoke_test.py --output docs/verification/smoke-windows.json
```

The recorded run produced finite loss `3.36796427`, changed the model weights, and reloaded the checkpoint with identical CPU predictions. This value describes one synthetic optimizer step and is not an accuracy metric or a target loss for other platforms.

## Reproducible new training runs

Use a fresh checkpoint directory. The trainer writes a `.split.json` sidecar beside each checkpoint; keep both files. Schema version 1 records the checkpoint filename and SHA-256, ordered sample IDs, training and validation IDs, seed and validation fraction, raw label values, image scale, upsampling configuration, and paired input filenames and SHA-256 hashes. The evaluator checks the checkpoint and dataset contents before using the recorded validation IDs. Missing legacy records cause a clear failure instead of reconstructing a guessed split from a seed.

Run `python train.py --help` and `python scripts/analyze_results.py --help` for supported options. All CLI examples must be run at the repository root. Use `--no-wandb` for offline training. Adding a seed makes sample allocation and random streams repeatable; it does not guarantee identical CUDA kernels, hardware arithmetic or optimizer continuation across environments.

`--load` initializes weights for a new run; it does not restore optimizer, scheduler or random-generator state. It therefore does not resume an interrupted run exactly. Evaluation outputs `evaluation_report.json` in the selected output directory; use a separate directory for each result you want to retain. Add `--figures-dir` to generate a per-class IoU plot.

Current data loading preloads the dataset into memory. This release does not add streaming or large-dataset performance claims. The smoke example is deliberately small and cannot validate full-resolution memory requirements or the final model's accuracy.

## Publication scope

`v0.1.0-rc.1` is a maintenance pre-release for the public 2D adaptation. The maintained examples, tests and result provenance can be reviewed without the original course data. The original dataset scores remain historical and unverified; see [results](results.md).
