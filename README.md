# AMtown02 semantic segmentation with U-Net

[![Tests](https://github.com/yyslyd/U-net/actions/workflows/tests.yml/badge.svg)](https://github.com/yyslyd/U-net/actions/workflows/tests.yml)

A coursework-origin adaptation of [milesial/PyTorch-UNet](https://github.com/milesial/PyTorch-UNet) for AMtown02 semantic segmentation, maintained in this repository by [yyslyd](https://github.com/yyslyd).

**The public implementation performs 2D image segmentation:** RGB images enter a `Conv2d` U-Net and the output assigns a class to each pixel. It does not include a point-cloud loader, a 3D segmentation network, or projection/back-projection between pixels and 3D points.

中文说明：本仓库公开的是 AMtown02 图像语义分割适配、训练和评估代码。历史分数与本次合成数据验证分开记录，未宣称重新复现原始数据集成绩。

## What this adaptation adds

- Remapping of 15 non-contiguous AMtown02 raw mask labels to contiguous class indices.
- Windows-compatible loading and RAM preloading of images and masks.
- Focal Loss + Dice Loss, AdamW, cosine learning-rate scheduling and training augmentation.
- Per-class IoU/Dice, frequency-weighted IoU, pixel accuracy and plots.
- Maintenance release: deterministic sample ordering, checkpoint-associated split records, stricter data validation, regression tests and a runnable synthetic CPU example.

The network architecture originates from upstream. The earlier adaptation work is visible in [commits](https://github.com/yyslyd/U-net/commits/master/) and [implementation issues #1–#5](https://github.com/yyslyd/U-net/issues?q=is%3Aissue+is%3Aclosed).

## Quick check without the original dataset

Use Python **3.11** in a virtual environment. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/smoke_test.py --output output/smoke-local.json
python scripts/audit_results.py
```

The smoke example creates two synthetic 32×32 RGB images and 15-class masks, runs an actual U-Net optimizer step on CPU, saves and reloads a temporary checkpoint, and checks output equivalence and label decoding. It needs no downloaded data, pretrained weights, API keys or external service. It does **not** measure AMtown02 accuracy. See [reproducibility notes](docs/reproducibility.md).

The CPU wheel command above is the reference test setup. GPU training requires a PyTorch build appropriate for your driver. The original `requirements.txt`, Dockerfile and Carvana download helpers are retained upstream artifacts; the tested maintenance environment is specified by the commands above and `requirements-dev.txt`.

## Dataset layout and label contract

Obtain AMtown02 images and labels from your authorized course/data source. The dataset and original trained checkpoint are not distributed here. Do not use the upstream Carvana download scripts to obtain AMtown02.

```text
data/
  imgs/
    frame_0001.png       # RGB raster image
    frame_0002.png
  masks/
    frame_0001.png       # single-channel raw class IDs, same size as image
    frame_0002.png
```

Each image must have one matching mask of the same width and height. Use equally sized square tiles for batched training: random 90-degree augmentation can swap width and height, so rectangular or mixed-sized inputs are not supported by the default batch collation. The AMtown02 configuration uses these raw label values:

```text
0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24, 25
```

Checkpoint `mask_values` defines the mapping between model class indices and original labels. The pipeline expects already prepared image/mask pairs; the preprocessing that produced the historical course dataset is not included. Data are preloaded into RAM, so memory scales with image resolution and dataset size.

## Train and evaluate a new run

Training below is an example configuration, not a reproduction of the historical checkpoint. Use a new checkpoint directory for each run.

```bash
python train.py --images-dir data/imgs --masks-dir data/masks --checkpoint-dir checkpoints/amtown02-run1 --epochs 100 --batch-size 8 --learning-rate 5e-5 --scale 1.0 --validation 15 --classes 15 --seed 0 --no-wandb
```

Add `--amp` when using a suitable CUDA setup. `run_training.bat` provides the corresponding Windows launch command without a machine-specific directory or a dependency on Unix `tee`.

```bash
python scripts/analyze_results.py --checkpoint checkpoints/amtown02-run1/checkpoint_epoch100.pth --images-dir data/imgs --masks-dir data/masks --output-dir output/evaluations/amtown02-run1
```

New checkpoints have a `.split.json` sidecar recording the exact sample split and evaluation context. Keep the checkpoint and sidecar together. Evaluation refuses to guess the original split for legacy checkpoints without this record. New evaluation reports are separate from historical results.

For prediction with a compatible 15-class checkpoint:

```bash
python predict.py --model checkpoints/amtown02-run1/checkpoint_epoch100.pth --classes 15 --scale 1.0 --input data/imgs/frame_0001.png --output prediction.png
```

## Historical results and their limits

All numbers below are percentages stored in the repository, **not independently reproduced scores**.

| Artifact | mIoU | Mean Dice | FWIoU |
| --- | ---: | ---: | ---: |
| `output/training_report.json` | 85.23 | 91.55 | 93.27 |
| `leaderboard/submission.json`, updated April 10, 2026 | 85.19 | 91.54 | 93.26 |

The later commit changed only the submission scores and provided no new evaluation trace. Both sets are retained. Historical split ordering and metric conventions also differed between training and analysis; the maintenance release makes future runs explicit rather than retroactively certifying those scores. [Full provenance and metric definitions](docs/results.md).

## Project layout

| Path | Purpose |
| --- | --- |
| `unet/` | Upstream-derived 2D network |
| `utils/data_loading.py`, `train.py` | Dataset loading and training |
| `evaluate.py` | Training-time monitoring Dice |
| `scripts/analyze_results.py` | Checkpoint-based evaluation with provenance |
| `scripts/smoke_test.py` | Synthetic CPU training/checkpoint example |
| `scripts/audit_results.py` | Check the two recorded historical score sets |
| `tests/` | Data, split, metrics, artifact and smoke regression tests |
| `docs/`, `CHANGELOG.md` | Scope, reproducibility and release notes |

## Contributing and attribution

Bug reports should include the commit, Python/PyTorch versions, the command and a minimal non-sensitive example. Run `python -m pytest tests -q` before proposing a change. Dataset/model benchmark claims should include a checkpoint, the exact split, preprocessing and metric protocol.

Licensed under [GPL-3.0](LICENSE), retaining the upstream license. Credit for the U-Net architecture and original PyTorch implementation belongs to their original authors; this repository maintains the AMtown02 adaptation and supporting tools.

- [Original PyTorch-UNet implementation](https://github.com/milesial/PyTorch-UNet)
- [U-Net paper: Ronneberger, Fischer and Brox](https://arxiv.org/abs/1505.04597)
