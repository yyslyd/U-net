"""Evaluate a checkpoint against its saved validation split without changing historical artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unet import UNet
from utils.data_loading import BasicDataset
from utils.segmentation import metrics_from_confusion
from utils.splits import dataset_file_records, file_sha256, load_split_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a U-Net checkpoint using its checkpoint-associated split manifest"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .pth file")
    parser.add_argument("--images-dir", type=Path, required=True, help="Evaluation image directory")
    parser.add_argument("--masks-dir", type=Path, required=True, help="Evaluation mask directory")
    parser.add_argument(
        "--split-manifest", type=Path,
        help="Split sidecar (default: CHECKPOINT with .split.json suffix)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New evaluation directory (default: output/evaluations/CHECKPOINT_STEM)",
    )
    parser.add_argument(
        "--scale", type=float,
        help="Require this scale to match the manifest; preprocessing always follows the manifest",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device (default: auto)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--figures-dir", type=Path, help="Optional directory for evaluation plots")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise ValueError("MPS was requested but is unavailable")
    return device


def load_checkpoint_model(checkpoint_path: Path, manifest: dict, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict) or "mask_values" not in state_dict:
        raise ValueError("Checkpoint has no mask_values metadata")
    mask_values = state_dict.pop("mask_values")
    if list(mask_values) != manifest["mask_values"]:
        raise ValueError("Checkpoint mask_values do not match the split manifest")
    model = UNet(n_channels=3, n_classes=len(mask_values), bilinear=manifest["bilinear"])
    model.load_state_dict(state_dict)
    model.to(device=device).eval()
    return model, list(mask_values)


@torch.inference_mode()
def compute_metrics(model, dataloader, n_classes: int, mask_values, device: torch.device) -> dict:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch", disable=not sys.stderr.isatty()):
        images = batch["image"].to(device=device, dtype=torch.float32)
        ground_truth = batch["mask"].numpy()
        prediction = model(images).argmax(dim=1).cpu().numpy()
        valid = ((ground_truth >= 0) & (ground_truth < n_classes)
                 & (prediction >= 0) & (prediction < n_classes))
        np.add.at(confusion, (ground_truth[valid], prediction[valid]), 1)
    metrics = metrics_from_confusion(confusion, mask_values)
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def plot_figures(metrics: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = list(metrics["per_class"])
    ious = [metrics["per_class"][name]["iou"] for name in classes]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(classes)), ious, color="#2E86AB")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IoU (%)")
    ax.set_title(f"Evaluation per-class IoU (mIoU: {metrics['miou']:.2f}%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_iou.png", dpi=150)
    plt.close(fig)


def evaluate_checkpoint(args) -> Path:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    manifest_path = (args.split_manifest or checkpoint.with_suffix(".split.json")).resolve()
    checkpoint_hash = file_sha256(checkpoint)
    manifest = load_split_manifest(
        manifest_path,
        checkpoint_filename=checkpoint.name,
        checkpoint_sha256=checkpoint_hash,
    )
    if args.scale is not None and args.scale != manifest["image_scale"]:
        raise ValueError(
            f"Requested scale {args.scale} conflicts with split manifest image_scale "
            f"{manifest['image_scale']}"
        )
    if args.batch_size < 1:
        raise ValueError("batch size must be at least 1")

    dataset = BasicDataset(
        args.images_dir.resolve(), args.masks_dir.resolve(),
        scale=manifest["image_scale"], mask_values=manifest["mask_values"],
    )
    current_files = dataset_file_records(dataset)
    manifest = load_split_manifest(
        manifest_path,
        dataset_ids=dataset.ids,
        checkpoint_filename=checkpoint.name,
        checkpoint_sha256=checkpoint_hash,
        dataset_files=current_files,
    )
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.ids)}
    validation_indices = [index_by_id[sample_id] for sample_id in manifest["val_ids"]]
    if not validation_indices:
        raise ValueError("Split manifest validation set is empty")
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )

    device = resolve_device(args.device)
    model, mask_values = load_checkpoint_model(checkpoint, manifest, device)
    metrics = compute_metrics(model, validation_loader, len(mask_values), mask_values, device)

    output_dir = (args.output_dir or
                  PROJECT_ROOT / "output" / "evaluations" / checkpoint.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    report = {
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "split_manifest": str(manifest_path),
            "split_manifest_sha256": file_sha256(manifest_path),
            "images_dir": str(args.images_dir.resolve()),
            "masks_dir": str(args.masks_dir.resolve()),
            "dataset_files": current_files,
            "validation_ids": manifest["val_ids"],
            "seed": manifest["seed"],
            "validation_fraction": manifest["validation_fraction"],
            "image_scale": manifest["image_scale"],
            "bilinear": manifest["bilinear"],
            "device": str(device),
        },
        "evaluation": {
            "num_validation_images": len(validation_indices),
            "macro_average_rule": "classes with non-zero ground-truth/prediction union",
            "metrics_percent": metrics,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.figures_dir:
        plot_figures(metrics, args.figures_dir.resolve())
    return report_path


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report_path = evaluate_checkpoint(args)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Evaluation report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
