"""
AAE5303 - UNet 训练结果分析与 Leaderboard 提交生成

用法: python scripts/analyze_results.py
"""

import sys, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from unet import UNet

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "checkpoint_epoch100.pth"
IMG_DIR = PROJECT_ROOT / "data" / "imgs"
MASK_DIR = PROJECT_ROOT / "data" / "masks"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# AMtown02 类名 (按 mask_values 顺序，运行时从 checkpoint 读取)
AMTOWN02_NAMES = {
    0: "unlabeled", 1: "rooftop", 2: "guard_rail", 3: "road",
    5: "dynamic", 6: "ground", 13: "vegetation", 14: "terrain",
    15: "traffic_sign", 16: "pole", 17: "caravan", 19: "building_facade",
    20: "car", 24: "truck", 25: "rider"
}


def load_model(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    mask_values = state_dict.pop("mask_values", list(range(15)))
    n_classes = len(mask_values)
    model = UNet(n_channels=3, n_classes=n_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model, mask_values


def get_val_files(img_dir, mask_dir, val_ratio=0.15, seed=0):
    """复现训练时的 val split"""
    all_ids = sorted([f.stem for f in img_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
    n = len(all_ids)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=gen).tolist()
    n_val = int(n * val_ratio)
    val_indices = indices[n - n_val:]  # random_split 取最后 n_val 个
    return [all_ids[i] for i in sorted(val_indices)]


def compute_metrics(model, mask_values, val_ids, img_dir, mask_dir, scale=1.0):
    """计算 per-class IoU, Dice, pixel accuracy"""
    n_classes = len(mask_values)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    # Build mask LUT
    max_val = max(int(v) for v in mask_values)
    lut = np.full(max(256, max_val + 1), 0, dtype=np.int64)
    for i, v in enumerate(mask_values):
        lut[int(v)] = i

    for name in tqdm(val_ids, desc="Evaluating"):
        # Load image
        img_path = list(img_dir.glob(name + ".*"))[0]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        newW, newH = int(scale * w), int(scale * h)
        img_resized = img.resize((newW, newH), Image.BICUBIC)
        img_np = np.asarray(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(DEVICE)

        # Load mask
        mask_path = list(mask_dir.glob(name + ".*"))[0]
        mask_pil = Image.open(mask_path)
        mask_resized = mask_pil.resize((newW, newH), Image.NEAREST)
        mask_np = np.asarray(mask_resized)
        gt = lut[mask_np]

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        # Accumulate confusion matrix
        valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
        np.add.at(confusion, (gt[valid], pred[valid]), 1)

    # Compute metrics
    per_class = {}
    ious, dices, freqs = [], [], []
    total_correct = np.trace(confusion)
    total_pixels = confusion.sum()

    for c in range(n_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        freq = confusion[c, :].sum() / max(total_pixels, 1)

        iou = tp / max(tp + fp + fn, 1) * 100
        dice = 2 * tp / max(2 * tp + fp + fn, 1) * 100

        mv = int(mask_values[c])
        class_name = AMTOWN02_NAMES.get(mv, f"class_{mv}")
        per_class[class_name] = {
            "mask_value": mv,
            "iou": round(iou, 2),
            "dice": round(dice, 2),
            "frequency": round(freq * 100, 2),
        }
        ious.append(iou)
        dices.append(dice)
        freqs.append(freq)

    # Exclude classes with zero ground-truth pixels (undefined IoU = 0/0)
    present_ious = [v for v, f in zip(ious, freqs) if f > 0]
    present_dices = [v for v, f in zip(dices, freqs) if f > 0]
    n_absent = n_classes - len(present_ious)
    if n_absent > 0:
        print(f"  Note: {n_absent} classes have 0 ground-truth pixels, excluded from mIoU/Dice mean")

    miou = np.mean(present_ious) if present_ious else 0.0
    dice_mean = np.mean(present_dices) if present_dices else 0.0
    fwiou = sum(f * i for f, i in zip(freqs, ious)) / max(sum(freqs), 1e-8)
    pixel_acc = total_correct / max(total_pixels, 1) * 100

    return {
        "miou": round(miou, 2),
        "dice_score": round(dice_mean, 2),
        "fwiou": round(fwiou, 2),
        "pixel_accuracy": round(pixel_acc, 2),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def parse_training_log(log_path):
    """从训练日志提取 loss 和 dice"""
    losses, dices = [], []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "Validation Dice score:" in line:
                val = float(line.strip().split(":")[-1].strip())
                dices.append(val)
    return dices


def plot_figures(dices, metrics, output_dir):
    """生成图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dice curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(dices) + 1), dices, color="#2E86AB", linewidth=1.5)
    ax.set_xlabel("Validation Step")
    ax.set_ylabel("Dice Score")
    ax.set_title("Validation Dice Score During Training")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_dice_curve.png", dpi=150)
    plt.close()

    # 2. Per-class IoU bar chart
    classes = list(metrics["per_class"].keys())
    ious = [metrics["per_class"][c]["iou"] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#DC3545" if v < 30 else "#F18F01" if v < 60 else "#28A745" for v in ious]
    bars = ax.bar(range(len(classes)), ious, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IoU (%)")
    ax.set_title(f"Per-Class IoU (mIoU: {metrics['miou']:.2f}%)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_iou.png", dpi=150)
    plt.close()

    # 3. Class distribution
    freqs = [metrics["per_class"][c]["frequency"] for c in classes]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(classes)), freqs, color="#2E86AB")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pixel Frequency (%)")
    ax.set_title("Class Distribution in Validation Set")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    # 4. Summary dashboard
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Metric gauges
    metric_names = ["mIoU", "Dice", "Pixel Acc"]
    metric_vals = [metrics["miou"], metrics["dice_score"], metrics["pixel_accuracy"]]
    bar_colors = ["#2E86AB", "#28A745", "#F18F01"]
    for ax, name, val, col in zip(axes, metric_names, metric_vals, bar_colors):
        ax.barh([0], [val], color=col, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_title(f"{name}: {val:.2f}%", fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Training Summary Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "summary_dashboard.png", dpi=150)
    plt.close()

    print(f"Figures saved to {output_dir}")


def main():
    print("=" * 60)
    print("UNet Training Analysis - AMtown02 15-class")
    print("=" * 60)

    # Load model
    print("Loading model...")
    model, mask_values = load_model(CHECKPOINT)
    print(f"  mask_values ({len(mask_values)}): {mask_values}")

    # Get val files
    val_ids = get_val_files(IMG_DIR, MASK_DIR, val_ratio=0.15, seed=0)
    print(f"  Validation set: {len(val_ids)} images")

    # Compute metrics
    print("Computing metrics on validation set...")
    metrics = compute_metrics(model, mask_values, val_ids, IMG_DIR, MASK_DIR, scale=1.0)

    # Remove confusion matrix for JSON serialization
    confusion = metrics.pop("confusion_matrix")

    print(f"\n=== Results ===")
    print(f"  mIoU:           {metrics['miou']:.2f}%")
    print(f"  Dice Score:     {metrics['dice_score']:.2f}%")
    print(f"  FWIoU:          {metrics['fwiou']:.2f}%")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.2f}%")
    print(f"\n  Per-Class IoU:")
    for name, info in metrics["per_class"].items():
        print(f"    {name:20s}: IoU={info['iou']:6.2f}%  Dice={info['dice']:6.2f}%  Freq={info['frequency']:5.2f}%")

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "training_summary": {
            "total_epochs": 100,
            "total_images": 1380,
            "num_classes": len(mask_values),
            "image_scale": 1.0,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "optimizer": "AdamW",
            "loss": "FocalLoss(gamma=2) + DiceLoss",
            "scheduler": "CosineAnnealingLR",
            "augmentation": "flip + rot90 + brightness + contrast",
        },
        "test_metrics": {
            "dice_score": metrics["dice_score"],
            "miou": metrics["miou"],
            "fwiou": metrics["fwiou"],
            "pixel_accuracy": metrics["pixel_accuracy"],
        },
        "per_class_results": metrics["per_class"],
    }

    with open(OUTPUT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUTPUT_DIR / 'training_report.json'}")

    # Leaderboard submission
    submission = {
        "group_name": "YD_Team",
        "project_private_repo_url": "",
        "metrics": {
            "dice_score": metrics["dice_score"],
            "miou": metrics["miou"],
            "fwiou": metrics["fwiou"],
        },
    }
    leaderboard_dir = PROJECT_ROOT / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    with open(leaderboard_dir / "submission.json", "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Submission saved to {leaderboard_dir / 'submission.json'}")

    # Parse training log & plot
    log_path = PROJECT_ROOT / "training_full.log"
    dices = parse_training_log(log_path) if log_path.exists() else []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if dices:
        plot_figures(dices, metrics, FIGURES_DIR)
    else:
        print("No training log found, skipping plots")

    print("\nDone!")


if __name__ == "__main__":
    main()
