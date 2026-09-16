import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unet import UNet
from utils.data_loading import BasicDataset
from utils.segmentation import AMTOWN02_MASK_VALUES
from utils.splits import (create_split_manifest, dataset_file_records, file_sha256,
                          save_split_manifest)


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(images: Path, masks: Path) -> None:
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(images / "tile.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(masks / "tile.png")


def _checkpoint(path: Path) -> None:
    model = UNet(n_channels=3, n_classes=len(AMTOWN02_MASK_VALUES))
    state = model.state_dict()
    state["mask_values"] = list(AMTOWN02_MASK_VALUES)
    torch.save(state, path)


def test_analysis_requires_checkpoint_associated_manifest(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    _write_fixture(images, masks)
    checkpoint = tmp_path / "checkpoint.pth"
    _checkpoint(checkpoint)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_results.py"),
         "--checkpoint", str(checkpoint), "--images-dir", str(images),
         "--masks-dir", str(masks), "--output-dir", str(tmp_path / "evaluation")],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )

    assert result.returncode != 0
    assert "split manifest" in result.stderr.lower()


def test_analysis_writes_provenanced_report_to_explicit_output(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    _write_fixture(images, masks)
    checkpoint = tmp_path / "checkpoint.pth"
    _checkpoint(checkpoint)
    dataset = BasicDataset(images, masks, mask_values=AMTOWN02_MASK_VALUES)
    manifest = create_split_manifest(
        ["tile"], 1.0, 5, checkpoint.name, file_sha256(checkpoint),
        AMTOWN02_MASK_VALUES, image_scale=1.0, bilinear=False,
        dataset_files=dataset_file_records(dataset),
    )
    manifest_path = checkpoint.with_suffix(".split.json")
    save_split_manifest(manifest_path, manifest)
    output = tmp_path / "evaluation"

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_results.py"),
         "--checkpoint", str(checkpoint), "--images-dir", str(images),
         "--masks-dir", str(masks), "--output-dir", str(output),
         "--device", "cpu"],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["provenance"]["checkpoint_sha256"]
    assert report["provenance"]["split_manifest_sha256"]
    assert report["provenance"]["validation_ids"] == ["tile"]
    assert report["evaluation"]["num_validation_images"] == 1
    assert not (output / "training_report.json").exists()
    assert not (output / "submission.json").exists()
