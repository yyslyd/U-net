"""Exercise the actual trainer -> checkpoint sidecar -> evaluator on tiny data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from train import train_model
from unet import UNet
from utils.segmentation import AMTOWN02_MASK_VALUES
from utils.splits import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_training_checkpoint_can_be_evaluated_with_its_exact_split(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    images, masks, checkpoints = (tmp_path / name for name in ('images', 'masks', 'checkpoints'))
    images.mkdir()
    masks.mkdir()
    rng = np.random.default_rng(7)
    for name in ('z_last', 'a_first', 'm_middle'):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(images / f'{name}.png')
        raw = np.array(AMTOWN02_MASK_VALUES, dtype=np.uint8)[np.arange(1024).reshape(32, 32) % 15]
        Image.fromarray(raw).save(masks / f'{name}.png')

    train_model(UNet(3, 15), torch.device('cpu'), epochs=1, batch_size=2,
                val_percent=0.34, img_scale=1.0, amp=False, images_dir=images,
                masks_dir=masks, checkpoint_dir=checkpoints, seed=7, use_wandb=False)

    checkpoint = checkpoints / 'checkpoint_epoch1.pth'
    sidecar = checkpoint.with_suffix('.split.json')
    manifest = json.loads(sidecar.read_text(encoding='utf-8'))
    assert manifest['checkpoint_sha256'] == file_sha256(checkpoint)
    assert manifest['dataset_ids'] == ['a_first', 'm_middle', 'z_last']
    assert len(manifest['val_ids']) == 1
    assert len(manifest['train_ids']) == 2
    assert len(manifest['dataset_files']) == 3

    output = tmp_path / 'evaluation'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/analyze_results.py'),
         '--checkpoint', str(checkpoint), '--images-dir', str(images),
         '--masks-dir', str(masks), '--output-dir', str(output), '--device', 'cpu'],
        capture_output=True, text=True, encoding='utf-8', timeout=120,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'evaluation_report.json').read_text(encoding='utf-8'))
    assert report['provenance']['validation_ids'] == manifest['val_ids']
    assert report['provenance']['checkpoint_sha256'] == file_sha256(checkpoint)
    assert report['evaluation']['num_validation_images'] == 1

    # Same filename/IDs but changed contents must not silently reuse the split.
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(images / 'a_first.png')
    rejected = subprocess.run(
        result.args, capture_output=True, text=True, encoding='utf-8', timeout=120,
        env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    )
    assert rejected.returncode != 0
    assert 'hash' in rejected.stderr.lower()
