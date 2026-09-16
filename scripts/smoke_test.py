"""Run a small CPU U-Net training/checkpoint example using synthetic raster data."""
import argparse
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import FocalLoss, dice_loss

MASK_VALUES = [0, 1, 2, 3, 5, 6, 13, 14, 15, 16, 17, 19, 20, 24, 25]


def run_example():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # Use the public preprocessing path on actual raster images and masks.
    images, masks = [], []
    for _ in range(2):
        rgb = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        labels = np.asarray(MASK_VALUES, dtype=np.uint8)[np.arange(1024).reshape(32, 32) % 15]
        images.append(torch.as_tensor(BasicDataset.preprocess(None, rgb, 1.0, False)).float())
        masks.append(torch.as_tensor(BasicDataset.preprocess(MASK_VALUES, Image.fromarray(labels), 1.0, True)).long())
    inputs, targets = torch.stack(images), torch.stack(masks)
    model = UNet(n_channels=3, n_classes=15).cpu().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = model.outc.conv.weight.detach().clone()
    logits = model(inputs)
    loss = FocalLoss(gamma=2)(logits, targets) + dice_loss(
        logits.softmax(dim=1), F.one_hot(targets, 15).permute(0, 3, 1, 2).float(), multiclass=True)
    if not torch.isfinite(loss):
        raise RuntimeError('Synthetic training loss is not finite')
    loss.backward()
    optimizer.step()
    updated = not torch.equal(before, model.outc.conv.weight.detach())
    model.eval()
    with torch.inference_mode():
        expected = model(inputs)
    # A temporary checkpoint demonstrates serialization without distributing weights.
    with tempfile.TemporaryDirectory(prefix='unet-smoke-') as temp:
        checkpoint = Path(temp) / 'checkpoint.pth'
        state = model.state_dict()
        state['mask_values'] = MASK_VALUES
        torch.save(state, checkpoint)
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        values = loaded.pop('mask_values')
        restored = UNet(n_channels=3, n_classes=len(values)).cpu().eval()
        restored.load_state_dict(loaded)
        with torch.inference_mode():
            actual = restored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        decoded = np.asarray(values)[actual.argmax(dim=1).numpy()]
        if not set(np.unique(decoded)).issubset(set(MASK_VALUES)):
            raise RuntimeError('Decoded predictions contain an unknown raw label')
    if not updated:
        raise RuntimeError('Optimizer did not update model weights')
    return {
        'data': 'synthetic RGB images and raster masks',
        'seed': 0, 'device': 'cpu', 'optimizer_steps': 1,
        'output_shape': list(actual.shape),
        'loss': round(loss.item(), 8), 'loss_is_finite': math.isfinite(loss.item()),
        'weights_updated': updated, 'checkpoint_reload_equal': True,
        'benchmark_reproduced': False,
        'scope': 'Preprocessing, 2D model forward/backward, optimizer step, checkpoint reload, label decoding',
        'environment': {'python': platform.python_version(), 'platform': platform.system(),
                        'torch': torch.__version__, 'numpy': np.__version__, 'Pillow': PIL.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON output file')
    args = parser.parse_args()
    report = run_example()
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
