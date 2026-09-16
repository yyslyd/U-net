import json
import subprocess
import sys
from pathlib import Path


def test_cpu_example_trains_and_reloads_real_unet(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'smoke.json'
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/smoke_test.py'), '--output', str(output)],
        capture_output=True, text=True, encoding='utf-8', timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report['data'] == 'synthetic RGB images and raster masks'
    assert report['output_shape'] == [2, 15, 32, 32]
    assert report['weights_updated'] is True
    assert report['checkpoint_reload_equal'] is True
    assert report['benchmark_reproduced'] is False
    assert report['loss_is_finite'] is True
