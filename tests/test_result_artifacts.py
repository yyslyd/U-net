import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_audit(report, submission, provenance):
    return subprocess.run(
        [sys.executable, str(ROOT / 'scripts/audit_results.py'),
         '--report', str(report), '--submission', str(submission),
         '--provenance', str(provenance)],
        capture_output=True, text=True, encoding='utf-8',
    )


def test_recorded_discrepancy_is_explained_without_changing_scores():
    result = run_audit(ROOT / 'output/training_report.json',
                       ROOT / 'leaderboard/submission.json',
                       ROOT / 'output/result_provenance.json')
    assert result.returncode == 0, result.stderr
    audit = json.loads(result.stdout)
    assert audit['verification_scope'] == 'artifact consistency only; no benchmark rerun'
    assert audit['differences_percentage_points'] == {
        'dice_score': -0.01, 'miou': -0.04, 'fwiou': -0.01,
    }
    assert audit['benchmark_reproduced'] is False


def test_unexplained_metric_change_is_rejected(tmp_path):
    submission = json.loads((ROOT / 'leaderboard/submission.json').read_text())
    submission['metrics']['miou'] = 99.99
    path = tmp_path / 'submission.json'
    path.write_text(json.dumps(submission))
    result = run_audit(ROOT / 'output/training_report.json', path,
                       ROOT / 'output/result_provenance.json')
    assert result.returncode != 0
    assert 'recorded provenance' in result.stderr


def test_nonfinite_metric_is_rejected(tmp_path):
    submission = json.loads((ROOT / 'leaderboard/submission.json').read_text())
    submission['metrics']['miou'] = float('nan')
    path = tmp_path / 'submission.json'
    path.write_text(json.dumps(submission))
    result = run_audit(ROOT / 'output/training_report.json', path,
                       ROOT / 'output/result_provenance.json')
    assert result.returncode != 0
    assert 'finite percentage' in result.stderr
