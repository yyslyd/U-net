"""Check recorded result provenance without claiming a benchmark rerun (stdlib only)."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('dice_score', 'miou', 'fwiou')


def validate_metrics(values):
    for key in METRICS:
        value = values[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError(f'{key} must be a finite percentage between 0 and 100')


def audit(report, submission, provenance):
    recorded = report['test_metrics']
    submitted = submission['metrics']
    validate_metrics(recorded)
    validate_metrics(submitted)
    for key in METRICS:
        if (recorded[key] != provenance['training_report_metrics'][key]
                or submitted[key] != provenance['leaderboard_metrics'][key]):
            raise ValueError(f'{key} differs from recorded provenance; investigate its source')
    if submission['project_private_repo_url'] != provenance['repository_url']:
        raise ValueError('Submission repository URL differs from recorded provenance')
    return {
        'verification_scope': 'artifact consistency only; no benchmark rerun',
        'benchmark_reproduced': False,
        'differences_percentage_points': {
            key: round(submitted[key] - recorded[key], 2) for key in METRICS
        },
        'explanation': provenance['explanation'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'output/training_report.json')
    parser.add_argument('--submission', type=Path, default=ROOT / 'leaderboard/submission.json')
    parser.add_argument('--provenance', type=Path, default=ROOT / 'output/result_provenance.json')
    args = parser.parse_args()
    try:
        inputs = [json.loads(path.read_text(encoding='utf-8'))
                  for path in (args.report, args.submission, args.provenance)]
        print(json.dumps(audit(*inputs), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f'Result audit failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
