# Changelog

## v0.1.0-rc.1 — 2026-09-16

Maintenance pre-release of the public AMtown02 **2D image segmentation** adaptation.

- Rewrite the README around the actual input/output contract, upstream attribution, adaptation work and executable commands.
- Trace the report/leaderboard score discrepancy to the April 10 commit; preserve both historical score sets and repair the repository URL.
- Add a result-provenance audit and document validation-split and metric-definition limitations.
- Make dataset ordering and checkpoint-associated validation splits explicit; reject malformed or missing evaluation provenance.
- Add regression coverage for labels, data pairing, splits and confusion-matrix metrics.
- Add a synthetic CPU U-Net optimizer/checkpoint example and Windows/Linux CI.
- Remove the machine-specific path and Unix `tee` requirement from the Windows launcher.

No original AMtown02 dataset or trained checkpoint is included. This release does not claim a new benchmark score, a reproduction of the historical scores, or implementation of a 3D point-cloud model.
