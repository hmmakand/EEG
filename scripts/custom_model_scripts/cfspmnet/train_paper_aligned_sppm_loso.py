"""Run paper-aligned CFSPMNet with full-target transductive SPPM LOSO.

Loads direct Figshare raw MAT trials, keeps 30 EEG channels including CPz, and
applies the documented paper-aligned preprocessing profile while retaining raw
left/right labels and the original channel orientation. It then adapts on all
held-out trials without labels before final evaluation. Because both data and
training recipe differ, this is paper alignment rather than a pure ablation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.cfspmnet.cli import run_cli  # noqa: E402
from eeg_bci.cfspmnet.presets import paper_aligned_sppm_config  # noqa: E402


def main() -> int:
    return run_cli(paper_aligned_sppm_config())


if __name__ == "__main__":
    raise SystemExit(main())
