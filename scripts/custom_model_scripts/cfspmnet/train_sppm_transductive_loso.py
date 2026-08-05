"""Run project-profile canonical CFSPMNet with transductive SPPM LOSO.

Uses the same MOABB data, canonicalization, and total training budget as
experiment 2. Stage I learns from the other 49 subjects; Stage II adapts with
all held-out trials unlabeled before their labels are used for final evaluation.
Compare with experiment 2 to isolate the effect of SPPM target adaptation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.cfspmnet.cli import run_cli  # noqa: E402
from eeg_bci.cfspmnet.presets import sppm_transductive_config  # noqa: E402


def main() -> int:
    return run_cli(sppm_transductive_config())


if __name__ == "__main__":
    raise SystemExit(main())
