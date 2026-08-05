"""Run canonicalized source-only CFSPMNet LOSO on the project data profile.

Uses the same MOABB preprocessing and training recipe as experiment 1, but
maps labels to affected/unaffected and flips left-paralysis hemispheres. It
performs no target adaptation; compare it with experiment 1 to isolate
canonicalization and with experiment 3 to isolate SPPM adaptation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.cfspmnet.cli import run_cli  # noqa: E402
from eeg_bci.cfspmnet.presets import canonical_source_only_config  # noqa: E402


def main() -> int:
    return run_cli(canonical_source_only_config())


if __name__ == "__main__":
    raise SystemExit(main())
