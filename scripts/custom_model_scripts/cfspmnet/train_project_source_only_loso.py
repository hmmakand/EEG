"""Run the project-profile, raw-label CFSPMNet source-only LOSO baseline.

Uses the processed 29-channel MOABB EDF and original left/right labels. Each
fold trains only on the other 49 subjects; the held-out target is used only
during final evaluation. This is experiment 1 of the comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.cfspmnet.cli import run_cli  # noqa: E402
from eeg_bci.cfspmnet.presets import project_source_only_config  # noqa: E402


def main() -> int:
    return run_cli(project_source_only_config())


if __name__ == "__main__":
    raise SystemExit(main())
