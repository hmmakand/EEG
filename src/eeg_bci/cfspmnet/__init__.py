"""CFSPMNet: Fourier-Reorganized State Mamba Network + Shared-Private Prototype
Matching for cross-subject (LOSO) motor-imagery decoding on Liu2024.

Ported from the reference implementation in ``temp/cfspmnet_frsmamba_model.py``
and ``temp/sppm_strategy.py`` (see ``temp/CFSPMNET.md`` for the paper). Liu2024
*is* the paper's "XW-Stroke" dataset -- ``data/moabb/MNE-liu2024-data/files/participants.tsv``
matches its clinical fields and channel montage.
"""

from __future__ import annotations
