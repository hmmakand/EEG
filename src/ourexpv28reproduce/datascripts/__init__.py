"""File readers, shared by all papers: every dataset as subjects, trials and left/right labels at 250 Hz.

How a paper prepares the trials (filter, reference, window, subjects) belongs
to that paper's folder in papers/.
"""
from . import bci2a, jia2019, liu2024
from .trial import SAMPLE_RATE, Trial

READERS = dict(liu2024=liu2024, bci2a=bci2a, jia2019=jia2019)

__all__ = ["READERS", "SAMPLE_RATE", "Trial", "bci2a", "jia2019", "liu2024"]
