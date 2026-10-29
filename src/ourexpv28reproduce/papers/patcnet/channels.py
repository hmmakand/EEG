"""The sensorimotor ROI of PGTC (paper, section 2.3): left {FC3, C3, CP3}, middle {FCz, Cz, CPz}, right {FC4, C4, CP4}.

The original code reads the channel names and the ROI groups from a JSON file
(`--channel-info-path`) that is not published. This module gives the same
structure, as read_channel_info returns it.
"""
from ...datascripts import bci2a, liu2024

ROI_GROUPS = dict(left=["FC3", "C3", "CP3"], right=["FC4", "C4", "CP4"], midline=["FCz", "Cz", "CPz"])


def channel_info(dataset: str, cohort: str = "published") -> dict:
    """{"channel_names": [...], "roi_groups": {...}} for a dataset whose channel names are known."""
    if dataset == "liu2024" and cohort == "version27":
        names = list(liu2024.EEG_CHANNELS)
        # Version 27 omits the all-zero CPz reference electrode.
        groups = {group: [name for name in members if name in names] for group, members in ROI_GROUPS.items()}
        return dict(channel_names=names, roi_groups=groups)
    if dataset == "liu2024":
        return dict(channel_names=list(liu2024.CHANNELS_WITH_REFERENCE), roi_groups=ROI_GROUPS)
    if dataset == "bci2a":
        return dict(channel_names=list(bci2a.EEG_CHANNELS), roi_groups=ROI_GROUPS)
    raise ValueError(f"The channel names of {dataset} are not known, so the ROI of PGTC cannot be built "
                     "(see papers/patcnet/README.md, open decisions)")
