"""Shared-Private Prototype Matching (SPPM), CFSPMNet paper section 2.3.

Ported from ``temp/sppm_strategy.py``. The private-signature channel groups
are hardcoded to Liu2024's montage (``canonicalization.LIU2024_EEG_CHANNELS``)
instead of the reference file's per-dataset ``channel_name_reader`` dispatch,
since this project only targets Liu2024. Note the reference "midline" group
was ``{FCz, Cz, CPz}``; Liu2024's ``CPz`` is consumed as MOABB's reference
electrode and isn't present in the windowed data, so midline here is
``{FCz, Cz}``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from eeg_bci.cfspmnet.canonicalization import LIU2024_EEG_CHANNELS

PRIVATE_SIGNATURE_CHANNEL_GROUPS: dict[str, list[str]] = {
    "left": ["FC3", "C3", "CP3"],
    "right": ["FC4", "C4", "CP4"],
    "midline": ["FCz", "Cz"],
}


def _resolve_indices(channel_names: tuple[str, ...], signature_channel_names: list[str]) -> list[int]:
    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}
    missing = [name for name in signature_channel_names if name not in channel_to_index]
    if missing:
        raise ValueError(f"Missing private signature channels: {missing}")
    return [channel_to_index[name] for name in signature_channel_names]


def get_private_signature_channel_indices(
    channel_names: tuple[str, ...] = LIU2024_EEG_CHANNELS,
) -> dict[str, list[int]]:
    groups = {
        group: list(names)
        for group, names in PRIVATE_SIGNATURE_CHANNEL_GROUPS.items()
    }
    if "CPz" in channel_names:
        groups["midline"].append("CPz")
    return {
        group: _resolve_indices(channel_names, names)
        for group, names in groups.items()
    }


def l2_normalize(vectors: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def compute_private_signature_features(
    x: np.ndarray,
    channel_names: tuple[str, ...] = LIU2024_EEG_CHANNELS,
) -> np.ndarray:
    """Compact sensorimotor channel summary (Eq. 9): left/right/asymmetry/midline."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected input shape (N, C, T), got {x.shape}")

    signature_indices = get_private_signature_channel_indices(channel_names)
    left_motor = x[:, signature_indices["left"], :].mean(axis=1)
    right_motor = x[:, signature_indices["right"], :].mean(axis=1)
    midline = x[:, signature_indices["midline"], :].mean(axis=1)
    asymmetry = np.abs(left_motor - right_motor)

    signature_vector = np.concatenate([left_motor, right_motor, asymmetry, midline], axis=1)
    return l2_normalize(signature_vector, axis=1).astype(np.float32)


def build_shared_private_signature_prototypes(
    signature_vectors: np.ndarray, labels: np.ndarray, num_classes: int = 2, floor: float = 0.70
) -> tuple[np.ndarray, np.ndarray]:
    """Shared class prototypes c_k and matching tolerances delta_k (Eq. 8)."""

    signature_vectors = np.asarray(signature_vectors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    shared_prototypes = []
    class_wise_matching_tolerance = []
    for class_id in range(num_classes):
        class_vectors = signature_vectors[labels == class_id]
        if len(class_vectors) == 0:
            raise ValueError(f"No source samples found for class {class_id}.")

        shared_prototype = l2_normalize(class_vectors.mean(axis=0, keepdims=True), axis=1)[0]
        similarities = class_vectors @ shared_prototype
        matching_tolerance = max(float(similarities.mean() - similarities.std()), float(floor))

        shared_prototypes.append(shared_prototype.astype(np.float32))
        class_wise_matching_tolerance.append(matching_tolerance)

    return (
        np.stack(shared_prototypes, axis=0),
        np.asarray(class_wise_matching_tolerance, dtype=np.float32),
    )


def _summarize_matching(
    probabilities: torch.Tensor,
    accepted_mask: torch.Tensor,
    *,
    confidence_pass_mask: torch.Tensor,
    signature_pass_mask: torch.Tensor | None,
) -> dict:
    confidences, _ = torch.max(probabilities, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    rejected_mask = ~accepted_mask

    def masked_mean(values, mask):
        if int(mask.sum().item()) == 0:
            return 0.0
        return float(values[mask].mean().item())

    sample_count = int(probabilities.shape[0])
    signature_pass_count = (
        None
        if signature_pass_mask is None
        else int(signature_pass_mask.sum().item())
    )
    return {
        "sample_count": sample_count,
        "confidence_pass_count": int(confidence_pass_mask.sum().item()),
        "signature_pass_count": signature_pass_count,
        "accepted_count": int(accepted_mask.sum().item()),
        "rejected_count": int(rejected_mask.sum().item()),
        "accepted_confidence": masked_mean(confidences, accepted_mask),
        "rejected_confidence": masked_mean(confidences, rejected_mask),
        "accepted_entropy": masked_mean(entropy, accepted_mask),
        "rejected_entropy": masked_mean(entropy, rejected_mask),
    }


def apply_confidence_pseudo_label_matching(
    probabilities: torch.Tensor, prob_threshold: float = 0.60, existing_pseudo: torch.Tensor | None = None
) -> dict:
    """Confidence-only pseudo-label gating (Eq. 10 without the private-signature term)."""

    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities with shape (B, C), got {probabilities.shape}")

    confidences, predictions = torch.max(probabilities, dim=1)
    accepted_mask = confidences >= prob_threshold

    one_hot = torch.zeros_like(probabilities)
    one_hot[accepted_mask, predictions[accepted_mask]] = 1.0

    if existing_pseudo is not None and existing_pseudo.shape != probabilities.shape:
        raise ValueError(
            "existing_pseudo must have the same shape as probabilities; "
            f"got {existing_pseudo.shape} and {probabilities.shape}."
        )

    # Recompute Equation 10's accepted set; rejected samples return to zero.
    updated_pseudo = one_hot

    summary = _summarize_matching(
        probabilities,
        accepted_mask,
        confidence_pass_mask=accepted_mask,
        signature_pass_mask=None,
    )
    summary.update(
        {
            "updated_pseudo": updated_pseudo,
            "accepted_mask": accepted_mask,
            "predictions": predictions,
            "confidences": confidences,
            "similarities": None,
        }
    )
    return summary


def apply_sppm_signature_prototype_matching(
    probabilities: torch.Tensor,
    signature_vectors: torch.Tensor,
    shared_prototypes: torch.Tensor,
    class_wise_matching_tolerance: torch.Tensor,
    prob_threshold: float = 0.60,
    existing_pseudo: torch.Tensor | None = None,
) -> dict:
    """Calibrated pseudo-label gating: semantic confidence AND physiological consistency (Eq. 10)."""

    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities with shape (B, C), got {probabilities.shape}")
    if signature_vectors.ndim != 2:
        raise ValueError(f"Expected signature_vectors with shape (B, D), got {signature_vectors.shape}")

    signature_vectors = F.normalize(signature_vectors, p=2, dim=1)
    shared_prototypes = F.normalize(shared_prototypes, p=2, dim=1)

    confidences, predictions = torch.max(probabilities, dim=1)
    predicted_shared_prototypes = shared_prototypes[predictions]
    predicted_matching_tolerance = class_wise_matching_tolerance[predictions]
    similarities = torch.sum(signature_vectors * predicted_shared_prototypes, dim=1)

    confidence_pass_mask = confidences >= prob_threshold
    signature_pass_mask = similarities >= predicted_matching_tolerance
    accepted_mask = confidence_pass_mask & signature_pass_mask

    one_hot = torch.zeros_like(probabilities)
    one_hot[accepted_mask, predictions[accepted_mask]] = 1.0

    if existing_pseudo is not None and existing_pseudo.shape != probabilities.shape:
        raise ValueError(
            "existing_pseudo must have the same shape as probabilities; "
            f"got {existing_pseudo.shape} and {probabilities.shape}."
        )

    # Dynamic refresh replaces, rather than accumulates, the accepted set.
    updated_pseudo = one_hot

    summary = _summarize_matching(
        probabilities,
        accepted_mask,
        confidence_pass_mask=confidence_pass_mask,
        signature_pass_mask=signature_pass_mask,
    )
    summary.update(
        {
            "updated_pseudo": updated_pseudo,
            "accepted_mask": accepted_mask,
            "predictions": predictions,
            "confidences": confidences,
            "similarities": similarities,
        }
    )
    return summary


def init_matching_aggregator() -> dict:
    return {
        "sample_count": 0,
        "confidence_pass_count": 0,
        "signature_pass_count": 0,
        "signature_evaluated_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "accepted_conf_sum": 0.0,
        "rejected_conf_sum": 0.0,
        "accepted_entropy_sum": 0.0,
        "rejected_entropy_sum": 0.0,
    }


def accumulate_matching_stats(agg: dict, matching_result: dict) -> None:
    sample_count = int(matching_result["sample_count"])
    accepted_count = int(matching_result["accepted_count"])
    rejected_count = int(matching_result["rejected_count"])
    agg["sample_count"] += sample_count
    agg["confidence_pass_count"] += int(
        matching_result["confidence_pass_count"]
    )
    signature_pass_count = matching_result["signature_pass_count"]
    if signature_pass_count is not None:
        agg["signature_pass_count"] += int(signature_pass_count)
        agg["signature_evaluated_count"] += sample_count
    agg["accepted_count"] += accepted_count
    agg["rejected_count"] += rejected_count
    agg["accepted_conf_sum"] += accepted_count * float(matching_result["accepted_confidence"])
    agg["rejected_conf_sum"] += rejected_count * float(matching_result["rejected_confidence"])
    agg["accepted_entropy_sum"] += accepted_count * float(matching_result["accepted_entropy"])
    agg["rejected_entropy_sum"] += rejected_count * float(matching_result["rejected_entropy"])


def finalize_matching_stats(agg: dict) -> dict:
    sample_count = agg["sample_count"]
    accepted_count = agg["accepted_count"]
    rejected_count = agg["rejected_count"]
    signature_evaluated_count = agg["signature_evaluated_count"]
    return {
        "sample_count": sample_count,
        "confidence_pass_count": agg["confidence_pass_count"],
        "signature_pass_count": (
            agg["signature_pass_count"]
            if signature_evaluated_count > 0
            else None
        ),
        "confidence_pass_ratio": (
            agg["confidence_pass_count"] / max(sample_count, 1)
        ),
        "signature_pass_ratio": (
            agg["signature_pass_count"] / signature_evaluated_count
            if signature_evaluated_count > 0
            else None
        ),
        "accepted_ratio": accepted_count / max(sample_count, 1),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_confidence": agg["accepted_conf_sum"] / max(accepted_count, 1),
        "rejected_confidence": agg["rejected_conf_sum"] / max(rejected_count, 1),
        "accepted_entropy": agg["accepted_entropy_sum"] / max(accepted_count, 1),
        "rejected_entropy": agg["rejected_entropy_sum"] / max(rejected_count, 1),
    }
