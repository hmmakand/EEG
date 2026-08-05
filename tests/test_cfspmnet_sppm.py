from __future__ import annotations

import numpy as np
import pytest

from eeg_bci.cfspmnet.sppm import build_shared_private_signature_prototypes


def _make_class_cluster(rng, center, n_samples, dim, noise=0.05):
    samples = center + noise * rng.standard_normal((n_samples, dim))
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    return (samples / norms).astype(np.float32)


def test_prototypes_are_unit_norm_and_close_to_class_centers() -> None:
    rng = np.random.default_rng(0)
    dim = 8
    center_0 = np.zeros(dim)
    center_0[0] = 1.0
    center_1 = np.zeros(dim)
    center_1[1] = 1.0

    class_0 = _make_class_cluster(rng, center_0, 40, dim)
    class_1 = _make_class_cluster(rng, center_1, 40, dim)
    signature_vectors = np.concatenate([class_0, class_1], axis=0)
    labels = np.concatenate([np.zeros(40, dtype=np.int64), np.ones(40, dtype=np.int64)])

    prototypes, tolerances = build_shared_private_signature_prototypes(
        signature_vectors, labels, num_classes=2, floor=0.0
    )

    assert prototypes.shape == (2, dim)
    np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-5)
    assert np.dot(prototypes[0], center_0) > 0.9
    assert np.dot(prototypes[1], center_1) > 0.9
    assert tolerances.shape == (2,)


def test_matching_tolerance_floor_wins_when_computed_value_is_lower() -> None:
    rng = np.random.default_rng(1)
    dim = 4
    center = np.array([1.0, 0.0, 0.0, 0.0])
    # High noise -> low mean-minus-std similarity, so the floor should dominate.
    signature_vectors = _make_class_cluster(rng, center, 30, dim, noise=0.8)
    labels = np.zeros(30, dtype=np.int64)

    _, tolerances = build_shared_private_signature_prototypes(
        signature_vectors, labels, num_classes=1, floor=0.95
    )

    assert tolerances[0] == pytest.approx(0.95)


def test_matching_tolerance_uses_computed_value_when_above_floor() -> None:
    rng = np.random.default_rng(2)
    dim = 4
    center = np.array([1.0, 0.0, 0.0, 0.0])
    signature_vectors = _make_class_cluster(rng, center, 30, dim, noise=0.01)
    labels = np.zeros(30, dtype=np.int64)

    _, tolerances = build_shared_private_signature_prototypes(
        signature_vectors, labels, num_classes=1, floor=0.0
    )

    assert 0.0 < tolerances[0] < 1.0


def test_raises_when_a_class_has_no_samples() -> None:
    signature_vectors = np.ones((5, 3), dtype=np.float32)
    labels = np.zeros(5, dtype=np.int64)

    with pytest.raises(ValueError, match="No source samples found for class 1"):
        build_shared_private_signature_prototypes(signature_vectors, labels, num_classes=2, floor=0.5)
