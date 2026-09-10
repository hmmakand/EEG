"""Tests for experiment three's fixed broadcast-11 LOSO data pipeline."""

from __future__ import annotations

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from src.datautils.PlvLiu2024Broadcast11 import PlvLiu2024GraphDataset
from src.ourexperimentversionthree.data import (
    BROADCAST_NODE_FEATURE_NAMES,
    DATASET_DIR,
    EXPECTED_GRAPHS,
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    EXPECTED_SUBJECT_IDS,
    NODE_FEATURE_NAMES,
    GraphDataLoaderConfig,
    create_loso_dataloaders,
    create_loso_splits,
    load_broadcast_source_features,
    load_dataset,
    validate_dataset,
)


class Broadcast11DatasetContractTests(unittest.TestCase):
    dataset: PlvLiu2024GraphDataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_fixed_schema_feature_order_finiteness_and_exact_broadcast(self) -> None:
        validate_dataset(self.dataset)
        self.assertEqual(len(self.dataset), EXPECTED_GRAPHS)
        self.assertEqual(
            self.dataset.arrays.node_features.shape,
            (EXPECTED_GRAPHS, EXPECTED_NODES, EXPECTED_NODE_FEATURES),
        )
        self.assertEqual(
            tuple(self.dataset.metadata["node_feature_names"]), NODE_FEATURE_NAMES
        )
        self.assertIsNone(self.dataset.arrays.graph_features)

        source = load_broadcast_source_features(self.dataset)
        self.assertEqual(source.shape, (EXPECTED_GRAPHS, 7))
        self.assertTrue(np.isfinite(source).all())
        expected = np.broadcast_to(
            source[:, None, :], self.dataset.arrays.node_features[:, :, 4:].shape
        )
        self.assertTrue(
            np.array_equal(self.dataset.arrays.node_features[:, :, 4:], expected)
        )

    def test_schema_validation_rejects_reordered_features(self) -> None:
        original_metadata = self.dataset.metadata
        changed_metadata = dict(original_metadata)
        names = list(NODE_FEATURE_NAMES)
        names[0], names[1] = names[1], names[0]
        changed_metadata["node_feature_names"] = names
        self.dataset.metadata = changed_metadata
        try:
            with self.assertRaisesRegex(ValueError, "names or order"):
                validate_dataset(self.dataset)
        finally:
            self.dataset.metadata = original_metadata

    def test_schema_validation_rejects_one_inexact_broadcast_value(self) -> None:
        writable_copy = PlvLiu2024GraphDataset(DATASET_DIR, mmap_mode="c")
        writable_copy.arrays.node_features[0, 0, 4] += np.float32(1.0)
        with self.assertRaisesRegex(ValueError, "not exact per-graph broadcasts"):
            validate_dataset(writable_copy)

    def test_explicit_feature_contract_contains_four_local_and_seven_broadcast(self) -> None:
        self.assertEqual(len(NODE_FEATURE_NAMES), 11)
        self.assertEqual(NODE_FEATURE_NAMES[4:], BROADCAST_NODE_FEATURE_NAMES)


class Broadcast11LosoDataLoaderTests(unittest.TestCase):
    dataset: PlvLiu2024GraphDataset
    config: GraphDataLoaderConfig

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.config = GraphDataLoaderConfig(
            batch_size=8,
            pin_memory=False,
            seed=42,
        )

    def test_all_loso_splits_are_reproducible_complete_and_subject_disjoint(self) -> None:
        subject_ids = np.asarray(self.dataset.arrays.subject_ids)
        first = create_loso_splits(subject_ids, seed=42)
        repeated = create_loso_splits(subject_ids, seed=42)

        self.assertEqual(len(first), 50)
        self.assertEqual(
            tuple(split.test_subject_id for split in first), EXPECTED_SUBJECT_IDS
        )
        for split, same_split in zip(first, repeated):
            self.assertEqual(len(split.train_subject_ids), 44)
            self.assertEqual(len(split.validation_subject_ids), 5)
            self.assertEqual(len(split.train_graph_indices), 1_760)
            self.assertEqual(len(split.validation_graph_indices), 200)
            self.assertEqual(len(split.test_graph_indices), 40)
            self.assertEqual(
                split.validation_subject_ids, same_split.validation_subject_ids
            )
            self.assertTrue(
                np.array_equal(
                    split.train_graph_indices, same_split.train_graph_indices
                )
            )
            subject_sets = (
                set(split.train_subject_ids),
                set(split.validation_subject_ids),
                {split.test_subject_id},
            )
            self.assertFalse(subject_sets[0] & subject_sets[1])
            self.assertFalse(subject_sets[0] & subject_sets[2])
            self.assertFalse(subject_sets[1] & subject_sets[2])

    def test_loaders_preserve_graph_data_and_normalize_eleven_columns(self) -> None:
        bundle = create_loso_dataloaders(
            test_subject_id=1,
            config=self.config,
            dataset=self.dataset,
        )
        self.assertIsInstance(bundle.train.sampler, RandomSampler)
        self.assertIsInstance(bundle.validation.sampler, SequentialSampler)
        self.assertIsInstance(bundle.test.sampler, SequentialSampler)

        x_batch, adjacency_batch, y_batch = next(iter(bundle.train))
        self.assertEqual(tuple(x_batch.shape), (8, EXPECTED_NODES, 11))
        self.assertEqual(tuple(adjacency_batch.shape), (8, EXPECTED_NODES, EXPECTED_NODES))
        self.assertEqual(x_batch.dtype, torch.float32)
        self.assertEqual(adjacency_batch.dtype, torch.float32)
        self.assertEqual(tuple(y_batch.shape), (8,))
        self.assertTrue(set(torch.unique(adjacency_batch).tolist()) <= {0.0, 1.0})
        self.assertEqual(tuple(bundle.normalization.mean.shape), (11,))
        self.assertEqual(
            tuple(bundle.normalization.standard_deviation.shape), (11,)
        )

        labels = np.asarray(self.dataset.arrays.labels)
        self.assertEqual(
            Counter(map(int, labels[bundle.split.train_graph_indices])),
            Counter({0: 880, 1: 880}),
        )

    def test_normalization_uses_only_training_subject_nodes(self) -> None:
        bundle = create_loso_dataloaders(
            test_subject_id=1,
            config=self.config,
            dataset=self.dataset,
        )
        raw_training = np.asarray(
            self.dataset.arrays.node_features[bundle.split.train_graph_indices],
            dtype=np.float64,
        )
        expected_mean = raw_training.mean(axis=(0, 1))
        expected_standard_deviation = raw_training.std(axis=(0, 1))
        np.testing.assert_allclose(
            bundle.normalization.mean.numpy(), expected_mean, rtol=1e-6
        )
        np.testing.assert_allclose(
            bundle.normalization.standard_deviation.numpy(),
            expected_standard_deviation,
            rtol=1e-6,
        )

        # Verify an evaluation sample is transformed with those training-only
        # statistics, rather than being fitted independently.
        first_validation_index = int(bundle.split.validation_graph_indices[0])
        raw_validation = np.asarray(
            self.dataset.arrays.node_features[first_validation_index],
            dtype=np.float32,
        )
        x_validation_batch, _, _ = next(iter(bundle.validation))
        expected_validation = (
            raw_validation - bundle.normalization.mean.numpy()
        ) / bundle.normalization.standard_deviation.numpy()
        np.testing.assert_allclose(
            x_validation_batch[0].numpy(),
            expected_validation,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_unknown_test_subject_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown test subject"):
            create_loso_dataloaders(
                test_subject_id=999,
                config=self.config,
                dataset=self.dataset,
            )

    def test_prevalidated_dataset_can_skip_redundant_full_scan(self) -> None:
        with patch(
            "src.ourexperimentversionthree.data.dataloaders.validate_dataset"
        ) as validate:
            create_loso_dataloaders(
                test_subject_id=1,
                config=self.config,
                dataset=self.dataset,
                dataset_validated=True,
            )
        validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
