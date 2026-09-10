"""Tests for experiment four's without-CSD alpha-band wPLI LOSO data pipeline."""

from __future__ import annotations

import dataclasses
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

from src.datautils.graphdataversionone.saved_dataset import SavedDataset
from src.ourexperimentversionfour.data import (
    EXPECTED_EDGES,
    EXPECTED_GRAPHS,
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    EXPECTED_SUBJECT_IDS,
    NODE_FEATURE_NAMES,
    GraphDataLoaderConfig,
    create_loso_dataloaders,
    create_loso_splits,
    load_dataset,
    validate_dataset,
)


class WithoutCsdAlphaWpliContractTests(unittest.TestCase):
    dataset: SavedDataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_fixed_schema_and_shapes(self) -> None:
        validate_dataset(self.dataset)
        self.assertEqual(len(self.dataset), EXPECTED_GRAPHS)
        self.assertEqual(
            self.dataset.nodes["without_csd"].shape,
            (EXPECTED_GRAPHS, EXPECTED_NODES, EXPECTED_NODE_FEATURES),
        )
        self.assertEqual(
            self.dataset.edges["wpli_without_csd"].shape,
            (EXPECTED_GRAPHS, EXPECTED_EDGES, 5),
        )
        self.assertEqual(
            tuple(self.dataset.metadata["node_feature_names"]), NODE_FEATURE_NAMES
        )

    def test_schema_validation_rejects_reordered_features(self) -> None:
        changed_metadata = dict(self.dataset.metadata)
        names = list(NODE_FEATURE_NAMES)
        names[0], names[1] = names[1], names[0]
        changed_metadata["node_feature_names"] = names
        changed_dataset = dataclasses.replace(self.dataset, metadata=changed_metadata)
        with self.assertRaisesRegex(ValueError, "names or order"):
            validate_dataset(changed_dataset)

    def test_explicit_feature_contract_has_six_without_csd_columns(self) -> None:
        self.assertEqual(len(NODE_FEATURE_NAMES), 6)
        self.assertEqual(NODE_FEATURE_NAMES[-1], "spectral_entropy")


class WithoutCsdAlphaWpliLosoDataLoaderTests(unittest.TestCase):
    dataset: SavedDataset
    config: GraphDataLoaderConfig

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.config = GraphDataLoaderConfig(
            batch_size=8,
            pin_memory=False,
            seed=42,
        )

    def _subject_ids(self) -> np.ndarray:
        return np.asarray(
            [sample["subject"] for sample in self.dataset.samples], dtype=np.int64
        )

    def test_all_loso_splits_are_reproducible_complete_and_subject_disjoint(self) -> None:
        subject_ids = self._subject_ids()
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

    def test_loaders_preserve_graph_data_and_normalize_six_columns(self) -> None:
        bundle = create_loso_dataloaders(
            test_subject_id=1,
            config=self.config,
            dataset=self.dataset,
        )
        batch = next(iter(bundle.train))
        self.assertEqual(tuple(batch.x.shape), (8 * EXPECTED_NODES, EXPECTED_NODE_FEATURES))
        self.assertEqual(tuple(batch.edge_attr.shape), (8 * EXPECTED_EDGES, 1))
        self.assertEqual(batch.x.dtype, torch.float32)
        self.assertEqual(batch.edge_attr.dtype, torch.float32)
        self.assertEqual(tuple(batch.y.shape), (8,))
        fold_subjects = (
            bundle.split.train_subject_ids
            + bundle.split.validation_subject_ids
            + (bundle.split.test_subject_id,)
        )
        for subject in fold_subjects:
            self.assertEqual(
                tuple(bundle.normalization.mean_by_subject[subject].shape),
                (EXPECTED_NODE_FEATURES,),
            )
            self.assertEqual(
                tuple(bundle.normalization.standard_deviation_by_subject[subject].shape),
                (EXPECTED_NODE_FEATURES,),
            )

        subject_ids = self._subject_ids()
        labels = np.asarray(self.dataset.labels)
        self.assertEqual(
            Counter(map(int, labels[bundle.split.train_graph_indices])),
            Counter({0: 880, 1: 880}),
        )
        self.assertNotIn(1, subject_ids[bundle.split.train_graph_indices])

    def test_normalization_is_per_subject_using_only_that_subjects_own_nodes(self) -> None:
        bundle = create_loso_dataloaders(
            test_subject_id=1,
            config=self.config,
            dataset=self.dataset,
        )
        subject_ids = self._subject_ids()

        # Check a training subject's stats come only from its own 40 trials,
        # not pooled across the other 43 training subjects.
        one_train_subject = bundle.split.train_subject_ids[0]
        raw_subject = np.asarray(
            self.dataset.nodes["without_csd"][subject_ids == one_train_subject],
            dtype=np.float64,
        )
        expected_mean = raw_subject.mean(axis=(0, 1))
        expected_standard_deviation = raw_subject.std(axis=(0, 1))
        np.testing.assert_allclose(
            bundle.normalization.mean_by_subject[one_train_subject].numpy(),
            expected_mean,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            bundle.normalization.standard_deviation_by_subject[one_train_subject].numpy(),
            expected_standard_deviation,
            rtol=1e-6,
        )

        # The held-out test subject also gets its own stats (from its own raw
        # features only, never its labels) rather than the training pool's.
        test_subject = bundle.split.test_subject_id
        raw_test_subject = np.asarray(
            self.dataset.nodes["without_csd"][subject_ids == test_subject],
            dtype=np.float64,
        )
        expected_test_mean = raw_test_subject.mean(axis=(0, 1))
        np.testing.assert_allclose(
            bundle.normalization.mean_by_subject[test_subject].numpy(),
            expected_test_mean,
            rtol=1e-6,
        )
        self.assertFalse(
            np.allclose(expected_test_mean, expected_mean, rtol=1e-3)
        )

        first_validation_index = int(bundle.split.validation_graph_indices[0])
        validation_subject = int(self.dataset.samples[first_validation_index]["subject"])
        raw_validation = np.asarray(
            self.dataset.nodes["without_csd"][first_validation_index],
            dtype=np.float32,
        )
        first_batch = next(iter(bundle.validation))
        expected_validation = (
            raw_validation - bundle.normalization.mean_by_subject[validation_subject].numpy()
        ) / bundle.normalization.standard_deviation_by_subject[validation_subject].numpy()
        np.testing.assert_allclose(
            first_batch.x[:EXPECTED_NODES].numpy(),
            expected_validation,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_edge_attr_matches_alpha_band_column(self) -> None:
        bundle = create_loso_dataloaders(
            test_subject_id=1,
            config=self.config,
            dataset=self.dataset,
        )
        first_test_index = int(bundle.split.test_graph_indices[0])
        expected_alpha = np.asarray(
            self.dataset.edges["wpli_without_csd"][first_test_index, :, 2],
            dtype=np.float32,
        )
        first_batch = next(iter(bundle.test))
        np.testing.assert_allclose(
            first_batch.edge_attr[:EXPECTED_EDGES, 0].numpy(),
            expected_alpha,
            rtol=1e-6,
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
            "src.ourexperimentversionfour.data.without_csd_alpha_wpli.validate_dataset"
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
