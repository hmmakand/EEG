"""Tests for experiment five's CSD alpha-band wPLI LOSO data pipeline."""

from __future__ import annotations

import dataclasses
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

from src.datautils.graphdataversiontwo.saved_dataset import SavedDataset
from src.ourexperimentversionfive.data.validation import SOURCE_NODE_FEATURE_NAMES, select_node_features
from src.ourexperimentversionfive.data.validation import (
    EXPECTED_EDGES,
    EXPECTED_GRAPHS,
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    EXPECTED_SUBJECT_IDS,
    NODE_FEATURE_NAMES,
)
from src.ourexperimentversionfive.data.csd_alpha_wpli import (
    NODE_VARIANT,
    GraphDataLoaderConfig,
    create_loso_dataloaders,
    create_loso_splits,
    fit_feature_normalization,
    load_dataset,
    validate_dataset,
)


class CsdAlphaWpliContractTests(unittest.TestCase):
    dataset: SavedDataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def test_fixed_schema_and_shapes(self) -> None:
        validate_dataset(self.dataset)
        self.assertEqual(len(self.dataset), EXPECTED_GRAPHS)
        self.assertEqual(
            self.dataset.nodes["csd"].shape,
            (EXPECTED_GRAPHS, EXPECTED_NODES, len(SOURCE_NODE_FEATURE_NAMES)),
        )
        self.assertEqual(
            self.dataset.edges["wpli_csd"].shape,
            (EXPECTED_GRAPHS, EXPECTED_EDGES, 5),
        )
        self.assertEqual(
            tuple(self.dataset.metadata["node_feature_names"]), SOURCE_NODE_FEATURE_NAMES
        )

    def test_schema_validation_rejects_reordered_features(self) -> None:
        changed_metadata = dict(self.dataset.metadata)
        names = list(SOURCE_NODE_FEATURE_NAMES)
        names[0], names[1] = names[1], names[0]
        changed_metadata["node_feature_names"] = names
        changed_dataset = dataclasses.replace(self.dataset, metadata=changed_metadata)
        with self.assertRaisesRegex(ValueError, "names or order"):
            validate_dataset(changed_dataset)

    def test_explicit_feature_contract_has_eight_csd_columns(self) -> None:
        self.assertEqual(len(NODE_FEATURE_NAMES), 8)
        self.assertEqual(NODE_FEATURE_NAMES[-1], "hjorth_complexity")


class CsdAlphaWpliLosoDataLoaderTests(unittest.TestCase):
    dataset: SavedDataset
    config: GraphDataLoaderConfig

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.config = GraphDataLoaderConfig(
            batch_size=8,
            node_normalization="zscore",
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

    def test_loaders_preserve_graph_data_and_normalize_eight_columns(self) -> None:
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

        labels = np.asarray(self.dataset.labels)
        self.assertEqual(
            Counter(map(int, labels[bundle.split.train_graph_indices])),
            Counter({0: 880, 1: 880}),
        )

    def test_normalization_uses_training_pool_for_every_subject(self):
        bundle = create_loso_dataloaders(test_subject_id=1, config=self.config, dataset=self.dataset)
        raw = np.asarray(select_node_features(self.dataset, "csd", bundle.split.train_graph_indices), dtype=np.float64)
        expected_mean = raw.mean(axis=(0, 1))
        expected_std = raw.std(axis=(0, 1))
        for subject in (1, *bundle.split.train_subject_ids, *bundle.split.validation_subject_ids):
            np.testing.assert_allclose(bundle.normalization.mean_by_subject[subject], expected_mean, rtol=1e-6)
            np.testing.assert_allclose(bundle.normalization.standard_deviation_by_subject[subject], expected_std, rtol=1e-6)
        first_index = bundle.split.validation_graph_indices[0]
        raw_validation = select_node_features(self.dataset, "csd", first_index)
        expected = (raw_validation - expected_mean) / expected_std
        np.testing.assert_allclose(next(iter(bundle.validation)).x[:29], expected, atol=1e-6, rtol=1e-5)


    def test_edge_attr_matches_alpha_band_column(self) -> None:
        bundle = create_loso_dataloaders(
            test_subject_id=1,
            config=self.config,
            dataset=self.dataset,
        )
        first_test_index = int(bundle.split.test_graph_indices[0])
        expected_alpha = np.asarray(
            self.dataset.edges["wpli_csd"][first_test_index, :, 2],
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
            "src.ourexperimentversionfive.data.csd_alpha_wpli.validate_dataset"
        ) as validate:
            create_loso_dataloaders(
                test_subject_id=1,
                config=self.config,
                dataset=self.dataset,
                dataset_validated=True,
            )
        validate.assert_not_called()


class FitFeatureNormalizationTests(unittest.TestCase):
    dataset: SavedDataset

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()

    def _subject_ids(self) -> np.ndarray:
        return np.asarray(
            [sample["subject"] for sample in self.dataset.samples], dtype=np.int64
        )

    def test_graph_indices_restrict_fitting_scope(self) -> None:
        subject_ids = self._subject_ids()
        two_subject_indices = np.flatnonzero(np.isin(subject_ids, (1, 2)))
        normalization = fit_feature_normalization(
            self.dataset, graph_indices=two_subject_indices
        )
        self.assertEqual(set(normalization.mean_by_subject), set(range(1, 51)))
        self.assertEqual(set(normalization.fit_graph_indices), set(two_subject_indices))
        self.assertEqual(set(normalization.standard_deviation_by_subject), set(range(1, 51)))

    def test_graph_indices_scoped_stats_match_manual_computation(self) -> None:
        # Mimic a within-subject fold: only a subset of one subject's own
        # trials (e.g. a train split with a held-out evaluation fold
        # excluded), not that subject's every trial.
        subject_ids = self._subject_ids()
        subject_indices = np.flatnonzero(subject_ids == 3)
        train_only_indices = subject_indices[:32]  # exclude the other 8 (eval fold)

        normalization = fit_feature_normalization(
            self.dataset, graph_indices=train_only_indices
        )
        raw = np.asarray(
            select_node_features(self.dataset, NODE_VARIANT, train_only_indices), dtype=np.float64
        )
        expected_mean = raw.mean(axis=(0, 1))
        expected_standard_deviation = raw.std(axis=(0, 1))
        np.testing.assert_allclose(
            normalization.mean_by_subject[3].numpy(), expected_mean, rtol=1e-6
        )
        np.testing.assert_allclose(
            normalization.standard_deviation_by_subject[3].numpy(),
            expected_standard_deviation,
            rtol=1e-6,
        )

        # And it must differ from fitting on that subject's full 40 trials --
        # otherwise the scoping wouldn't be doing anything.
        full_normalization = fit_feature_normalization(self.dataset, graph_indices=subject_indices)
        self.assertFalse(
            np.allclose(
                normalization.mean_by_subject[3].numpy(),
                full_normalization.mean_by_subject[3].numpy(),
            )
        )

    def test_zscore_requires_training_indices_and_none_has_no_statistics(self):
        with self.assertRaisesRegex(ValueError, "explicit training"):
            fit_feature_normalization(self.dataset)
        disabled = fit_feature_normalization(self.dataset, mode="none")
        self.assertEqual(disabled.mode, "none")
        self.assertEqual(disabled.mean_by_subject, {})
        self.assertEqual(disabled.standard_deviation_by_subject, {})



if __name__ == "__main__":
    unittest.main()
