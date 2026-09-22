"""Version-five feature selection and optional train-only normalization regressions."""

import dataclasses
import unittest

import numpy as np
import torch

from .combinations import COMBINATIONS
from .loso_split import GraphDataLoaderConfig
from .validation import band_index, load_dataset, select_node_features, selected_node_indices
from ..model import EEGGCN1
from ..training.config import TrainingConfig, WithinSubjectConfig
from ..training.train import build_parser as loso_parser
from ..training.search_train import build_parser as search_parser
from ..training.within_subject_cli import build_parser as within_parser


class FeaturePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset()

    def test_all_columns_are_selected_and_source_is_preserved(self):
        for combination in COMBINATIONS.values():
            variant = combination.node_variant
            values = np.broadcast_to(np.arange(8, dtype=np.float32), (2000, 29, 8)).copy()
            modified = dataclasses.replace(self.dataset, nodes={**self.dataset.nodes, variant: values})
            selected = select_node_features(modified, variant, 0)
            np.testing.assert_array_equal(selected[0], np.arange(8))
            self.assertEqual(selected_node_indices(modified), list(range(8)))
            selected[:] = 0
            np.testing.assert_array_equal(values[0, 0], np.arange(8))

    def test_disabled_graphs_and_weighted_model(self):
        for combination in COMBINATIONS.values():
            with self.subTest(combination=combination.name):
                bundle = combination.create_loso_dataloaders(
                    dataset=self.dataset, dataset_validated=True, test_subject_id=1,
                    config=GraphDataLoaderConfig(batch_size=2, pin_memory=False, node_normalization="none"))
                batch = next(iter(bundle.test))
                expected = select_node_features(self.dataset, combination.node_variant, bundle.split.test_graph_indices[:2])
                np.testing.assert_array_equal(batch.x, expected.reshape(58, 8))
                self.assertEqual(bundle.normalization.mean_by_subject, {})
                self.assertEqual(bundle.normalization.fit_graph_indices, ())
                self.assertEqual(tuple(batch.edge_attr.shape), (1624, 1))
                first_index = bundle.split.test_graph_indices[0]
                expected_band_index = band_index(self.dataset, combination.band_name)
                np.testing.assert_array_equal(batch.edge_attr[:812, 0], self.dataset.edges[combination.edge_variant][first_index, :, expected_band_index])
                np.testing.assert_array_equal(batch.edge_attr[:406], batch.edge_attr[406:812])
                model = EEGGCN1()
                logits = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(-1), batch.batch)
                self.assertEqual(tuple(logits.shape), (2, 2))
                logits.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                # Disabled fitting must not inspect the dataset at all.
                disabled = combination.fit_feature_normalization(object(), mode="none")
                self.assertEqual(disabled.standard_deviation_by_subject, {})

    def test_held_out_changes_never_change_fitted_statistics(self):
        cfg = GraphDataLoaderConfig(batch_size=2, pin_memory=False, node_normalization="zscore")
        for combination in COMBINATIONS.values():
            for protocol in ("loso", "inner", "within"):
                def loaders(dataset):
                    if protocol == "loso":
                        b = combination.create_loso_dataloaders(dataset=dataset, dataset_validated=True, test_subject_id=1, config=cfg)
                        return b.normalization, b.split.train_graph_indices, b.test
                    if protocol == "inner":
                        train, evaluation, norm = combination.create_group_dataloaders(dataset=dataset,
                            train_subject_ids=(2, 3), validation_subject_ids=(1,), config=cfg, dataset_validated=True)
                    else:
                        train, evaluation, norm = combination.create_within_subject_dataloaders(dataset=dataset,
                            train_graph_indices=np.arange(32), evaluation_graph_indices=np.arange(32, 40),
                            config=cfg, dataset_validated=True)
                    return norm, train.dataset.graph_indices, evaluation
                with self.subTest(combination=combination.name, protocol=protocol):
                    norm, train_indices, evaluation = loaders(self.dataset)
                    held_out = np.setdiff1d(np.arange(len(self.dataset)), train_indices)
                    values = self.dataset.nodes[combination.node_variant].copy()
                    values[held_out] += 1000
                    changed = dataclasses.replace(self.dataset, nodes={**self.dataset.nodes, combination.node_variant: values})
                    changed_norm, _, _ = loaders(changed)
                    subject = next(iter(norm.mean_by_subject))
                    torch.testing.assert_close(norm.mean_by_subject[subject], changed_norm.mean_by_subject[subject], rtol=0, atol=0)
                    torch.testing.assert_close(norm.standard_deviation_by_subject[subject], changed_norm.standard_deviation_by_subject[subject], rtol=0, atol=0)
                    self.assertEqual(set(norm.fit_graph_indices), set(train_indices))
                    raw = select_node_features(self.dataset, combination.node_variant, train_indices).astype(np.float64)
                    np.testing.assert_allclose(norm.mean_by_subject[subject], raw.mean(axis=(0, 1)), rtol=1e-6)
                    graph_index = evaluation.dataset.graph_indices[0]
                    expected = (select_node_features(self.dataset, combination.node_variant, graph_index) - norm.mean_by_subject[subject].numpy()) / norm.standard_deviation_by_subject[subject].numpy()
                    np.testing.assert_allclose(evaluation.dataset.get(0).x, expected, atol=1e-6)

    def test_modes_and_invalid_contracts(self):
        for parser, args in ((loso_parser, ["--test-subject", "1"]),
                             (search_parser, ["--test-subject", "1"]),
                             (within_parser, ["--subjects", "1"])):
            for mode in ("none", "zscore"):
                self.assertEqual(parser().parse_args(args + ["--node-normalization", mode]).node_normalization, mode)
        for config, default in (
            (TrainingConfig, "none"),
            (WithinSubjectConfig, "zscore"),
            (GraphDataLoaderConfig, "none"),
        ):
            self.assertEqual(config().node_normalization, default)
            with self.assertRaises(ValueError):
                config(node_normalization="unknown")
        for metadata in ({**self.dataset.metadata, "schema_version": 1},
                         {**self.dataset.metadata, "node_feature_names": list(reversed(self.dataset.metadata["node_feature_names"]))}):
            with self.assertRaises(ValueError):
                selected_node_indices(dataclasses.replace(self.dataset, metadata=metadata))
