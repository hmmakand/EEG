"""Tests for Liu2024 experiment splitters and DataLoaders."""

import unittest
from collections import Counter

import torch
from torch.utils.data import RandomSampler, SequentialSampler

from src.datautils.MoabbLiu2024 import Liu2024TorchDataset

from .config import DataLoaderConfig
from .dataloaders import (
    create_group_kfold_dataloaders,
    create_loso_dataloaders,
    create_within_subject_dataloaders,
)


class Liu2024DataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Liu2024TorchDataset()
        cls.config = DataLoaderConfig(batch_size=16, pin_memory=False, seed=42)

    def _counts(self, indices):
        return Counter(map(int, self.dataset.y[indices]))

    def test_group_kfold(self):
        bundle = create_group_kfold_dataloaders(
            fold=0, config=self.config, dataset=self.dataset
        )
        split = bundle.split
        self.assertEqual(tuple(map(len, (
            split.train_subject_ids,
            split.validation_subject_ids,
            split.test_subject_ids,
        ))), (32, 8, 10))
        self.assertEqual(tuple(map(len, (
            split.train_indices,
            split.validation_indices,
            split.test_indices,
        ))), (1280, 320, 400))
        self.assertEqual(self._counts(split.test_indices), Counter({0: 200, 1: 200}))
        self.assertIsInstance(bundle.train.sampler, RandomSampler)
        self.assertIsInstance(bundle.test.sampler, SequentialSampler)

    def test_loso(self):
        bundle = create_loso_dataloaders(
            test_subject_id=1, config=self.config, dataset=self.dataset
        )
        split = bundle.split
        self.assertEqual(tuple(map(len, (
            split.train_subject_ids,
            split.validation_subject_ids,
            split.test_subject_ids,
        ))), (49, 0, 1))
        self.assertEqual(tuple(map(len, (
            split.train_indices,
            split.validation_indices,
            split.test_indices,
        ))), (1960, 0, 40))
        self.assertEqual(split.test_subject_ids, (1,))
        self.assertIsNone(bundle.validation)

    def test_within_subject(self):
        bundle = create_within_subject_dataloaders(
            subject_id=1, fold=0, config=self.config, dataset=self.dataset
        )
        split = bundle.split
        self.assertEqual(tuple(map(len, (
            split.train_indices,
            split.validation_indices,
            split.test_indices,
        ))), (24, 8, 8))
        self.assertEqual(self._counts(split.train_indices), Counter({0: 12, 1: 12}))
        self.assertEqual(self._counts(split.validation_indices), Counter({0: 4, 1: 4}))
        self.assertEqual(self._counts(split.test_indices), Counter({0: 4, 1: 4}))

    def test_batch_shape_and_metadata(self):
        bundle = create_group_kfold_dataloaders(
            fold=0, config=self.config, dataset=self.dataset
        )
        x, y, info = next(iter(bundle.train))
        self.assertEqual(tuple(x.shape), (16, 29, 2000))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(tuple(y.shape), (16,))
        self.assertEqual(set(info), {"subject", "trial"})


if __name__ == "__main__":
    unittest.main()
