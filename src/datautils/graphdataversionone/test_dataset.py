"""Small synthetic checks for variant combinations and alignment rejection."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .saved_dataset import EDGE_FILES, NODE_FILES, load_dataset
from .validation import validate_features, validate_saved_dataset, validate_trial_pair


class SavedDatasetTests(unittest.TestCase):
    """Exercise graph assembly and reject corrupted files without EEG sources."""

    def setUp(self) -> None:
        """Write a complete two-trial fixture in a temporary directory.

        Returns
        -------
        None
            Fixture files are removed automatically after each test.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        n_samples = 2
        first, second = np.triu_indices(29, k=1)
        arrays: dict[str, np.ndarray] = {
            "edge_index.npy": np.concatenate(
                (np.stack((first, second)), np.stack((second, first))), axis=1
            ).astype(np.int64),
            "labels.npy": np.array([0, 1], dtype=np.int64),
        }
        for variant_index, filename in enumerate(NODE_FILES.values()):
            values = np.full((n_samples, 29, 6), -20 + variant_index, dtype=np.float32)
            values[:, :, -1] = 0.5
            arrays[filename] = values
        for variant_index, filename in enumerate(EDGE_FILES.values()):
            values = np.full((n_samples, 812, 5), (variant_index + 1) / 10, dtype=np.float32)
            values += np.arange(5, dtype=np.float32)[None, None, :] / 100
            arrays[filename] = values
        self.metadata = {
            "n_samples": n_samples,
            "channel_names": [f"EEG{index}" for index in range(29)],
            "node_feature_names": ["delta", "theta", "alpha", "beta", "gamma", "entropy"],
            "band_names": ["delta", "theta", "alpha", "beta", "gamma"],
            "class_mapping": {"left_hand": 0, "right_hand": 1},
            "subjects": [1],
            "trial_n_samples": 2000,
            "expected_trials_per_subject": 2,
            "expected_class_counts_per_subject": {"left_hand": 1, "right_hand": 1},
            "arrays": {
                filename: {"shape": list(values.shape), "dtype": str(values.dtype)}
                for filename, values in arrays.items()
            },
            "edge_variants": {
                name: {"file": filename, "band_names": ["delta", "theta", "alpha", "beta", "gamma"]}
                for name, filename in EDGE_FILES.items()
            },
        }
        coordinates = {name: [0.01, 0.02, 0.09] for name in self.metadata["channel_names"]}
        self.metadata["montage"] = {
            "channel_names": list(self.metadata["channel_names"]),
            "source_electrodes": {"channel_positions": coordinates},
            "analysis_montage": {
                "units": "m", "coordinate_frame": "head", "channel_positions": coordinates,
                "fiducials": {"nasion": [0, 0.1, 0], "lpa": [-0.1, 0, 0], "rpa": [0.1, 0, 0]},
            },
        }
        for filename, values in arrays.items():
            np.save(self.path / filename, values, allow_pickle=False)
        (self.path / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        self.samples = [
            {
                "sample_index": index,
                "subject": 1,
                "trial_index": index,
                "label": index,
                "target": ["left_hand", "right_hand"][index],
                "source_file": "subject_1.edf",
                "start_sample": index * 3000,
                "stop_sample": index * 3000 + 2000,
                "sampling_frequency_hz": 500.0,
            }
            for index in range(n_samples)
        ]
        self._write_samples()

    def _write_samples(self) -> None:
        """Persist the current fixture records in their current order.

        Returns
        -------
        None
            Overwrites only this test's temporary ``samples.tsv``.
        """
        with (self.path / "samples.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.samples[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(self.samples)

    def test_graph_combines_variants_in_requested_column_order(self) -> None:
        """Verify independent node selection, edge order, and copy isolation.

        Returns
        -------
        None
            Asserts graph values and feature names agree with saved arrays.
        """
        dataset = load_dataset(self.path)
        self.assertIsInstance(dataset.nodes["without_csd"], np.memmap)
        self.assertFalse(dataset.nodes["without_csd"].flags.writeable)
        graph = dataset.get_graph(1, "csd", ("plv_csd", "wpli_without_csd"))
        self.assertEqual(graph["x"].shape, (29, 6))
        self.assertEqual(graph["edge_attr"].shape, (812, 10))
        self.assertEqual(graph["y"], 1)
        self.assertEqual(graph["sample"]["trial_index"], 1)
        np.testing.assert_array_equal(graph["x"], dataset.nodes["csd"][1])
        np.testing.assert_array_equal(graph["edge_attr"][:, :5], dataset.edges["plv_csd"][1])
        np.testing.assert_array_equal(graph["edge_attr"][:, 5:], dataset.edges["wpli_without_csd"][1])
        self.assertEqual(graph["edge_feature_names"][0], "plv_csd:delta")
        self.assertEqual(graph["edge_feature_names"][5], "wpli_without_csd:delta")
        graph["x"][0, 0] = 99
        graph["edge_index"][0, 0] = 99
        self.assertEqual(dataset.nodes["csd"][1, 0, 0], -19)
        self.assertEqual(dataset.edge_index[0, 0], 0)

    def test_graph_rejects_invalid_or_ambiguous_selection(self) -> None:
        """Reject invalid indices and variant choices before returning a graph.

        Returns
        -------
        None
            Asserts meaningful selection errors are raised.
        """
        dataset = load_dataset(self.path)
        for index in (-1, 2, True, 0.5):
            with self.subTest(index=index), self.assertRaises(IndexError):
                dataset.get_graph(index)  # type: ignore[arg-type]  # deliberately non-integer/bool
        for selection in ((), ("missing",), ("plv_csd", "plv_csd"), "plv_csd"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                dataset.get_graph(0, edge_variants=selection)
        with self.assertRaises(ValueError):
            dataset.get_graph(0, node_variant="missing")

    def test_validation_reports_counts_for_consistent_files(self) -> None:
        """Validate the fixture and check returned subject and class counts.

        Returns
        -------
        None
            Asserts all trial records are accounted for.
        """
        result = validate_saved_dataset(self.path)
        self.assertEqual(result["n_samples"], 2)
        self.assertEqual(result["subject_counts"], {"1": 2})
        self.assertEqual(result["class_counts"], {"left_hand": 1, "right_hand": 1})

    def test_validation_rejects_swapped_trial_records(self) -> None:
        """Reject records whose row order no longer matches feature arrays.

        Returns
        -------
        None
            Asserts the corrupted file is identified by sample alignment.
        """
        self.samples.reverse()
        self._write_samples()
        with self.assertRaisesRegex(ValueError, "sample_index"):
            validate_saved_dataset(self.path)

    def test_validation_rejects_mislabeled_trial(self) -> None:
        """Reject a target name changed independently of its saved label.

        Returns
        -------
        None
            Asserts trial labels cannot silently drift between files.
        """
        self.samples[0]["target"] = "right_hand"
        self._write_samples()
        with self.assertRaisesRegex(ValueError, "target name"):
            validate_saved_dataset(self.path)

    def test_validation_rejects_asymmetric_and_nonfinite_features(self) -> None:
        """Reject one-sided edge corruption and undefined node values.

        Returns
        -------
        None
            Asserts both numerical errors include their feature context.
        """
        dataset = load_dataset(self.path)
        nodes = {name: values[0].copy() for name, values in dataset.nodes.items()}
        edges = {name: values[0].copy() for name, values in dataset.edges.items()}
        edges["wpli_csd"][0, 0] = 0.9
        with self.assertRaisesRegex(ValueError, "reverse directions"):
            validate_features(nodes, edges, dataset.edge_index)
        edges["wpli_csd"] = dataset.edges["wpli_csd"][0].copy()
        nodes["csd"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_features(nodes, edges, dataset.edge_index)

    def test_loader_rejects_incorrect_band_metadata(self) -> None:
        """Reject edge metadata that disagrees with the shared band order.

        Returns
        -------
        None
            Asserts output column labels cannot misdescribe array columns.
        """
        self.metadata["edge_variants"]["plv_csd"]["band_names"].reverse()
        (self.path / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different band order"):
            load_dataset(self.path)

    def test_trial_pair_rejects_different_windows(self) -> None:
        """Reject EEG/CSD windows that cannot describe the same trial.

        Returns
        -------
        None
            Asserts shape and expected-length checks reject misalignment.
        """
        eeg = np.zeros((29, 2000))
        validate_trial_pair(eeg, eeg.copy(), expected_n_samples=2000)
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_trial_pair(eeg, np.zeros((29, 1999)))
        with self.assertRaisesRegex(ValueError, "expected 1000"):
            validate_trial_pair(eeg, eeg, expected_n_samples=1000)

    def test_validation_rejects_unknown_analysis_coordinate_frame(self) -> None:
        """Reject analysis coordinates that cannot be interpreted as head/m.

        Returns
        -------
        None
            Asserts unknown source conventions are not accepted for analysis.
        """
        self.metadata["montage"]["analysis_montage"]["coordinate_frame"] = "unknown"
        (self.path / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "head coordinate frame"):
            validate_saved_dataset(self.path)


if __name__ == "__main__":
    unittest.main()
