from __future__ import annotations

from omegaconf import OmegaConf

from eeg_bci.tracking.naming import tensorboard_dir


def test_tensorboard_dir_mirrors_hydra_run_layout() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"label": "liu2024", "split": {"source": "chronological", "method": "train_test"}},
            "experiment_name": "within_subject_full_liu2024",
            "model": {"label": "eegnet"},
        }
    )

    run_root = tensorboard_dir(cfg, "2026-06-28_03-54-35__seed42")
    subject_path = tensorboard_dir(cfg, "2026-06-28_03-54-35__seed42", "s01")

    assert str(run_root) == (
        "outputs/tensorboard/liu2024/within_subject_full_liu2024/train_test/eegnet/"
        "2026-06-28_03-54-35__seed42"
    )
    assert str(subject_path) == (
        "outputs/tensorboard/liu2024/within_subject_full_liu2024/train_test/eegnet/"
        "2026-06-28_03-54-35__seed42/s01"
    )
