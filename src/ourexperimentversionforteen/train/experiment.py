"""Orchestrate epochs, folds, and subjects with explicit configuration."""
from tqdm import tqdm
from .config import TrainingConfig, validate_training_config
from .engine import train_epoch, evaluate
from .metrics import (compute_classification_metrics, summarize_fold_results,
                      FoldResult, SubjectResult)
from .reporting import print_subject_result
from .selection import initialize_selection, update_selection, finalize_selection
from .setup import seed_fold, build_model, build_optimizer, build_criterion
from .splitting import make_fold_indices, select_fold_data, build_loaders

if __package__ == "train":
    from dataset import DatasetConfig, build_subject_dataset
else:
    from ..dataset import DatasetConfig, build_subject_dataset


def run_fold(train_data, test_data, device, training_config: TrainingConfig,
             subject_number: int, fold_number: int) -> FoldResult:
    validate_training_config(training_config)
    seed_fold(training_config.fold_seed)
    train_loader, test_loader = build_loaders(train_data, test_data, training_config)
    model = build_model(train_data[0].num_node_features, device, training_config)
    optimizer = build_optimizer(model, training_config)
    criterion = build_criterion(training_config)
    selection = initialize_selection()
    for epoch in tqdm(range(1, training_config.num_epochs + 1),
                      desc=f"Training Subject {subject_number} Fold {fold_number}",
                      leave=False, disable=not training_config.show_progress):
        train_epoch(model, train_loader, optimizer, criterion, device)
        train_accuracy = evaluate(model, train_loader, device)
        evaluation = evaluate(model, test_loader, device, collect_predictions=True)
        if selection.epoch is None or evaluation.accuracy > selection.test_accuracy:
            metrics = compute_classification_metrics(evaluation.labels, evaluation.predictions)
            selection = update_selection(selection, epoch, train_accuracy, evaluation.accuracy,
                                         training_config.selection_policy, metrics)
    assert selection.epoch is not None and selection.metrics is not None
    return {"fold": fold_number, "selected_epoch": selection.epoch,
            "accuracy": finalize_selection(selection), **selection.metrics}


def run_subject(data_dir, subject_number: int, device, dataset_config: DatasetConfig,
                training_config: TrainingConfig) -> SubjectResult:
    validate_training_config(training_config)
    if len(dataset_config.trials.label_map) != 2:
        raise ValueError("The current GAT classifier requires exactly two target classes")
    data_list = build_subject_dataset(data_dir, subject_number, dataset_config)
    fold_results: list[FoldResult] = []
    for fold, (train_indices, test_indices) in enumerate(make_fold_indices(data_list, training_config), start=1):
        train_data, test_data = select_fold_data(data_list, train_indices, test_indices)
        fold_results.append(run_fold(train_data, test_data, device, training_config, subject_number, fold))
    return summarize_fold_results(fold_results)


def run_experiment(data_dir, device, dataset_config: DatasetConfig,
                   training_config: TrainingConfig) -> dict[int, SubjectResult]:
    validate_training_config(training_config)
    subject_results = {}
    for subject_number in tqdm(training_config.subjects, desc="Processing Subjects",
                               disable=not training_config.show_progress):
        results = run_subject(data_dir, subject_number, device, dataset_config, training_config)
        subject_results[subject_number] = results
        print_subject_result(subject_number, results)
    return subject_results
