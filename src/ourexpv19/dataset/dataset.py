"""Orchestrate subject preparation; individual rules live in sibling modules."""
from .config import DatasetConfig, default_dataset_config
from .data_io import load_mat_file, read_sessions, resolve_data_dir, subject_file_path
from .trials import extract_trials, order_trials
from .preprocessing import preprocess_trial
from .alignment import align_trials
from .connectivity import compute_connectivity, compute_edge_features
from .features import compute_node_features
from .graphs import make_pyg_graph
from .validation import validate_config, validate_trial, validate_graph


def build_trial_graph(trial, config: DatasetConfig | None = None):
    """One graph from one trial. Per-subject alignment is not applied here:
    pass trials through `align_trials` first (see build_subject_dataset)."""
    config = config or default_dataset_config()
    validate_config(config)
    validate_trial(trial)
    filtered = preprocess_trial(trial, config.sample_rate, config.preprocessing)
    connectivity = compute_connectivity(filtered.signal, config.connectivity)
    features = compute_node_features(filtered.signal, config.sample_rate, config.features)
    edge_features = None
    if config.connectivity.edge_features:
        edge_features = compute_edge_features(trial.signal, connectivity, config.sample_rate,
                                              config.preprocessing, config.connectivity)
    graph = make_pyg_graph(features, connectivity, trial, config.graphs, edge_features)
    validate_graph(graph)
    return graph


def build_subject_dataset(data_dir, subject_number, config: DatasetConfig | None = None):
    config = config or default_dataset_config()
    validate_config(config)
    sessions = read_sessions(load_mat_file(subject_file_path(resolve_data_dir(data_dir), subject_number)))
    trials = extract_trials(sessions, subject_number, config.sample_rate, config.trials)
    if not trials:
        raise ValueError(f"No selected trials for subject {subject_number}")
    trials = align_trials(trials, config.sample_rate, config.preprocessing)
    graphs = order_trials([build_trial_graph(trial, config) for trial in trials], config.trials.order)
    if not graphs:
        raise ValueError("Trial ordering produced no graphs; check trial counts and ordering policy")
    return graphs
