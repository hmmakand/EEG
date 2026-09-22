"""Cross-validation partitions and graph batch loaders."""
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from .config import TrainingConfig


def make_fold_indices(data_list, config: TrainingConfig):
    if len(data_list) < config.n_folds:
        raise ValueError(f"Need at least {config.n_folds} graphs; received {len(data_list)}")
    splitter = KFold(n_splits=config.n_folds, shuffle=config.split_shuffle,
                     random_state=config.split_seed if config.split_shuffle else None)
    return splitter.split(data_list)


def select_fold_data(data_list, train_indices, test_indices):
    return ([data_list[index] for index in train_indices],
            [data_list[index] for index in test_indices])


def build_loaders(train_data, test_data, config: TrainingConfig):
    if not train_data or not test_data:
        raise ValueError("Both training and test folds must be nonempty")
    return (DataLoader(train_data, batch_size=config.batch_size, shuffle=config.train_shuffle),
            DataLoader(test_data, batch_size=config.batch_size, shuffle=config.test_shuffle))
