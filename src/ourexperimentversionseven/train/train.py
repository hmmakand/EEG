"""Reproduce the BCI Competition IV 2a GAT experiment (per-subject, 10-fold CV).

Ported as-is from src/ourexperimentversionseven/kaggleeeg.ipynb (cell 2793b17a),
the pipeline that produced BCI_IV_2a_GAT_Results.json. All hyperparameters are
hardcoded to match the notebook -- this script is meant to reproduce that run,
not to be a general-purpose trainer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ourexperimentversionseven.dataset.dataset import (
    BANDS,
    BAND_FILTER,
    FS,
    aggregate_eeg_data,
    bandpass,
    bandpass1,
    bandpowercalc,
    compute_plv,
    create_graphs,
    load_bci_competition_data,
    resolve_data_dir,
)
from src.ourexperimentversionseven.model.gat import GAT

SUBJECT_NUMBERS = [1, 2, 3, 5, 6, 7, 8, 9]
THRESHOLD = 0
N_FOLDS = 10
SEED = 42
TORCH_MANUAL_SEED = 12345
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 250
HIDDEN_CHANNELS = 22
HEADS = 3
RESULTS_PATH = Path(__file__).resolve().parents[1] / "output/BCI_IV_2a_GAT_Results.json"


def run_subject(data_dir: str, subject_number: int, device: torch.device) -> dict:
    S1 = load_bci_competition_data(data_dir, subject_number)

    idx = ["L", "R"]
    for key in idx:
        for i in range(S1[key].shape[1]):
            S1[key][0, i] = bandpass(S1[key][0, i], BAND_FILTER, FS)

    plv, y = compute_plv(S1)
    graphs = create_graphs(plv, THRESHOLD)
    numElectrodes = S1["L"][0, 0].shape[1]

    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)

    edge_indices = []
    for i in range(adj.shape[2]):
        source_nodes, target_nodes = [], []
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                if adj[row, col, i] >= THRESHOLD:
                    source_nodes.append(row)
                    target_nodes.append(col)
                else:
                    source_nodes.append(0)
                    target_nodes.append(0)
        edge_indices.append(torch.tensor([source_nodes, target_nodes], dtype=torch.long))
    edge_indices = torch.stack(edge_indices, dim=-1)

    l, r = aggregate_eeg_data(S1, BANDS)
    l, r = np.transpose(l, [1, 0, 2, 3]), np.transpose(r, [1, 0, 2, 3])

    for i in range(l.shape[3]):
        bp = [BANDS[i], BANDS[i + 1]]
        for j in range(l.shape[2]):
            l[:, :, j, i] = bandpass1(l[:, :, j, i], bp, sample_rate=FS)
            r[:, :, j, i] = bandpass1(r[:, :, j, i], bp, sample_rate=FS)

    l = bandpowercalc(l, BANDS, FS)
    r = bandpowercalc(r, BANDS, FS)

    x = np.concatenate([l, r], axis=2)
    x = torch.tensor(x, dtype=torch.float32)

    data_list = []
    for i in range(np.size(adj, 2)):
        data_list.append(Data(x=x[:, :, i], edge_index=edge_indices[:, :, i], y=y[i, 0]))

    size = len(data_list)
    idx_split = size // 2
    datal = data_list[:idx_split]
    datar = data_list[idx_split:]

    data_list = []
    for i in range(idx_split):
        data_list.extend([datal[i], datar[i]])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    highest_test_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list)):
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        torch.manual_seed(TORCH_MANUAL_SEED)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        model = GAT(hidden_channels=HIDDEN_CHANNELS, heads=HEADS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()

        def train_epoch():
            model.train()
            for data in train_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def evaluate(loader):
            model.eval()
            correct = 0
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())
            return correct / len(loader.dataset)

        optimal = [0, 0, 0]
        for _ in tqdm(
            range(1, EPOCHS),
            desc=f"Training Subject {subject_number} Fold {fold + 1}",
            leave=False,
        ):
            train_epoch()
            train_acc = evaluate(train_loader)
            test_acc = evaluate(test_loader)
            av_acc = np.mean([train_acc, test_acc])
            if test_acc > optimal[2]:
                optimal = [av_acc, train_acc, test_acc]

        highest_test_accuracies.append(optimal[2])

    return {
        "mean": float(np.mean(highest_test_accuracies)),
        "max": float(np.max(highest_test_accuracies)),
        "min": float(np.min(highest_test_accuracies)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = resolve_data_dir()
    subject_results = {}

    for subject_number in tqdm(SUBJECT_NUMBERS, desc="Processing Subjects"):
        results = run_subject(data_dir, subject_number, device)
        subject_results[subject_number] = results
        print(f"S{subject_number}: Mean: {results['mean']:.4f}, Max: {results['max']:.4f}, Min: {results['min']:.4f}")

    print("\nSummary of Results for All Subjects:")
    for subject_number, results in subject_results.items():
        print(f"S{subject_number}: Mean: {results['mean']:.4f}, Max: {results['max']:.4f}, Min: {results['min']:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as json_file:
        json.dump(subject_results, json_file, indent=4)

    print(f"\nResults saved to '{RESULTS_PATH}'")


if __name__ == "__main__":
    main()
