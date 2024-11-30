import torch
from torch_geometric.datasets import QM9
from pathlib import Path

TARGET = 9
HOME = Path(__file__).resolve().parent.parent


def load_qm9_data():
    # Load and preprocess the QM9 dataset.
    dataset = QM9(HOME)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]
    return dataset
