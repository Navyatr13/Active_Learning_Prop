# This script is supposed to work.
# There is no catch beyond code quality.
# With default args (32, 32) the inference should take ~1min on CPU.
# If this script does not work, get back to Entalpic.
# You should delete these comments in your refactor.
import argparse

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNetPlusPlus
from tqdm import tqdm

from entalpic_al import HOME, TARGET


def main(batch_size, num_batches):
    print(f"Using batch size: {batch_size}")

    # Load the QM9 dataset
    dataset = QM9(HOME)
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained model
    model, datasets = DimeNetPlusPlus.from_qm9_pretrained(HOME, dataset, TARGET)
    model = model.to(device)

    # Prepare the test dataset and dataloader
    _, _, test_dataset = datasets
    loader = DataLoader(test_dataset, batch_size=batch_size)

    # Evaluate the pretrained model
    mae_pretrained = evaluate_model(model, loader, num_batches, device)

    # Reset parameters and evaluate again
    model.reset_parameters()
    mae_random = evaluate_model(model, loader, num_batches, device)

    print(f"Pretrained MAE: {mae_pretrained:.4f} eV")
    print(f"Random init MAE: {mae_random:.4f} eV")

def evaluate_model(model, loader, num_batches, device):
    mae_values = []
    for batch_idx, data in enumerate(tqdm(loader, total=num_batches)):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, TARGET]).abs()
        mae_values.append(mae)

        if batch_idx == num_batches - 1:
            break

    mae_values = torch.cat(mae_values, dim=0)
    return mae_values.mean().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the inference loop")
    parser.add_argument("--num_batches", type=int, default=32, help="Number of batches to test on")
    args = parser.parse_args()
    main(args.batch_size, args.num_batches)