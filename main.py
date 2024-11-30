import os
import sys
import warnings
import argparse
from pathlib import Path
import multiprocessing

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer

# Import project-specific modules
from entalpic_al import HOME, TARGET
from src.data.dataset import load_qm9_data
from src.models.pretrained import load_pretrained_model
from src.training.active_learning import active_learning_loop
from src.utils.helper import load_custom_model, get_data_split

# Suppress warnings and unnecessary logs
warnings.filterwarnings("ignore") 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

# Define constants
TARGET = 9
HOME = Path(__file__).resolve().parent.parent

def active_learning_main(batch_size, num_cycles):
    """
    Active learning pipeline using a pretrained model and custom GNN.
    """
    # Load the QM9 dataset
    dataset = load_qm9_data()
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.num_node_features
    hidden_dim = 128
    output_dim = 1
    checkpoint_path = None

    # Initialize models
    custom_model = load_custom_model(input_dim, hidden_dim, output_dim, checkpoint_path, device)
    pretrained_model, _ = load_pretrained_model(dataset, TARGET, HOME, device)

    # Run the active learning loop
    active_learning_loop(
        custom_model=custom_model,
        pretrained_model=pretrained_model,
        dataset=dataset,
        batch_size=batch_size,
        num_cycles=num_cycles,
        initial_size=1000,
        device=device
    )

def main(batch_size, num_batches):
    """
    Standard training pipeline for the custom GNN.
    """
    print(f"Using batch size: {batch_size}")

    # Load the QM9 dataset
    dataset = load_qm9_data()
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.num_node_features
    hidden_dim = 128
    output_dim = 1
    checkpoint_path = None

    # Initialize the custom model
    model = load_custom_model(input_dim, hidden_dim, output_dim, checkpoint_path, device)

    # Split dataset
    train_dataset, val_dataset, test_dataset = get_data_split(dataset, train_size=0.7, val_size=0.2, test_size=0.1)

    # Prepare dataloaders
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # Train and evaluate
    trainer = Trainer(max_epochs=10, accelerator="auto", devices=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument("--num_batches", type=int, default=32, help="Number of batches for evaluation")
    parser.add_argument("--active_learning", action="store_true", help="Run active learning loop instead of standard training")
    parser.add_argument("--num_cycles", type=int, default=5, help="Number of active learning cycles")
    args = parser.parse_args()

    if args.active_learning:
        print("Starting Active Learning...")
        active_learning_main(args.batch_size, args.num_cycles)
    else:
        print("Starting Standard Training...")
        main(args.batch_size, args.num_batches)
