import torch
import csv

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.utils.helper import split_dataset
from pytorch_lightning import Trainer

def get_initial_data(dataset, initial_size=1000):
    
    # Split the dataset into an initial labeled set and an unlabeled set.
    initial_set, unlabeled_set = random_split(dataset, [initial_size, len(dataset) - initial_size])
    return initial_set, unlabeled_set


def label_with_pretrained(pretrained_model, loader, device):

    # Generate pseudo-labels using the pretrained model.

    pretrained_model.eval()
    pseudo_labels = []
    for batch in tqdm(loader, desc="Generating pseudo-labels"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = pretrained_model(batch.z, batch.pos, batch.batch).view(-1)
        pseudo_labels.append(pred.cpu())
    return torch.cat(pseudo_labels, dim=0)


def estimate_uncertainty(model, loader, device, num_samples=10):

    # Estimate uncertainty using MC Dropout.

    model.train()  # Enable dropout even during inference
    uncertainties = []
    all_samples = []
    for batch in tqdm(loader, desc="Estimating uncertainties"):
        batch = batch.to(device)
        preds = []
        for _ in range(num_samples):
            with torch.no_grad():
                preds.append(model(batch.x, batch.edge_index, batch.batch).view(-1).cpu())
        preds = torch.stack(preds, dim=0)
        uncertainty = preds.var(dim=0)  # Variance across predictions
        uncertainties.append(uncertainty)
        all_samples.append(batch)
    return torch.cat(uncertainties, dim=0), all_samples

def active_learning_loop(custom_model, pretrained_model, dataset, batch_size, num_cycles, initial_size, device):

    # Active Learning loop to iteratively train the model and expand the labeled dataset.
    

    # Split dataset into initial labeled and unlabeled sets
    labeled_set, unlabeled_set = split_dataset(dataset, initial_size)

    # Convert subsets to lists for dynamic modification
    labeled_set = list(labeled_set)
    unlabeled_set = list(unlabeled_set)

    # Create CSV to log results
    with open("active_learning_metrics.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Cycle", "Labeled Dataset Size", "Test MAE"])

    for cycle in range(num_cycles):
        print(f"\n--- Active Learning Cycle {cycle + 1}/{num_cycles} ---")
        
        # Prepare labeled set DataLoader
        labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)

        # Train custom model
        trainer = Trainer(max_epochs=10, accelerator="auto", devices=1)
        trainer.fit(custom_model, labeled_loader)

        # Evaluate the model on the test dataset
        test_loader = DataLoader(dataset, batch_size=batch_size)
        test_results = trainer.test(custom_model, dataloaders=test_loader)
        test_mae = test_results[0]["test_mae"]
        print(f"Cycle {cycle + 1}: Test MAE: {test_mae:.4f}")

        # Estimate uncertainties and acquire new labels
        uncertainties, all_samples = estimate_uncertainty(custom_model, DataLoader(unlabeled_set, batch_size=batch_size), device)
        num_to_label = min(batch_size, len(uncertainties))
        uncertain_indices = torch.topk(uncertainties, num_to_label).indices.tolist()
        selected_samples = [unlabeled_set[i] for i in uncertain_indices]

        # Label selected samples with pretrained model
        pseudo_labels = label_with_pretrained(pretrained_model, DataLoader(selected_samples, batch_size=batch_size), device)

        # Add newly labeled samples to the labeled set
        labeled_set.extend(selected_samples)

        # Remove selected samples from the unlabeled set
        unlabeled_set = [sample for i, sample in enumerate(unlabeled_set) if i not in uncertain_indices]

        # Log metrics
        with open("active_learning_metrics.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([cycle + 1, len(labeled_set), test_mae])

    print("\nActive Learning Completed. Metrics logged to 'active_learning_metrics.csv'.")
    
    
    
def acquire_new_labels(pretrained_model, unlabeled_set, batch_size, device):
    #  Selects the most uncertain samples and labels them using the pretrained model.
    #  list  Selected samples with pseudo-labels is returned
    
    # Create DataLoader for the unlabeled set
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)

    # Pseudo-labeling
    pseudo_labels = label_with_pretrained(pretrained_model, unlabeled_loader, device)

    # Select samples based on uncertainty
    uncertainties, _ = estimate_uncertainty(pretrained_model, unlabeled_loader, device)
    num_to_label = min(batch_size, len(uncertainties))
    uncertain_indices = torch.topk(uncertainties, num_to_label).indices.tolist()

    # Extract the selected samples
    selected_samples = [unlabeled_set[i] for i in uncertain_indices]

    return selected_samples, pseudo_labels
    

