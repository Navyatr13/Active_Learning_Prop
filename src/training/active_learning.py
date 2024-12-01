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

    # Generate pseudo-labels for individual data samples using the pretrained model.

    pretrained_model.eval()
    pseudo_labels = []
    
    for batch in tqdm(loader, desc="Generating pseudo-labels"):
        batch = batch.to(device)
        with torch.no_grad():
            # Predict for the batch
            preds = pretrained_model(batch.z, batch.pos, batch.batch).view(-1, 1)  # Ensure proper shape
            
            # Append predictions, ensuring each has at least one dimension
            pseudo_labels.extend([pred.unsqueeze(0).cpu() for pred in preds])
    
    # Concatenate all predictions into a single tensor
    return pseudo_labels #torch.cat(pseudo_labels, dim=0)


def estimate_uncertainty(model, loader, device, num_samples=10):
    model.train()  # Enable dropout even during inference
    uncertainty_sample_pairs = []

    for batch in tqdm(loader, desc="Estimating uncertainties"):
        batch = batch.to(device)
        preds = []
        for _ in range(num_samples):
            with torch.no_grad():
                preds.append(model(batch.x, batch.edge_index, batch.batch).view(-1).cpu())
        preds = torch.stack(preds, dim=0)
        uncertainties = preds.var(dim=0)  # Variance across predictions

        # Include the index of each data point
        for i, (data_point, uncertainty) in enumerate(zip(batch.to_data_list(), uncertainties)):
            uncertainty_sample_pairs.append((uncertainty.item(), data_point, i))  # Add the index here

    return uncertainty_sample_pairs


def active_learning_loop(custom_model, pretrained_model, dataset, batch_size, num_cycles, initial_size, device):
    # Split dataset
    labeled_set, unlabeled_set, test_set = split_dataset(dataset, initial_size,test_set_size = 0.3)
    labeled_set = list(labeled_set)  # Convert to list for dynamic modification
    unlabeled_set = list(unlabeled_set)

    # Create CSV for logging
    with open("active_learning_metrics.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Cycle", "Labeled Dataset Size", "Test MAE"])

    for cycle in range(num_cycles):
        print(f"\n--- Active Learning Cycle {cycle + 1}/{num_cycles} ---")

        # Train the Custom Model
        labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle= False)
        trainer = Trainer(max_epochs=1, accelerator="auto", devices=1)
        trainer.fit(custom_model, labeled_loader)

        # Evaluate the Model
        test_loader = DataLoader(test_set, batch_size=batch_size)
        test_results = trainer.test(custom_model, dataloaders=test_loader)
        test_mae = test_results[0]["test_mae"]
        print(f"Cycle {cycle + 1}: Test MAE: {test_mae:.4f}")

        # Estimate Uncertainties
        uncertainty_sample_pairs = estimate_uncertainty(custom_model, DataLoader(unlabeled_set, batch_size=batch_size), device)
        
        # Sort and Select
        uncertainty_sample_pairs.sort(key=lambda x: x[0], reverse=True)
        most_uncertain_samples = [pair[1] for pair in uncertainty_sample_pairs[:batch_size]]

        selected_indices = [pair[2] for pair in uncertainty_sample_pairs[:batch_size]]
    
        # Pseudo-Labeling
        pseudo_labels = label_with_pretrained(pretrained_model, DataLoader(most_uncertain_samples, batch_size=batch_size), device)
        
        for sample, pseudo_label in zip(most_uncertain_samples, pseudo_labels):
            sample.y[:, 9] = pseudo_label

        # Update Datasets
        labeled_set.extend(most_uncertain_samples)
        unlabeled_set = [sample for i, sample in enumerate(unlabeled_set) if i not in selected_indices]


        # Step 9: Log Metrics
        with open("active_learning_metrics.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([cycle + 1, len(labeled_set), test_mae])

    print("\nActive Learning Completed. Metrics logged to 'active_learning_metrics.csv'.")



