import torch
from src.models.gnn import CustomGNN
from torch.utils.data import random_split

def load_custom_model(input_dim, hidden_dim, output_dim, checkpoint_path, device):
   
    #  model (torch.nn.Module): Initialized or pre-trained custom GNN model.
    
    # Initialize the model
    model = CustomGNN(input_dim, hidden_dim, output_dim)
    
    # Load model weights if a checkpoint path is provided
    if checkpoint_path:
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to the specified device
    if device:
        model = model.to(device)

    return model

def get_data_split(dataset, train_size, val_size, test_size):
  
  # Define split lengths
  total_size = len(dataset)
  trn_size = int(train_size * total_size)
  vl_size = int(val_size * total_size)
  tst_size = total_size - trn_size - vl_size
  
  # Randomly split the dataset
  train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
  return train_dataset, val_dataset, test_dataset
  

def split_dataset(dataset, initial_size):
    
    # Split the dataset into an initial labeled set and an unlabeled set.
    
    total_size = len(dataset)
    labeled_size = initial_size
    unlabeled_size = total_size - labeled_size

    labeled_set, unlabeled_set = random_split(dataset, [labeled_size, unlabeled_size])

    return labeled_set, unlabeled_set
