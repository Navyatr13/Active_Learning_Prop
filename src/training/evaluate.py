import torch
from tqdm import tqdm

def evaluate_model(model, loader, num_batches, target, device):
    # Evaluate the model and compute MAE.
    mae_values = []
    for batch_idx, data in enumerate(tqdm(loader, total=num_batches)):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.z, data.pos, data.batch)
        mae = (pred.view(-1) - data.y[:, target]).abs()
        mae_values.append(mae)

        if batch_idx == num_batches - 1:
            break

    mae_values = torch.cat(mae_values, dim=0)
    return mae_values.mean().item()
