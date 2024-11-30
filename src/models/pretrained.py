from torch_geometric.nn import DimeNetPlusPlus

def load_pretrained_model(dataset, target, home, device):
    # Load the pretrained model.
    model, datasets = DimeNetPlusPlus.from_qm9_pretrained(home, dataset, target)
    model = model.to(device)
    return model, datasets
