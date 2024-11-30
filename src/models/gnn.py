import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
from torchmetrics import MeanAbsoluteError


class CustomGNN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3, weight_decay = 1e-4):
        super(CustomGNN, self).__init__()
        # Save hyperparameters 
        self.save_hyperparameters()

        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Linear layers
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, output_dim)

        # Activation function
        self.relu = ReLU()
        
        # Loss function
        self.criterion = MeanAbsoluteError() #torch.nn.L1Loss() 
        
        #optimizer
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, edge_index, batch):
        # GCN layers
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def training_step(self, batch, batch_idx):
        # Forward pass
        pred = self(batch.x, batch.edge_index, batch.batch).view(-1)
        target = batch.y[:, self.hparams.output_dim]
        loss = self.criterion(pred, target)
        batch_size = batch.batch.max().item() + 1
        # Logging the training loss
        self.log("train_loss", loss, on_step=True, batch_size=batch_size, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        pred = self(batch.x, batch.edge_index, batch.batch).view(-1)
        target = batch.y[:, self.hparams.output_dim]
        loss = self.criterion(pred, target)
        batch_size = batch.batch.max().item() + 1
        
        # Logging the validation loss
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        pred = self(batch.x, batch.edge_index, batch.batch).view(-1)
        target = batch.y[:, self.hparams.output_dim]
        mae = (pred - target).abs().mean()
        batch_size = batch.batch.max().item() + 1

        # Logging the test MAE
        self.log("test_mae", mae, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return mae

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        
