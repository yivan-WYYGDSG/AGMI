import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_max_pool as gmp
from drp.models.registry import COMPONENTS, BACKBONES


# drug graph branch

@COMPONENTS.register_module()
class DrugGCNncoder(torch.nn.Module):
    def __init__(self, drug_features=78, output_dim=128, dropout_rate=0.2):
        super(DrugGCNncoder, self).__init__()
        self.conv1_d = GCNConv(drug_features, 300)
        self.conv2_d = GCNConv(300, 300)
        self.drug_embed = nn.Sequential(
            nn.Linear(300, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, edge_index, batch):
        x_d = F.relu(self.conv1_d(x, edge_index))
        x_d = F.relu(self.conv2_d(x_d, edge_index))
        x_d = gmp(x_d, batch)
        x_d = self.drug_embed(x_d)
        return x_d
