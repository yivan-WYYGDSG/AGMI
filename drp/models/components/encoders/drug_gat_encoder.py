import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from drp.models.registry import COMPONENTS,BACKBONES


@COMPONENTS.register_module()
class DrugGATEncoder(torch.nn.Module):
    def __init__(self, num_features_xd=78, heads=10, output_dim=128, gat_dropout=0.2):
        super(DrugGATEncoder, self).__init__()

        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=heads, dropout=gat_dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=gat_dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        return x
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
