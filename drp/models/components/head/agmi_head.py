import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool,
    Set2Set,
)
from drp.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AGMIFusionHead(nn.Module):
    def __init__(self, out_channels):
        super(AGMIFusionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2, int(out_channels * 1.5)),
            nn.ReLU(),
            nn.Linear(int(out_channels * 1.5), out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
        )
        

    def forward(self, x_cell_embed, drug_embed):
        out = torch.cat((x_cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)