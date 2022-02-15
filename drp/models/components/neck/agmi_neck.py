import torch
import torch.nn as nn
from torch_geometric.nn import (
    global_add_pool,
    Set2Set,
)
from drp.models.registry import COMPONENTS


@COMPONENTS.register_module()
class AGMICellNeck(nn.Module):
    def __init__(self, in_channels=[6,8,16], out_channels=[8,16,32], kernel_size=[16,16,16], drop_rate=0.2, max_pool_size=[3,6,6], feat_dim=128):
        super(AGMICellNeck, self).__init__()
        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(5376, feat_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, x_cell_embed):
        x_cell_embed = self.cell_conv(x_cell_embed)

        x_cell_embed = x_cell_embed.view(-1, x_cell_embed.shape[1] * x_cell_embed.shape[2])
        
        # print(x_cell_embed.shape)

        x_cell_embed = self.fc(x_cell_embed)
        
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)