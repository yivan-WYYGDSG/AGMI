import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from drp.models.registry import COMPONENTS


@COMPONENTS.register_module()
class NaiveGenesEncoder(torch.nn.Module):
    def __init__(self, n_filters=32, output_dim=128):
        super(NaiveGenesEncoder, self).__init__()

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=3, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.LazyLinear(output_dim)

    def forward(self, x_cell):
        conv_xt = self.conv_xt_1(x_cell)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)

        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)

        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        return xt
