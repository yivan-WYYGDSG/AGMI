import torch
import torch.nn as nn
import torch.nn.functional as F
from drp.models.registry import COMPONENTS


@COMPONENTS.register_module()
class TcnnConvEncoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(TcnnConvEncoder, self).__init__()

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=in_channels, out_channels=40, kernel_size=7, bias=True, padding=3)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=7, bias=True, padding=3)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=80, out_channels=60, kernel_size=7, bias=True, padding=3)
        self.pool_xt_3 = nn.MaxPool1d(3)

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

        return conv_xt
