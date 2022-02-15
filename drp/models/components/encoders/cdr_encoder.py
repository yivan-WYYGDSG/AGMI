import torch
import torch.nn as nn
from drp.models.registry import COMPONENTS


@COMPONENTS.register_module()
class CDRConvEncoder(torch.nn.Module):
    def __init__(self, dropout_rate):
        super(CDRConvEncoder, self).__init__()

        # genomic mutation feature
        self.x_mut_conv1 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=(700, 1), bias=True)
        self.x_mut_pool1 = nn.MaxPool2d((5, 1))
        self.x_mut_conv2 = nn.Conv1d(in_channels=50, out_channels=30, kernel_size=(5, 1), bias=True)
        self.x_mut_pool2 = nn.MaxPool2d((10, 1))
        self.x_mut_fc1 = nn.LazyLinear(100)
        self.x_mut_dropout = nn.Dropout(dropout_rate)

        # gexp feature
        self.gexp_fc1 = nn.LazyLinear(256)
        self.gexp_bn = nn.BatchNorm1d(256)
        self.gexp_dropout = nn.Dropout(dropout_rate)
        self.gexp_fc2 = nn.Linear(256, 100)

        # methylation feature
        self.meth_fc1 = nn.LazyLinear(256)
        self.meth_bn = nn.BatchNorm1d(256)
        self.meth_dropout = nn.Dropout(dropout_rate)
        self.meth_fc2 = nn.Linear(256, 100)

        self.integration_fc1 = nn.Sequential(
            nn.Linear(300, 300),
            nn.Dropout(dropout_rate),
            nn.Tanh(),
        )

        self.integration_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(150, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(in_channels=30, out_channels=10, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(dropout_rate),
        )

        self.integration_fc2 = nn.LazyLinear(100)

    def forward(self, x_cell):
        x_gexpr, x_mut, x_methy = x_cell[:, :, 0], x_cell[:, :, 1], x_cell[:, :, 2]
        x_mut = x_mut.unsqueeze(1)
        x_mut = x_mut.unsqueeze(-1)
        x_mut = x_mut.float()
        x_mut = torch.tanh(self.x_mut_conv1(x_mut))
        x_mut = self.x_mut_pool1(x_mut)
        x_mut = torch.relu(self.x_mut_conv2(x_mut))
        x_mut = self.x_mut_pool2(x_mut)
        x_mut = x_mut.view(x_cell.shape[0], -1)
        x_mut = self.x_mut_fc1(x_mut)
        x_mut = self.x_mut_dropout(x_mut)
        x_gexpr = x_gexpr.float()
        x_gexpr = x_gexpr.view(x_cell.shape[0], -1)
        x_gexpr = torch.tanh(self.gexp_fc1(x_gexpr))
        x_gexpr = self.gexp_bn(x_gexpr)
        x_gexpr = self.gexp_dropout(x_gexpr)
        x_gexpr = torch.relu(self.gexp_fc2(x_gexpr))
        x_methy = x_methy.float()
        x_methy = x_methy.view(x_cell.shape[0], -1)
        x_methy = torch.tanh(self.meth_fc1(x_methy))
        x_methy = self.meth_bn(x_methy)
        x_methy = self.meth_dropout(x_methy)
        x_methy = torch.relu(self.meth_fc2(x_methy))

        x_cell = torch.cat((x_mut, x_gexpr, x_methy), 1)
        x_cell = self.integration_fc1(x_cell)
        x_cell = x_cell.unsqueeze(-1)
        x_cell = x_cell.unsqueeze(1)
        x_cell = self.integration_conv1(x_cell)
        x_cell = torch.dropout(x_cell.view(x_cell.shape[0], -1), 0.2, train=True)
        x_cell = self.integration_fc2(x_cell)

        return x_cell


if __name__ == '__main__':
    data = torch.rand((2, 18498, 3))
    net = CDRConvEncoder(dropout_rate=0.1)
    out = net(data)
