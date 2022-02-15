from drp.models.registry import COMPONENTS, BACKBONES
import torch.nn as nn
import torch


@COMPONENTS.register_module()
class Conv1dNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, pooling_size=6, dropout=0.2):
        super(Conv1dNeck, self).__init__()
        self.neckLayer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pooling_size),
            nn.Conv1d(in_channels=in_channels*2, out_channels=in_channels*4, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pooling_size),
            nn.Conv1d(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size),
        )
        self.fc = nn.Sequential(
            nn.Linear(492, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.neckLayer(x)
        x = x.view(-1, x.shape[1] * x.shape[2])
        # print(x.shape)
        return self.fc(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
