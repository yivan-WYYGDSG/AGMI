from drp.models.registry import COMPONENTS, BACKBONES
import torch.nn as nn
import torch


@COMPONENTS.register_module()
class TcnnFusionHead(nn.Module):
    def __init__(self, out_channels=1, dropout=0.2):
        super(TcnnFusionHead, self).__init__()
        self.fusion_layers = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_channels)
        )

    def forward(self, drug_embed, cell_embed):
        out = torch.cat((cell_embed, drug_embed), dim=2)
        if len(out.shape) == 3:
            out = out.view(-1, out.shape[1] * out.shape[2])
        out = self.fusion_layers(out)
        return out
