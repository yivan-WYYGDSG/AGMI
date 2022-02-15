from drp.models.registry import COMPONENTS
import torch.nn as nn
import torch


@COMPONENTS.register_module()
class BaseFusionHead(nn.Module):
    def __init__(self, d_in_channels, g_in_channels, out_channels=1, reduction=2, dropout=0.2):
        super(BaseFusionHead, self).__init__()
        in_channels = d_in_channels + g_in_channels
        self.fusion_layers = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // reduction, in_channels // reduction // reduction),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // reduction // reduction, out_channels)
        )

    def forward(self, drug_embed, cell_embed):
        if len(drug_embed.shape) == 3:
            drug_embed = drug_embed.view(cell_embed.shape[0], -1)
        out = torch.cat((cell_embed, drug_embed), dim=1)
        out = self.fusion_layers(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)