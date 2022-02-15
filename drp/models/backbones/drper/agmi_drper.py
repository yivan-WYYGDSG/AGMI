import torch.nn as nn
from drp.models.registry import BACKBONES
from drp.models.builder import build_component
import torch

class NativeAttention(torch.nn.Module):
    def __init__(self, net_i, net_j):
        super(NativeAttention, self).__init__()
        self.net_i = net_i
        self.net_j = net_j

    def forward(self, h_i, h_j):
        res_i = torch.cat([h_i, h_j], dim=1)
        res_j = self.net_j(h_j)
        return torch.nn.Softmax(dim=1)(self.net_i(res_i)) * res_j

class AttnPropagation(torch.nn.Module):
    def __init__(self, gate_layer, feat_size=19, gather_width=64):
        super(AttnPropagation, self).__init__()

        self.gather_width = gather_width
        self.feat_size = feat_size

        self.gate = gate_layer

        net_i = nn.Sequential(
            nn.Linear(self.feat_size * 2, self.feat_size),
            nn.Softsign(),
            nn.Linear(self.feat_size, self.gather_width),
            nn.Softsign(),
        )
        net_j = nn.Sequential(
            nn.Linear(self.feat_size, self.gather_width), nn.Softsign()
        )
        self.attention = NativeAttention(net_i, net_j)

    def forward(self, data, index, is_eval=False):
        # propagtion
        h_0 = data
        h_1 = self.gate(h_0, index, is_eval=is_eval)
        h_1 = self.attention(h_1, h_0)

        return h_1
    
    def init_weights(self):
        self.gate.init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
    
    def update_buffer(self, edge_attr, edge_index):
        self.gate.update_buffer(edge_attr, edge_index)
        

@BACKBONES.register_module()
class AGMIDRPer(torch.nn.Module):

    def __init__(self,
                 in_channel,
                 drug_encoder,
                 genes_encoder,
                 neck,
                 head,
                 attn_head=None,
                 gather_width=32):
        super().__init__()

        # body
        self.drug_encoder = build_component(drug_encoder)
        self.genes_encoder = AttnPropagation(feat_size=in_channel, gather_width=gather_width, gate_layer=build_component(genes_encoder))
        self.head = build_component(head)
        self.neck = build_component(neck)
        if attn_head is not None:
            self.attn_head = build_component(attn_head)
        else:
            self.attn_head = None
        

    def forward(self, data):
        """Forward function.

        Args:
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
            :param test_mode:
            :param gt:
            :param data:
        """
        x_cell = data.x_cell
        _, channels = x_cell.shape
        x_d, x_d_edge_index, x_d_batch = \
            data.x, data.edge_index, data.batch
        batch_size = x_d_batch.max().item() + 1
        # print(batch_size)
        
        x_cell_embed = self.genes_encoder(
            x_cell, None
        )
        x_cell_embed = x_cell_embed.view(batch_size, 18498, -1)
        x_cell_embed = torch.transpose(x_cell_embed, 1, 2)
        x_cell_embed = self.neck(x_cell_embed)
        # if self.attn_head is not None:
        #     x_cell_embed, A = self.attn_head(x_cell_embed)
        #     A = torch.mean(A, dim=0)
        # else:
        # A = None
        drug_embed = self.drug_encoder(x_d, x_d_edge_index, x_d_batch)
        output = self.head(x_cell_embed, drug_embed)
        return output



    def init_weights(self):
        self.drug_encoder.init_weights()
        self.genes_encoder.init_weights()
        self.neck.init_weights()
        self.head.init_weights()
        if self.attn_head is not None:
            self.attn_head.init_weights()
