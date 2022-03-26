import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from drp.models.registry import COMPONENTS,BACKBONES
from torch_geometric.nn import GATConv, GatedGraphConv
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor

@COMPONENTS.register_module()
class MultiEdgeGatedGraphConv(GatedGraphConv):
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, num_edges: int = None, include_edges=None,
                 **kwargs):
        super(MultiEdgeGatedGraphConv, self).__init__(out_channels=out_channels, num_layers=num_layers, bias=bias,
                                                     aggr=aggr, **kwargs)

        self.register_buffer('cell_edge_attr', None)
        self.register_buffer('cell_edge_attr_test', None)
        self.register_buffer('cell_edge_index', None)
        self.register_buffer('cell_edge_index_test', None)

        self.multi_edges_rnn = torch.nn.GRUCell(out_channels * num_edges, out_channels, bias=bias)
        
        if include_edges is None:
            include_edges = ['ppi', 'gsea', 'pcc']
        self.include_edges = include_edges
        self.allow_edges = ['ppi', 'gsea', 'pcc']
        
    def update_buffer(self, edge_attr, edge_index):
        print('update buffer')
        assert edge_attr.shape[-1] == edge_index.shape[-1], 'edge_attribution and edge_index are not compatible'
        device = next(self.parameters()).device
        self.cell_edge_attr = edge_attr.to(device)
        self.cell_edge_index = edge_index.to(device)
        
    
    def init_weights(self):
        self.multi_edges_rnn.reset_parameters()
    
    

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, is_eval=False) -> Tensor:

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        weights = self.cell_edge_attr
        edges = self.cell_edge_index

        for i in range(self.num_layers):
            x_cur = torch.matmul(x, self.weight[i])
            if self.multi_edges_rnn is not None:
                m = self.propagate(edges, x=x_cur, edge_weight=weights[0])
                for w in weights[1:]:
                    m = torch.cat((m, self.propagate(edges, x=x_cur, edge_weight=w)), dim=1)
                x = self.multi_edges_rnn(m, x)
            else:
                # this method require num_layers equals len(weights)
                m = self.propagate(edges, x=x_cur, edge_weight=weights[i], size=None)
                x = self.rnn(m, x)
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
