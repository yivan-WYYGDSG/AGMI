import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data
from drp.models.registry import COMPONENTS
import copy
from .EdgeGatedGraphEncoder import EdgeGatedGraphEncoder


@COMPONENTS.register_module()
class AGMIEncoder(EdgeGatedGraphEncoder):
    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 include_edges=None,
                 **kwargs):
        super(AGMIEncoder, self).__init__(out_channels=out_channels,
                                          num_layers=num_layers,
                                          bias=bias,
                                          aggr=aggr,
                                          include_edges=include_edges,
                                          **kwargs)

    def forward(self, x, edge_index=None, edge_weight=None):

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if self.cell_edge_attr is None:
            raise ValueError('cell_edge_attr is not init')

        if self.cell_edge_index is None:
            raise ValueError('cell_edge_index is not init')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # multiedge
            edge_m = self.propagate(self.cell_edge_index, x=m, edge_weight=self.cell_edge_attr[0])
            hidden_m = self.multi_edges_rnn(edge_m, m)

            for idx, w in enumerate(self.cell_edge_attr[1:]):
                if self.allow_edges[idx + 1] in self.include_edges:
                    # print('include {}'.format(self.allow_edges[idx+1]))
                    edge_m = self.propagate(self.cell_edge_index, x=m, edge_weight=w)
                    hidden_m = self.multi_edges_rnn(edge_m, hidden_m)
            x = self.rnn(hidden_m, x)

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={})'.format(self.__class__.__name__,
                                              self.out_channels,
                                              self.num_layers)
