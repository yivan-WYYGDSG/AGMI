import torch
from .base_InMemory_dataset import BaseInMemoryDataset
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from .registry import DATASETS
from drp.core import mse, rmse, r2, pearson, spearman, mae


@DATASETS.register_module()
class InMemoryMultiEdgeGraphGenesDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_items,
                 celllines_data,
                 num_genes_nodes,
                 metrics,
                 drug_graphs=None,
                 root='./work_dir/genes_drug_data',
                 name='MutiEdgeGraphGenes',
                 transform=None,
                 pre_transform=None):

        super(InMemoryMultiEdgeGraphGenesDataset, self).__init__(root, transform, pre_transform)
        self.dataset = name
        self.data_items = np.load(data_items, allow_pickle=True)
        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        self.graphs = np.load(drug_graphs, allow_pickle=True).item()
        self.num_genes_nodes = num_genes_nodes
        self.metrics = metrics
        self.diseases = []
        self.allowed_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                                'R2': r2, 'PEARSON': pearson, 'SPEARMAN': spearman}
        self.omic_data = {'expr': 0, 'mut': 1, 'cn': 2, 'dna':2}
        self.include_omic = ['expr', 'mut', 'cn']
            

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        data_list = []
        data_len = len(self.data_items)
        for i in tqdm(range(data_len)):
            disease, drug, c1, target = self.data_items[i]
            d_size, features, edge_index = self.graphs[drug]

            self.diseases.append(disease)
            x1_data = torch.Tensor()
            if len(self.include_omic) == 3:
                x1_data = self.celllines[c1]
            else:
                for key in self.include_omic:
                    x1_data = torch.cat((x1_data, self.celllines[c1].x[:, self.omic_data[key]]), 0)
                x1_data = x1_data.view(-1, len(self.include_omic))
            drug_cell_data = Data(x=torch.Tensor(features),
                                  edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                  y=torch.FloatTensor([target[0]]))

            drug_cell_data.x_cell = x1_data
            drug_cell_data.AUC = torch.FloatTensor([target[1]])
            drug_cell_data.disease_idx = disease

            drug_cell_data.__setitem__('d_size', torch.LongTensor([d_size]))

            data_list.append(drug_cell_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        self.diseases = list(set(self.diseases))

        print('Dataset construction done.')

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        pred = torch.Tensor()
        y = torch.Tensor()
        for res in results:
            pred = torch.cat((pred, res['output']), 0)
            y = torch.cat((y, res['labels']), 0)

        y = y.numpy().flatten()

        eval_results = dict()
        for metric in self.metrics:
            eval_results[metric] = self.allowed_metrics[metric](pred.numpy().flatten(), y)
        eval_results['output'] = pred.numpy()
        eval_results['labels'] = y

        return eval_results


