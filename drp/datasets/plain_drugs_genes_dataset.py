from collections import defaultdict

import torch
from .base_InMemory_dataset import BaseInMemoryDataset
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from .registry import DATASETS
from torch.utils.data import Dataset
from drp.core import mse, rmse, r2, pearson, spearman, mae


@DATASETS.register_module()
class DrugsGenesDataset(Dataset):
    def __init__(self,
                 data_items,
                 celllines_data,
                 name,
                 metrics,
                 include_omic,
                 drug_graphs=None):
        """

        :param data_items:
        :param celllines_data:

        """
        super(DrugsGenesDataset, self).__init__()
        self.dataset = name
        self.data_items = np.load(data_items, allow_pickle=True)
        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        self.graphs = np.load(drug_graphs, allow_pickle=True).item()
        self.data_list = []
        self.metrics = metrics
        self.include_omic = include_omic
        self.allowed_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                                'R2': r2, 'PEARSON': pearson, 'SPEARMAN': spearman}
        self.omic_data = {'expr':0, 'mut':1, 'R':2}
        self.process()


    def __getitem__(self, index):
        sample = self.data_list[index]
        return sample

    def __len__(self):
        return len(self.data_list)

    def process(self):
        max_atoms = 0
        data_len = len(self.data_items)
        for i in tqdm(range(data_len)):
            _, drug, c1, target = self.data_items[i]
            _, features, _ = self.graphs[drug]
            x1_data = torch.Tensor()
            for key in self.include_omic:
                x1_data = torch.cat((x1_data, self.celllines[c1].x[:, self.omic_data[key]]), 0)
            x1_data = x1_data.view(-1, len(self.include_omic))
            drug = torch.Tensor(features)
            max_atoms = max(max_atoms, drug.shape[0])
            self.data_list.append(dict(cell=x1_data, drug=drug, labels=target[0]))
        for idx, sample in enumerate(self.data_list):
            drug = sample['drug']
            new_drug_data = torch.zeros((max_atoms, drug.shape[1]))
            new_drug_data[:drug.shape[0], :] = drug
            self.data_list[idx]['drug'] = new_drug_data

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        # assert len(results['output']) == len(results['labels']), (
        #     'The length of output is not equal to the labels: '
        #     f'{len(results["output"])} != {len(results["labels"])}')
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
