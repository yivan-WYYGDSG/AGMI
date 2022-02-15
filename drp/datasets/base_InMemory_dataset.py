from itertools import repeat

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from .registry import DATASETS
from abc import ABCMeta, abstractmethod
import copy
import os.path as osp
import os
import numpy as np
from tqdm import tqdm


@DATASETS.register_module()
class BaseInMemoryDataset(InMemoryDataset, metaclass=ABCMeta):
    r"""

    Args:
        root (string, optional): Root directory where the dataset should be
            saved. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, transform, pre_transform, test_mode=False):
        super().__init__(root, transform, pre_transform)
        self.test_mode = test_mode
        self.transform = transform
        self.pre_transform = pre_transform

    @abstractmethod
    def process(self):
        """Abstract function for loading data

        All subclasses should overwrite this function
        """

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        return []

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get(self, idx: int) -> dict:
        if hasattr(self, '_data_list'):
            if self._data_list is None:
                self._data_list = self.len() * [None]
            else:
                data = self._data_list[idx]
                if data is not None:
                    return dict(data=copy.copy(data))

        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)
                if cat_dim is None:
                    cat_dim = 0
                s[cat_dim] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        if hasattr(self, '_data_list'):
            self._data_list[idx] = copy.copy(data)

        return dict(data=data)
