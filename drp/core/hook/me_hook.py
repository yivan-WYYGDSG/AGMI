from mmcv.runner import HOOKS, Hook
from drp.datasets.pipelines.utils import get_weight


@HOOKS.register_module()
class MEHook(Hook):
    def __init__(self,
                 gsea_path,
                 ppi_path,
                 pearson_path,
                 num_nodes=None):
        self.cell_edges_index, self.cell_edges_attr = get_weight(gsea_path, ppi_path, pearson_path)
        self.num_nodes = num_nodes
        self.pre_batch_size = -1
        
    def before_train_epoch(self, runner):
        bs = runner.data_loader._dataloader.batch_size
        if bs != self.pre_batch_size:
            print('update buffer')
            runner.model.update_encoder_buffer(bs,
                                               self.cell_edges_attr, self.cell_edges_index, self.num_nodes)
        self.pre_batch_size = bs
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        bs = runner.data_loader._dataloader.batch_size
        if bs != self.pre_batch_size:
            print('update buffer')
            runner.model.update_encoder_buffer(runner.data_loader.batch_size,
                                               self.cell_edges_attr, self.cell_edges_index, self.num_nodes)
        self.pre_batch_size = bs
        self.before_epoch(runner)

    def before_train_iter(self, runner):
        bs = runner.data_loader._dataloader.batch_size
        if bs != self.pre_batch_size:
            print('update buffer')
            runner.model.update_encoder_buffer(bs,
                                               self.cell_edges_attr, self.cell_edges_index, self.num_nodes)
        self.pre_batch_size = bs
        self.before_iter(runner)

    def before_val_iter(self, runner):
        bs = runner.data_loader._dataloader.batch_size
        if bs != self.pre_batch_size:
            print('update buffer')
            runner.model.update_encoder_buffer(bs,
                                               self.cell_edges_attr, self.cell_edges_index, self.num_nodes)
        self.pre_batch_size = bs
        self.before_iter(runner)
