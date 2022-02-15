from mmcv.runner import HOOKS, Hook
import numpy as np
from drp.datasets.pipelines.utils import get_weight
import os.path as osp
import os


@HOOKS.register_module()
class AttnHook(Hook):
    def __init__(self,
                 save_path,
                 interval):
        self.save_path = save_path
        self.interval = interval
        if not osp.exists(save_path):
            os.makedirs(self.save_path)

    def after_train_iter(self, runner):
        if 'attn_weights' in runner.outputs and self.every_n_iters(runner, self.interval):
            attn_weights = runner.outputs.pop('attn_weights')
            attn_weights = attn_weights.squeeze().cpu().numpy()
            np.save(osp.join(self.save_path,'train_attn.npy'), attn_weights)
            print(f'save train attn_weights to {self.save_path}')
        self.after_iter(runner)

    def after_val_iter(self, runner):
        if 'attn_weights' in runner.outputs and self.self.every_n_iters(runner, self.interval):
            attn_weights = runner.outputs.pop('attn_weights')
            attn_weights = attn_weights.squeeze().cpu().numpy()
            np.save(osp.join(self.save_path,'val_attn.npy'), attn_weights)
            print(f'save attn_weights to {self.save_path}')
        self.after_iter(runner)

    def after_train_epoch(self, runner):
        if 'attn_weights' in runner.outputs:
            attn_weights = runner.outputs.pop('attn_weights')
            attn_weights = attn_weights.squeeze().cpu().numpy()
            np.save(osp.join(self.save_path,'train_attn.npy'), attn_weights)
            print(f'save train attn_weights to {self.save_path}')
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        if 'attn_weights' in runner.outputs:
            attn_weights = runner.outputs.pop('attn_weights')
            attn_weights = attn_weights.squeeze().cpu().numpy()
            np.save(osp.join(self.save_path,'val_attn.npy'), attn_weights)
            print(f'save attn_weights to {self.save_path}')
        self.after_epoch(runner)