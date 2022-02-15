import os.path as osp
from tensorboardX import SummaryWriter
import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image
from mmcv.runner import LoggerHook, TextLoggerHook
import matplotlib.pyplot as plt


@HOOKS.register_module()
class TensorboardXHook(LoggerHook):
    def __init__(self,
                 log_dir,
                 exp_name,
                 interval,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        super(TensorboardXHook, self).__init__(interval, ignore_last,
                                               reset_flag, by_epoch)
        self.log_dir = osp.join(log_dir, 'runs')
        mmcv.mkdir_or_exist(self.log_dir)
        self.log_path = osp.join(self.log_dir, exp_name)

    @master_only
    def before_run(self, runner):
        self.writer = SummaryWriter(self.log_path)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            # print("tensorboard log:{}".format(tag))
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif tag == 'val/output':
                y = tags['val/labels']
                x = val
                self.writer.add_embedding(val, metadata=y, global_step=runner.iter)
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.plot(x, x, color='red')
                ax1.scatter(x=val, y=y)
                self.writer.add_figure(tag="val_figure", figure=fig, global_step=runner.iter)

                runner.log_buffer.output.pop('output')
                runner.log_buffer.output.pop('labels')
            elif tag == 'val/labels':
                continue
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()
