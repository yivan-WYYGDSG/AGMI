import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader
from drp.apis import single_gpu_test

class EvalIterHook(Hook):
    """Non-Distributed evaluation hook for iteration-based runner.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval. Default: 1.
        eval_kwargs (dict): Other eval kwargs. It contains:
            save_image (bool): Whether to save image.
            save_path (str): The path to save image.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, '
                            f'but got { type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        results = single_gpu_test(
            runner.model,
            self.dataloader)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        # runner.log_buffer.output['data_time'] = 0
        runner.log_buffer.ready = True


