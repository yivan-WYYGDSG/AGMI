from collections import OrderedDict

from ..base import BaseModel
from ..registry import MODELS
from drp.core import mse, rmse, r2, pearson, spearman, mae
from ..builder import build_backbone, build_loss, build_component
from mmcv.runner import auto_fp16
import torch
from .basic_module import BasicDRPNet
from torch import nn


@MODELS.register_module()
class MultiEdgeDRPNet(nn.Module):
    """Basic model for drug response prediction

    It must contain a drper that takes an drug garph and genes information as inputs and outputs a
    predicted IC50. It also has a mse loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
    """
    allowed_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse,
                       'R2': r2, 'PEARSON': pearson, 'SPEARMAN': spearman}

    def __init__(self,
                 drper,
                 loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.drper = build_backbone(drper)
        # self.init_weights(pretrained)

        # loss
        self.loss = build_loss(loss)

    def update_encoder_buffer(self, batch, cell_edges_attr, cell_edges_index, num_genes_nodes):
        self.drper.genes_encoder.update_buffer(self._scatter_edges_attr(batch, cell_edges_attr),
                                               self._scatter_edges_index(batch, cell_edges_index, num_genes_nodes),
                                               )

    def _scatter_edges_index(self, batch, cell_edges_index, num_genes_nodes):
        assert cell_edges_index is not None, "cell_edges_index is None!"
        tmp0 = cell_edges_index[0, :]
        tmp1 = cell_edges_index[1, :]
        for b in range(batch - 1):
            tmp0 = torch.cat((tmp0, cell_edges_index[0, :] + (b + 1) * num_genes_nodes))
            tmp1 = torch.cat((tmp1, cell_edges_index[1, :] + (b + 1) * num_genes_nodes))
        tmp0 = tmp0 - 1
        tmp1 = tmp1 - 1
        return torch.stack([tmp0, tmp1], dim=0)

    def _scatter_edges_attr(self, batch, cell_edges_attr):
        assert cell_edges_attr is not None, "cell_edges_attr is None!"
        tmp_weight = torch.zeros((3, cell_edges_attr[0].shape[0] * batch))
        for idx, w in enumerate(cell_edges_attr):
            tmp = w
            for _ in range(batch - 1):
                w = torch.cat((w, tmp))
            tmp_weight[idx, :] = w
        return tmp_weight

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.drper.init_weights(pretrained)

    @auto_fp16(apply_to=('data',))
    def forward(self, data, test_mode=False, **kwargs):
        """Forward function.

        Args:
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
            :param test_mode:
            :param gt:
            :param data:
        """

        labels = data.y

        if test_mode:
            return self.forward_test(data, labels, **kwargs)

        return self.forward_train(data, labels)

    def forward_train(self, data, labels):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
            :param gt:
            :param data:
        """
        losses = dict()
        data = data.cuda()
        labels = labels.view(-1, 1).float().cuda()
        output = self.drper(data)
        loss_drp = self.loss(output, labels)
        losses['loss_drp'] = loss_drp
        outputs = dict(
            losses=losses,
            num_samples=len(labels.data),
            results=dict(labels=labels.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:

        Returns:
            dict: Evaluation results.
        """

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt)
        return eval_result

    def forward_test(self,
                     data,
                     labels=None, ):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
            :param gt:
            :param data:
        """
        data.cuda()
        labels = labels.view(-1, 1)
        output = self.drper(data)
        results = dict(output=output.cpu())
        if labels is not None:
            results['labels'] = labels.cpu()

        return results

    def forward_dummy(self, data):
        """Used for computing network FLOPs.

        Args:
            data (Tensor): Input data.

        Returns:
            Tensor: Output data.
        """
        out = self.drper(data)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))
        optimizer['drper'].zero_grad()
        loss.backward()
        optimizer['drper'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def parse_losses(self, losses):
        """Parse losses dict for different loss variants.

        Args:
            losses (dict): Loss dict.

        Returns:
            loss (float): Sum of the total loss.
            log_vars (dict): loss dict for different variants.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars
