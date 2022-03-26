from torch import nn
from ..registry import MODELS
from drp.core import mse, rmse, r2, pearson, spearman, mae
from ..builder import build_backbone, build_loss, build_component
from mmcv.runner import auto_fp16


@MODELS.register_module()
class BasicDRPNet(nn.Module):
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

        # loss
        self.loss = build_loss(loss)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.drper.init_weights(pretrained)

    @auto_fp16(apply_to=('data',))
    def forward(self, cell, drug, labels, test_mode=False, **kwargs):
        """Forward function.

        Args:
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
            :param test_mode:
            :param gt:
            :param data:
        """

        if test_mode:
            return self.forward_test(cell, drug, labels)

        return self.forward_train(cell, drug, labels)

    def forward_train(self, cell, drug, labels):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
            :param data:
        """
        losses = dict()
        cell =cell.cuda()
        drug = drug.cuda()
        labels = labels.view(-1,1).float().cuda()
        output = self.drper(cell, drug)
        loss_drp = self.loss(output, labels)
        losses['loss_drp'] = loss_drp
        outputs = dict(
            losses=losses,
            num_samples=len(labels.data),
            results=dict(labels=labels.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, labels):
        """Evaluation function.

        Args:

        Returns:
            dict: Evaluation results.
        """

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, labels)
        return eval_result

    def forward_test(self,
                     cell,
                     drug,
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
        cell = cell.cuda()
        drug = drug.cuda()
        labels = labels.view(-1, 1)
        output = self.drper(cell, drug)
        results = dict(output=output.cpu())
        if labels is not None:
            results['labels'] = labels.cpu()

        return results

    def forward_dummy(self, cell, drug):
        """Used for computing network FLOPs.

        Args:
            data (Tensor): Input data.

        Returns:
            Tensor: Output data.
        """
        out = self.drper(cell, drug)
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
