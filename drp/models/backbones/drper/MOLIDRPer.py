import torch.nn
from drp.models.registry import BACKBONES
from drp.models.builder import build_component

from mmcv.runner import load_checkpoint

from drp.utils import get_root_logger

from drp.models.common import generation_init_weights


@BACKBONES.register_module()
class MOLIDRPer(torch.nn.Module):

    def __init__(self,
                 drug_encoder,
                 genes_encoder,
                 head):
        super().__init__()

        # body
        self.drug_encoder = build_component(drug_encoder)
        self.genes_encoder = build_component(genes_encoder)
        self.head = build_component(head)

    def init_weights(self, pretrained=None, strict=True):

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type='normal', init_gain=0.02)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

    def forward(self, cell, drug):
        """Forward function.

        Args:
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
            :param test_mode:
            :param gt:
            :param data:
        """
        x_cell, x_d = cell, drug
        x_d = x_d.permute(0, 2, 1)

        drug_embed = self.drug_encoder(x_d)
        genes_embed = self.genes_encoder(x_cell)
        output = self.head(drug_embed, genes_embed)

        return output
