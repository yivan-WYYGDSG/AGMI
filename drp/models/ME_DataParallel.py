from torch.nn.parallel.distributed import _find_tensors
from drp.parallel import GraphDataParallel
from mmcv import print_log
from mmcv.parallel import MMDistributedDataParallel
import torch
from torch_geometric.data import Batch
from itertools import chain
from mmcv.utils import TORCH_VERSION, digit_version
from drp.parallel import scatter_kwargs

class MEDataParallel(GraphDataParallel):
    def __init__(self, module, device_ids=None, output_device=None,
                 follow_batch=[], exclude_keys=[], **kwargs):
        super(MEDataParallel, self).__init__(module, device_ids, output_device, follow_batch, exclude_keys, **kwargs)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Use `self.to_kwargs` instead of `self.scatter` in pytorch1.8
        # to move all tensors to device_id
        return scatter_kwargs(inputs, kwargs, [device_id], dim=self.dim)

    def mm_scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        """Train step function.

        Args:
            inputs (Tensor): Input Tensor.
            kwargs (dict): Args for
                ``mmcv.parallel.scatter_gather.scatter_kwargs``.
        """
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')
        if getattr(self, 'require_forward_param_sync', True):
            self._sync_params()

        # if using DataListLoader for training
        # data_list = [d['data'] for d in inputs[0]]
        # if len(data_list) == 0:
        #     print_log('DataParallel received an empty data list, which '
        #               'may result in unexpected behaviour.', logger='MEDataParallel')
        #     return None
        #
        # if not self.device_ids or len(self.device_ids) == 1:  # Fallback
        #     data = Batch.from_data_list(
        #         data_list, follow_batch=self.follow_batch,
        #         exclude_keys=self.exclude_keys).to(self.src_device)
        #     return self.module(data)

        # for t in chain(self.module.parameters(), self.module.buffers()):
        #     if t.device != self.src_device:
        #         raise RuntimeError(
        #             ('Module must have its parameters and buffers on device '
        #              '{} but found one of them on device {}.').format(
        #                 self.src_device, t.device))

        if self.device_ids:
            # inputs = self.scatter(data_list, self.device_ids)
            inputs, kwargs = self.mm_scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)

            # replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            # outputs = self.parallel_apply(replicas, inputs, None)
            # output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        """val_step() API for module wrapped by DistributedDataParallel.

        This method is basically the same as
        ``DistributedDataParallel.forward()``, while replacing
        ``self.module.forward()`` with ``self.module.val_step()``.
        It is compatible with PyTorch 1.1 - 1.5.
        """
        # In PyTorch >= 1.7, ``reducer._rebuild_buckets()`` is moved from the
        # end of backward to the beginning of forward.
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.7')
                and self.reducer._rebuild_buckets()):
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        if getattr(self, 'require_forward_param_sync', True):
            self._sync_params()

        # data_list = [d['data'] for d in inputs[0]]
        # if len(data_list) == 0:
        #     print_log('DataParallel received an empty data list, which '
        #               'may result in unexpected behaviour.', logger='MEDataParallel')
        #     return None
        #
        # if not self.device_ids or len(self.device_ids) == 1:  # Fallback
        #     data = Batch.from_data_list(
        #         data_list, follow_batch=self.follow_batch,
        #         exclude_keys=self.exclude_keys).to(self.src_device)
        #     return self.module(data)
        #
        # for t in chain(self.module.parameters(), self.module.buffers()):
        #     if t.device != self.src_device:
        #         raise RuntimeError(
        #             ('Module must have its parameters and buffers on device '
        #              '{} but found one of them on device {}.').format(
        #                 self.src_device, t.device))

        if self.device_ids:
            # inputs = self.scatter(data_list, self.device_ids)
            inputs, kwargs = self.mm_scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        return output

    def update_encoder_buffer(self, batch, cell_edges_attr, cell_edges_index, num_genes_nodes):
        self.module.update_encoder_buffer(batch, cell_edges_attr, cell_edges_index, num_genes_nodes)
