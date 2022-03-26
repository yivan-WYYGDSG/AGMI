import argparse
import os

import mmcv
import torch
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from drp.apis import set_random_seed, single_gpu_test
from drp.datasets import build_dataloader, build_dataset
from drp.models import build_model
from drp.datasets.pipelines.utils import get_weight




def parse_args():
    parser = argparse.ArgumentParser(description='drp tester')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    # set random seeds
    if args.seed is not None:
        if rank == 0:
            print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    loader_cfg = {
        **dict((k, cfg.data[k]) for k in ['workers_per_gpu'] if k in cfg.data),
        **dict(
            samples_per_gpu=1,
            drop_last=False,
            shuffle=False,
            dist=distributed),
        **cfg.data.get('test_dataloader', {})
    }

    data_loader = build_dataloader(dataset, **loader_cfg)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    for k,v in model.state_dict().items():
        print(f'{k} : {v.shape}')
    model.load_state_dict(torch.load(args.checkpoint))
    
    if cfg.edges is not None:
        gsea_path, ppi_path, pearson_path = cfg.edges[0], cfg.edges[1], cfg.edges[2]
        cell_edges_index, cell_edges_attr = get_weight(gsea_path, ppi_path, pearson_path)
        model.update_encoder_buffer(cfg.test_batch_size, cell_edges_attr, cell_edges_index, 18498)

    for k,v in model.state_dict().items():
        print(f'{k} : {v.shape}')
    outputs = single_gpu_test(
        model.cuda(),
        data_loader)


    if rank == 0:
        print('')
        # print metrics
        stats = dataset.evaluate(outputs)
        for stat in stats:
            print('Eval-{}: {}'.format(stat, stats[stat]))

        # save result pickle
        if args.out:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()
