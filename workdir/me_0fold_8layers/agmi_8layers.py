model = dict(
    type='AGMIDRPNet',
    drper=dict(
        type='AGMIDRPer',
        in_channel=3,
        gather_width=6,
        drug_encoder=dict(
            type='DrugGATEncoder',
            num_features_xd=78,
            heads=10,
            output_dim=128,
            gat_dropout=0.2),
        genes_encoder=dict(
            type='MultiEdgeGatedGraphConv',
            out_channels=3,
            num_layers=6,
            num_edges=3,
            aggr='add',
            bias=True),
        head=dict(type='AGMIFusionHead', out_channels=128),
        neck=dict(
            type='AGMICellNeck',
            in_channels=[6, 8, 16],
            out_channels=[8, 16, 32],
            kernel_size=[16, 16, 16],
            drop_rate=0.2,
            max_pool_size=[3, 6, 6],
            feat_dim=128)),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'))
train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE', 'R2', 'PEARSON', 'SPEARMAN'])
data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(
        samples_per_gpu=32,
        drop_last=True,
        pin_memory=False,
        exclude_keys=[],
        follow_batch=['x_cell']),
    val_dataloader=dict(
        samples_per_gpu=32,
        drop_last=True,
        pin_memory=False,
        exclude_keys=[],
        follow_batch=['x_cell']),
    test_dataloader=dict(
        samples_per_gpu=8,
        drop_last=True,
        pin_memory=False,
        exclude_keys=[],
        follow_batch=['x_cell']),
    train=dict(
        type='InMemoryMultiEdgeGraphGenesDataset',
        data_items='data/split/0_fold_tr_items.npy',
        celllines_data='data/processed_raw_data/564_cellGraphs_exp_mu_cn.npy',
        num_genes_nodes=18498,
        metrics=['RMSE', 'MSE', 'R2', 'PEARSON', 'MAE', 'SPEARMAN'],
        drug_graphs='data/processed_raw_data/drugId_drugGraph.npy',
        root='data',
        name='MultiEdgeGraphGenes_0_fold_tr',
        transform=None,
        pre_transform=None),
    val=dict(
        type='InMemoryMultiEdgeGraphGenesDataset',
        data_items='data/split/0_fold_val_items.npy',
        celllines_data='data/processed_raw_data/564_cellGraphs_exp_mu_cn.npy',
        num_genes_nodes=18498,
        metrics=['RMSE', 'MSE', 'R2', 'PEARSON', 'MAE', 'SPEARMAN'],
        drug_graphs='data/processed_raw_data/drugId_drugGraph.npy',
        root='./data',
        name='MultiEdgeGraphGenes_0_fold_val',
        transform=None,
        pre_transform=None),
    test=dict(
        type='InMemoryMultiEdgeGraphGenesDataset',
        data_items='data/split/test_items.npy',
        celllines_data='data/processed_raw_data/564_cellGraphs_exp_mu_cn.npy',
        num_genes_nodes=18498,
        metrics=['RMSE', 'MSE', 'R2', 'PEARSON', 'MAE', 'SPEARMAN'],
        drug_graphs='data/processed_raw_data/drugId_drugGraph.npy',
        root='./data',
        name='MultiEdgeGraphGenes_test',
        transform=None,
        pre_transform=None))
optimizers = dict(drper=dict(type='Adam', lr=0.0001))
total_iters = 1000000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 800000, 1000000],
    gamma=0.5)
checkpoint_config = dict(
    interval=10000, save_optimizer=True, by_epoch=False, max_keep_ckpts=10)
evaluation = dict(interval=10000)
visual_config = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
log_config = dict(
    interval=2500, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
exp_name = 'me_0fold_8layers'
edges = [
    'data/edges/processed/GSEA_edge_indexes_all_pairs_426904_weighted.npy',
    'data/edges/processed/STRING_edge_index_all_10463181_pairs_weighted.npy',
    'data/edges/processed/edge_index_pearson_Thr06_148855_pairs.npy'
]
test_batch_size = 8
custom_hooks = [
    dict(
        type='TensorboardXHook',
        priority=85,
        log_dir='/data2/xieyufeng/drp_results/tb_data/',
        interval=2500,
        exp_name='me_0fold_8layers',
        ignore_last=True,
        reset_flag=False,
        by_epoch=False),
    dict(
        type='MEHook',
        priority='VERY_LOW',
        gsea_path=
        'data/edges/processed/GSEA_edge_indexes_all_pairs_426904_weighted.npy',
        ppi_path=
        'data/edges/processed/STRING_edge_index_all_10463181_pairs_weighted.npy',
        pearson_path=
        'data/edges/processed/edge_index_pearson_Thr06_148855_pairs.npy',
        num_nodes=18498)
]
work_dir = 'workdir/me_0fold_8layers'
gpus = 1
