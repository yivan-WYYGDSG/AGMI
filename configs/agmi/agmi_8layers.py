exp_name = 'me_0fold_8layers'

_base_ = [
    '../_base_/models/AGMI/agmi_8layers.py',
    '../_base_/drp_dataset/drugs_genes_dataset.py',
    '../_base_/exp_setting/base_setting.py',
    '../_base_/default_runtime.py'
]

edges = ['data/edges/processed/GSEA_edge_indexes_all_pairs_426904_weighted.npy', 
         'data/edges/processed/STRING_edge_index_all_10463181_pairs_weighted.npy' ,
         'data/edges/processed/edge_index_pearson_Thr06_148855_pairs.npy']

model = dict(
    drper=dict(
        genes_encoder=dict(
            num_layers=6,
        ),
    ),
)

test_batch_size = 8

data = dict(
    test_dataloader=dict(samples_per_gpu=test_batch_size, drop_last=True, pin_memory=False, exclude_keys=[], follow_batch=['x_cell']),
)

custom_hooks = [
    dict(type='TensorboardXHook',
         priority=85,
         log_dir='/data2/xieyufeng/drp_results/tb_data/',
         interval=2500,
         exp_name=exp_name,
         ignore_last=True,
         reset_flag=False,
         by_epoch=False
         ),
    dict(type='MEHook',
         priority='VERY_LOW',
         gsea_path=edges[0],
         ppi_path=edges[1],
         pearson_path=edges[2],
         num_nodes=18498
         )
]

work_dir = f'workdir/{exp_name}'
