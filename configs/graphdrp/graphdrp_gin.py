exp_name = 'grpahdrp_gin_0fold'

_base_ = [
    '../_base_/models/GraphDRP/GraphDRP_GIN.py',
    '../_base_/dataset/drp_dataset/drugs_genes_dataset.py',
    '../_base_/exp_setting/base_setting.py'
]

data = dict(
    train=dict(
        data_items='data/split/0_fold_tr_items.npy',
        name='MultiEdgeGraphGenes_0_fold_tr',
        include_omic=['expr', 'mut', 'cn']
    ),
    val=dict(
        data_items='data/split/0_fold_val_items.npy',
        name='MultiEdgeGraphGenes_0_fold_val',
        include_omic=['expr', 'mut', 'cn']
    ),
)

custom_hooks = [
    dict(type='TensorboardXHook',
         priority=85,
         log_dir='/data/xieyufeng/genes_drug_data/result/',
         interval=5000,
         exp_name=exp_name,
         ignore_last=True,
         reset_flag=False,
         by_epoch=False
         )
]

work_dir = f'workdir/{exp_name}'