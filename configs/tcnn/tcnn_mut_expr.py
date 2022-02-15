exp_name = 'tcnn_0fold_mut_expr'

_base_ = [
    '../_base_/models/tcnn/tcnn_drper.py',
    '../_base_/dataset/drp_dataset/plain_drugs_genes_dataset.py',
    '../_base_/exp_setting/base_setting.py'
]
data = dict(
    train=dict(
        include_omic=['expr', 'mut']
    ),
    val=dict(
        include_omic=['expr', 'mut']
    ),
    test=dict(
        include_omic=['expr', 'mut']
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
         ),
]

work_dir = f'workdir/{exp_name}'