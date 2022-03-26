exp_name = 'cdr_0fold'

_base_ = [
    '../_base_/models/late_integration_drp/base_deepCDR.py',
    '../_base_/dataset/drp_dataset/drugs_genes_dataset.py',
    '../_base_/exp_setting/base_setting.py',
    '../_base_/default_runtime.py'
]
data = dict(
    train=dict(
        celllines_data='data/processed_raw_data/564_cellGraphs_exp_mu_cn_new.npy',
        include_omic=['expr', 'mut', 'dna'],
    ),
    val=dict(
        celllines_data='data/processed_raw_data/564_cellGraphs_exp_mu_cn_new.npy',
        include_omic=['expr', 'mut', 'dna'],
    ),
    test=dict(
        celllines_data='data/processed_raw_data/564_cellGraphs_exp_mu_cn_new.npy',
        include_omic=['expr', 'mut', 'dna'],
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