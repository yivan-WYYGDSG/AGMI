dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
log_config = dict(
    interval=2500,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])