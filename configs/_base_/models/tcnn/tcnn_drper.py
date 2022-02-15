model = dict(
    type='BasicDRPNet',
    drper=dict(
        type='TcnnDRPer',
        drug_encoder=dict(
            type='TcnnConvEncoder',
            in_channels=78
        ),
        genes_encoder=dict(
            type='TcnnConvEncoder',
            in_channels=1
        ),
        head=dict(
            type='TcnnFusionHead',
            out_channels=1,
            dropout=0.2
        )
    ),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')

)

train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE',
                         'R2', 'PEARSON', 'SPEARMAN'])
