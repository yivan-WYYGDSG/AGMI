model = dict(
    type='BasicDRPNet',
    drper=dict(
        type='MOLIDRPer',
        drug_encoder=dict(
            type='TcnnConvEncoder',
            in_channels=78
        ),
        genes_encoder=dict(
            type='CDRConvEncoder',
            dropout_rate=0.1
        ),
        head=dict(
            type='BaseFusionHead',
            d_in_channels=180,
            g_in_channels=100,
            out_channels=1,
            reduction=2,
            dropout=0.2
        ),
    ),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')

)

train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE',
                         'R2', 'PEARSON', 'SPEARMAN'])
