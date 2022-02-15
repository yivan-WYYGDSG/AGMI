model = dict(
    type='MutiEdgeDRPNet',
    drper=dict(
        type='BasicDRPer',
        drug_encoder=dict(
            type='DrugGINEncoder',
            drug_features=78,
            output_dim=128,
            dropout=0.2,
            hidden_dim=32
        ),
        genes_encoder=dict(
            type='NaiveGenesEncoder',
            n_filters=32,
            output_dim=128
        ),
        head=dict(
            type='BaseFusionHead',
            d_in_channels=128,
            g_in_channels=128,
            out_channels=1,
            reduction=2,
            dropout=0.2
        )
    ),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')

)

train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE',
                         'R2', 'PEARSON', 'SPEARMAN'])
