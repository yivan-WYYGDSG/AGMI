model = dict(
    type='AGMIDRPNet',
    drper=dict(
        type='AGMIDRPer',
        in_channel = 3,
        gather_width=6,
        drug_encoder=dict(
            type='DrugGATEncoder',
            num_features_xd=78, 
            heads=10, 
            output_dim=128, 
            gat_dropout=0.2
        ),
        genes_encoder=dict(
            type='MultiEdgeGatedGraphConv',
            out_channels=3, 
            num_layers=6, 
            num_edges=3,
            aggr='add',
            bias=True,
        ),
        head=dict(
            type='AGMIFusionHead',
            out_channels=128,
        ),
        neck=dict(
            type='AGMICellNeck',
            in_channels=[6,8,16], 
            out_channels=[8,16,32], 
            kernel_size=[16,16,16], 
            drop_rate=0.2, 
            max_pool_size=[3,6,6], 
            feat_dim=128
        ),
    ),
    loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
)

train_cfg = None
test_cfg = dict(metrics=['MAE', 'MSE', 'RMSE',
                         'R2', 'PEARSON', 'SPEARMAN'])
