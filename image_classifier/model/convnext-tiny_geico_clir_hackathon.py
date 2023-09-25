model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt', arch='tiny', drop_path_rate=0.0, frozen_stages=3),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=False),
        init_cfg=None),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/best_on_generated2/epoch_100.pth'),
    train_cfg=dict())
dataset_type = 'Parquet'
resize_dim = 224
data_preprocessor = dict(
    num_classes=2,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Albumentations',
        transforms=[
            dict(type='Affine', rotate=(-15, 15), shear=(-3, 3), p=1.0)
        ]),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.7, 1.0),
        aspect_ratio_range=(0.75, 1.3333333333333333)),
    dict(
        type='Albumentations',
        transforms=[
            dict(type='HorizontalFlip', p=0.5),
            dict(type='RandomRotate90', p=0.1),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.2],
                contrast_limit=[0.1, 0.2],
                p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=20,
                        g_shift_limit=20,
                        b_shift_limit=20,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=40,
                        sat_shift_limit=60,
                        val_shift_limit=40,
                        p=1.0)
                ],
                p=0.5)
        ]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=96,
    num_workers=6,
    dataset=dict(
        type='Parquet',
        parquet_file='data/train.parquet',
        regression_mode=False,
        validate_files=False,
        subset_size=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Albumentations',
                transforms=[
                    dict(
                        type='Affine', rotate=(-15, 15), shear=(-3, 3), p=1.0)
                ]),
            dict(
                type='RandomResizedCrop',
                scale=224,
                crop_ratio_range=(0.7, 1.0),
                aspect_ratio_range=(0.75, 1.3333333333333333)),
            dict(
                type='Albumentations',
                transforms=[
                    dict(type='HorizontalFlip', p=0.5),
                    dict(type='RandomRotate90', p=0.1),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.2],
                        contrast_limit=[0.1, 0.2],
                        p=0.5),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RGBShift',
                                r_shift_limit=20,
                                g_shift_limit=20,
                                b_shift_limit=20,
                                p=1.0),
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=40,
                                sat_shift_limit=60,
                                val_shift_limit=40,
                                p=1.0)
                        ],
                        p=0.5)
                ]),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=96,
    num_workers=6,
    dataset=dict(
        type='Parquet',
        parquet_file='data/val.parquet',
        regression_mode=False,
        validate_files=False,
        subset_size=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=96,
    num_workers=6,
    dataset=dict(
        type='Parquet',
        parquet_file='data/test.parquet',
        regression_mode=False,
        validate_files=False,
        subset_size=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
num_gpu = 1
batch_size = 96
num_threads = 6
num_classes = 2
checkpoint_file = 'work_dirs/best_on_generated2/epoch_100.pth'
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator = dict(type='Accuracy', topk=(1, ))
multiplier = 0.5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=7.500000000000001e-05,
        weight_decay=0.01,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.4,
        by_epoch=True,
        end=5,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=5e-06, by_epoch=False, begin=1)
]
custom_hooks = [dict(type='EMAHook', momentum=0.0001, priority='ABOVE_NORMAL')]
launcher = 'none'
work_dir = 'work_dirs/test_all'
