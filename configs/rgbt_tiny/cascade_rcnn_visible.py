# Force MMDetection to load custom SAFit logic
custom_imports = dict(imports=['codes.loss.loss'], allow_failed_imports=False)

# Standard baseline inheritance
_base_ = ['../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py']

classes = ('ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane')

model = dict(
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=7,
                loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=7,
                loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=7,
                loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)
            )
        ]
    )
)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/rgbt_tiny/'

train_dataloader = dict(
    batch_size=4,  # Fits perfectly in 10GB/12GB RTX 3080
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/visible_train.json',
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/visible_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader

# Evaluator using authors' custom cocoeval logic
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/visible_test.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

# Optimization: Linear Scaling Rule Applied
# Batch 4 (yours) vs Batch 16 (baseline) = 0.25x scaling
# LR 0.02 (baseline) * 0.25 = 0.005
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2) # Added for SAFit stability
)

# Standard 1x Schedule
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

work_dir = './work_dirs/cascade_rcnn_visible'