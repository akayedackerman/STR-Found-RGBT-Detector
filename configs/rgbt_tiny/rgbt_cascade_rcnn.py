# =========================================================
# RGBT Cascade RCNN + SAFit (STABLE VERSION)
# =========================================================

_base_ = ['../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py']

# =========================================================
# CUSTOM IMPORTS
# =========================================================

custom_imports = dict(
    imports=[
        'codes.loss.loss',
        'codes.pipeline.loading',
        'codes.models.resnet6ch'
    ],
    allow_failed_imports=False
)

# =========================================================
# CLASSES
# =========================================================

classes = (
    'ship',
    'car',
    'cyclist',
    'pedestrian',
    'bus',
    'drone',
    'plane'
)

# =========================================================
# MODEL
# =========================================================

model = dict(

    # -----------------------------------------------------
    # DATA PREPROCESSOR
    # -----------------------------------------------------

    data_preprocessor=dict(
        type='DetDataPreprocessor',

        # already normalized manually
        mean=None,
        std=None,

        # IMPORTANT
        # prevents MMDetection from slicing channels
        bgr_to_rgb=False,

        pad_size_divisor=32
    ),

    # -----------------------------------------------------
    # BACKBONE
    # -----------------------------------------------------

    backbone=dict(
        type='ResNet6Channel',

        in_channels=6,

        depth=50,

        num_stages=4,

        out_indices=(0, 1, 2, 3),

        frozen_stages=1,

        norm_cfg=dict(
            type='BN',
            requires_grad=True
        ),

        norm_eval=True,

        style='pytorch',

        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50'
        )
    ),

    # -----------------------------------------------------
    # RPN HEAD
    # -----------------------------------------------------

    rpn_head=dict(

        anchor_generator=dict(

            type='AnchorGenerator',

            # tuned for tiny objects
            scales=[2, 4],

            ratios=[0.5, 1.0, 2.0],

            strides=[4, 8, 16, 32, 64]
        )
    ),

    # -----------------------------------------------------
    # ROI HEAD
    # -----------------------------------------------------

    roi_head=dict(

        bbox_head=[

            dict(
                type='Shared2FCBBoxHead',

                num_classes=7,

                # IMPORTANT STABILITY FIX
                reg_decoded_bbox=False,

                loss_bbox=dict(
                    type='SAFitLoss',
                    loss_weight=1.0
                )
            ),

            dict(
                type='Shared2FCBBoxHead',

                num_classes=7,

                reg_decoded_bbox=False,

                loss_bbox=dict(
                    type='SAFitLoss',
                    loss_weight=1.0
                )
            ),

            dict(
                type='Shared2FCBBoxHead',

                num_classes=7,

                reg_decoded_bbox=False,

                loss_bbox=dict(
                    type='SAFitLoss',
                    loss_weight=1.0
                )
            )
        ]
    )
)

# =========================================================
# DATASET
# =========================================================

dataset_type = 'CocoDataset'

data_root = 'data/rgbt_tiny/'

# =========================================================
# TRAIN PIPELINE
# =========================================================

train_pipeline = [

    dict(
        type='LoadImageFromFile',
        to_float32=True
    ),

    dict(
        type='LoadThermalImageFromFile'
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=True
    ),

    dict(
        type='Resize',
        scale=(640, 512),
        keep_ratio=True
    ),

    dict(
        type='ConcatThermalToRGB'
    ),

    dict(
        type='RandomFlip',
        prob=0.5
    ),

    dict(
        type='PackDetInputs'
    )
]

# =========================================================
# TEST PIPELINE
# =========================================================

test_pipeline = [

    dict(
        type='LoadImageFromFile',
        to_float32=True
    ),

    dict(
        type='LoadThermalImageFromFile'
    ),

    dict(
        type='Resize',
        scale=(640, 512),
        keep_ratio=True
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=True
    ),

    dict(
        type='ConcatThermalToRGB'
    ),

    dict(
        type='PackDetInputs'
    )
]

# =========================================================
# TRAIN DATALOADER
# =========================================================

train_dataloader = dict(

    batch_size=4,

    num_workers=4,

    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True
    ),

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        metainfo=dict(
            classes=classes
        ),

        ann_file='annotations/visible_train.json',

        data_prefix=dict(
            img='images/'
        ),

        pipeline=train_pipeline
    )
)

# =========================================================
# VALIDATION DATALOADER
# =========================================================

val_dataloader = dict(

    batch_size=1,

    num_workers=2,

    persistent_workers=True,

    dataset=dict(

        type=dataset_type,

        data_root=data_root,

        metainfo=dict(
            classes=classes
        ),

        ann_file='annotations/visible_test.json',

        data_prefix=dict(
            img='images/'
        ),

        test_mode=True,

        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# =========================================================
# EVALUATOR
# =========================================================

val_evaluator = dict(

    type='CocoMetric',

    ann_file=data_root + 'annotations/visible_test.json',

    metric='bbox',

    format_only=False
)

test_evaluator = val_evaluator

# =========================================================
# OPTIMIZER
# =========================================================

optim_wrapper = dict(

    type='OptimWrapper',

    optimizer=dict(

        type='SGD',

        # LOWERED FOR RGB-T STABILITY
        lr=0.001,

        momentum=0.9,

        weight_decay=0.0001
    ),

    clip_grad=dict(
        max_norm=35,
        norm_type=2
    )
)

# =========================================================
# TRAINING LOOP
# =========================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

val_cfg = dict(type='ValLoop')

test_cfg = dict(type='TestLoop')

# =========================================================
# LR SCHEDULER
# =========================================================

param_scheduler = [

    dict(
        type='LinearLR',

        start_factor=0.001,

        by_epoch=False,

        begin=0,

        end=500
    ),

    dict(
        type='MultiStepLR',

        begin=0,

        end=12,

        by_epoch=True,

        milestones=[8, 11],

        gamma=0.1
    )
]

# =========================================================
# DEFAULT HOOKS
# =========================================================

default_hooks = dict(

    timer=dict(type='IterTimerHook'),

    logger=dict(
        type='LoggerHook',
        interval=50
    ),

    param_scheduler=dict(
        type='ParamSchedulerHook'
    ),

    checkpoint=dict(
        type='CheckpointHook',
        interval=1
    ),

    sampler_seed=dict(
        type='DistSamplerSeedHook'
    ),

    visualization=dict(
        type='DetVisualizationHook'
    )
)

# =========================================================
# ENV CONFIG
# =========================================================

env_cfg = dict(
    cudnn_benchmark=False
)

# =========================================================
# WORK DIRECTORY
# =========================================================

work_dir = './work_dirs/rgbt_cascade_rcnn_stable'