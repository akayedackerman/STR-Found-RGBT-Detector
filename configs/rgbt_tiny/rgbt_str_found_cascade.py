# Inherit standard Cascade R-CNN blueprint properties dynamically
_base_ = ['../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py']

# Force MMDetection to load your custom pipeline, backbone, and contrastive loss modules
custom_imports = dict(
    imports=[
        'codes.loss.loss',
        'codes.pipeline.loading',                  # Loads your custom LoadTemporalRGBTPair
        'codes.models.dual_resnet_dual_attn_str'   # Loads your upgraded STR backbone class
    ],
    allow_failed_imports=False)

classes = ('ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane')

# ==============================================================================
# 1. MODEL ARCHITECTURE CONFIGURATION
# ==============================================================================
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,  # Offloaded to loading.py to bypass standard 3-ch verification checks
        std=None,   # Offloaded to loading.py to bypass standard 3-ch verification checks
        bgr_to_rgb=False,
        pad_size_divisor=32),

    backbone=dict(
        type='DualResNetSTRFound',  # Instantiates your custom STIT spatio-temporal backbone
        depth=50,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),

    # FPN input channels updated to standard non-concatenated additive sizes [256, 512, 1024, 2048]
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
        anchor_generator=dict(
            scales=[2, 4])),  # Anchor scale optimizations optimized specifically for tiny targets

    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True,
                 loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True,
                 loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True,
                 loss_bbox=dict(type='SAFitLoss', loss_weight=1.0))
        ])
)

# ==============================================================================
# 2. DATA PIPELINES (12-CHANNEL TEMPORAL PAIR SETUP)
# ==============================================================================
train_pipeline = [
    dict(type='LoadTemporalRGBTPair'),  # Custom loader yields [RGB_t, Th_t, RGB_t-1, Th_t-1]
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadTemporalRGBTPair'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# ==============================================================================
# 3. DATALOADERS & EVALUATORS
# ==============================================================================
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='data/rgbt_tiny/',
        metainfo=dict(classes=classes),
        ann_file='annotations/visible_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,  # Evaluation explicitly set to batch_size=1 for image-by-image AP precision maps
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/rgbt_tiny/',
        metainfo=dict(classes=classes),
        ann_file='annotations/visible_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/rgbt_tiny/annotations/visible_test.json',
    metric='bbox')

test_evaluator = val_evaluator

# ==============================================================================
# 4. OPTIMIZATION, SCHEDULE, AND GRADIENT ACCUMULATION
# ==============================================================================
# Synchronized single optimization block with 10GB VRAM-safe settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    accumulative_counts=2,        # 2 x batch_size 2 = stable effective batch size of 4!
    clip_grad=dict(max_norm=10, norm_type=2)
)

# Standard 1x schedules featuring linear warmup step Lifecycles
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),  # Extended stabilization warmup
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==============================================================================
# 5. UPGRADE 2: TCL HOOK INTERCEPTOR RUNTIME FOR TWO-STAGE LOSS
# ==============================================================================
# This custom intercept hook instantiates the contrastive loss component and patches
# the detector runtime execution flow dynamically without crashing basic MMDet modules.
custom_hooks = [
    dict(
        type='TCLFeatureInterceptHook',
        priority='NORMAL'
    )
]

# Instantiate your freshly added Temporal Contrastive Learning loss settings
tcl_loss_cfg = dict(type='TemporalContrastiveLoss', temperature=0.07, loss_weight=0.1)

# Output monitoring paths
work_dir = './work_dirs/rgbt_str_found_tcl_fortified'