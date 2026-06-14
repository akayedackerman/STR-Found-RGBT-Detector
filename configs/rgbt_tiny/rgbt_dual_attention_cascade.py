# Inherit standard Cascade R-CNN blueprint properties dynamically
_base_ = ['../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py']

custom_imports = dict(
    imports=[
        'codes.loss.loss',
        'codes.pipeline.loading',
        'codes.models.resnet6ch',
        'codes.models.dual_resnet_dual_attn' # Registry target module file mapping
    ],
    allow_failed_imports=False)

classes = ('ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane')

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='DualResNetDualAttn', # Final optimized double-attention backbone class
        depth=50,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048, 4096],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[2, 4])),
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True, loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True, loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True, loss_bbox=dict(type='SAFitLoss', loss_weight=1.0))
        ])
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadThermalImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='ConcatThermalToRGB'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadThermalImageFromFile'),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ConcatThermalToRGB'),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root='data/rgbt_tiny/',
        metainfo=dict(classes=classes),
        ann_file='annotations/visible_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
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

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.010, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)

work_dir = './work_dirs/rgbt_dual_attention_cascade'