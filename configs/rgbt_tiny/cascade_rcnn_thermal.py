# Inherit standard Cascade R-CNN blueprint properties dynamically
_base_ = ['../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py']

# Force MMDetection to load custom SAFit logic
custom_imports = dict(
    imports=['codes.loss.loss'],
    allow_failed_imports=False)

classes = ('ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane')

# 1. Overriding Model Parameters for Standard 3-Channel Thermal Inputs
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[2, 4])),  # Scale optimization for tiny objects
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True, loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True, loss_bbox=dict(type='SAFitLoss', loss_weight=1.0)),
            dict(type='Shared2FCBBoxHead', num_classes=7, reg_decoded_bbox=True, loss_bbox=dict(type='SAFitLoss', loss_weight=1.0))
        ])
)

# 2. Dataset Layout & Replicated Thermal Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True), # Automatically reads as 3-channel BGR/RGB
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='Resize', scale=(640, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# 1. Drop Batch Size to 8 to avoid OOM
train_dataloader = dict(
    batch_size=8,   # Dropped from 16 for VRAM safety
    num_workers=4,   # Adjusted to match the new batch size smoothly
    dataset=dict(
        data_root='data/rgbt_tiny/',
        metainfo=dict(classes=classes),
        ann_file='annotations/thermal_train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root='data/rgbt_tiny/',
        metainfo=dict(classes=classes),
        ann_file='annotations/thermal_test.json',  # Swapped to thermal annotations
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Evaluator tracking target thermal metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/rgbt_tiny/annotations/thermal_test.json',
    metric='bbox')
test_evaluator = val_evaluator

# 2. Adjust Learning Rate Linearly
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.010, momentum=0.9, weight_decay=0.0001), # Adjusted (0.020 / 2)
    clip_grad=dict(max_norm=35, norm_type=2)
)

work_dir = './work_dirs/cascade_rcnn_thermal'