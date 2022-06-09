_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'  # '../_base_/schedules/schedule_1x.py',
]

# model settings
model = dict(
    neck=dict(
        type='FPN_CSA',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        extra_convs_on_inputs=True,  # use P5
        num_outs=5),
    rpn_head=dict(anchor_generator=dict(strides=[8, 16, 32, 64, 128])),
    roi_head=dict(
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64])))


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

# optimizer
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=12)
