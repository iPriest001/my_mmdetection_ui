_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/dior_detection.py',
    '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='FPN_CSA',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5)),
    roi_head=dict(
        bbox_head=dict(num_classes=20))
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)