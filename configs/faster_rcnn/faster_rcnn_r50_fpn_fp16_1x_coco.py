#  _base_ = './faster_rcnn_r50_fpn_1x_coco.py'
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# fp16 settings
fp16 = dict(loss_scale=512.)
