# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(type='ResNet',
                  pretrained='torchvision://resnet50',
                  depth=50,
                  norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=2,  # changed vj
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips=None)
# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/wnw_one_trim_per_instance_3sec_224'  # vj
data_root_val = 'data/wnw_one_trim_per_instance_3sec_224'  # vj
ann_file_train = 'data/wnw_one_trim_per_instance_3sec_224/group_leave_one_out/exp2/C2L1P-B/trn.txt'  # vj
ann_file_val = 'data/wnw_one_trim_per_instance_3sec_224/group_leave_one_out/exp2/C2L1P-B/val.txt'  # vj
ann_file_test = ''  #vj
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='MultiScaleCrop',
         input_size=224,
         scales=(1, 0.875, 0.75, 0.66),
         random_crop=False,
         max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=1,
         frame_interval=1,
         num_clips=8,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=1,
         frame_interval=1,
         num_clips=25,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=
    16,  # changed by vj (original = 32, have to adjust lr accordingly)
    workers_per_gpu=2,  # changed by vj (original = 4)
    train=dict(type=dataset_type,
               ann_file=ann_file_train,
               data_prefix=data_root,
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=ann_file_val,
             data_prefix=data_root_val,
             pipeline=val_pipeline),
    test=dict(type=dataset_type,
              ann_file=ann_file_test,
              data_prefix=data_root_val,
              pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD',
                 lr=0.01 / (8 * 2),
                 momentum=0.9,
                 weight_decay=0.0001)  # 0.0001 lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100 # 100 by default
checkpoint_config = dict(interval=1)  # Changed by vj
evaluation = dict(interval=1,
                  metrics=['top_k_accuracy', 'mean_class_accuracy'],
                  topk=(1, 5))  # Changed by vj
log_config = dict(
    interval=1,  # changed by vj (originally 20)
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])  # Changed by vj
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/wnw_one_trim_per_instance_3sec_224/tsn/'  # Changed by vj
load_from = './checkpoints/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth'  # changed by vj
resume_from = None
workflow = [('train', 1)]
