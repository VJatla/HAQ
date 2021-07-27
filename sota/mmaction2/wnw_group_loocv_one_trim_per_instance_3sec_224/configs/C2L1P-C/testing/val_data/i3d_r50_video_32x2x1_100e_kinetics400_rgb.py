# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='ResNet3d',
                  pretrained2d=True,
                  pretrained='torchvision://resnet50',
                  depth=50,
                  conv_cfg=dict(type='Conv3d'),
                  norm_eval=False,
                  inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1,
                                                                         0)),
                  zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=2,  # vj
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'VideoDataset'
data_root = ''  # vj
data_root_val = 'data/wnw_one_trim_per_instance_3sec_224'  # vj
ann_file_train = ''  # vj
ann_file_val = ''  # vj
ann_file_test = 'data/wnw_one_trim_per_instance_3sec_224/group_leave_one_out/exp2/C2L1P-C/val.txt'  #vj
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='MultiScaleCrop',
         input_size=224,
         scales=(1, 0.8),
         random_crop=False,
         max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames',
         clip_len=32,
         frame_interval=2,
         num_clips=1,
         test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(videos_per_gpu=16,
            workers_per_gpu=4,
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
optimizer = dict(type='SGD', lr=0.01 / 8, momentum=0.9,
                 weight_decay=0.0001)  # this lr is used for 8 gpus (vj)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100  # 100 by default
checkpoint_config = dict(interval=3)  # vj
evaluation = dict(interval=1,
                  metrics=['top_k_accuracy', 'mean_class_accuracy'],
                  topk=(1, 5))  # vj
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])  # vj ---> interval
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '<not needed, given via argument>'  # vj
load_from = './checkpoints/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'  # vj
resume_from = None
workflow = [('train', 1)]
