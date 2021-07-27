model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(type='resnet3d',
                          depth=50,
                          pretrained=None,
                          lateral=True,
                          conv1_kernel=(1, 7, 7),
                          dilations=(1, 1, 1, 1),
                          conv1_stride_t=1,
                          pool1_stride_t=1,
                          inflate=(0, 0, 1, 1),
                          norm_eval=False),
        fast_pathway=dict(type='resnet3d',
                          depth=50,
                          pretrained=None,
                          lateral=False,
                          base_channels=8,
                          conv1_kernel=(5, 7, 7),
                          conv1_stride_t=1,
                          pool1_stride_t=1,
                          norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5))
train_cfg = None
test_cfg = dict(average_clips='prob')
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
    dict(type='RandomResizedCrop'),
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
data = dict(videos_per_gpu=8,
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
optimizer = dict(type='SGD', lr=0.1 / 8, momentum=0.9,
                 weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=34)
total_epochs = 256 # 256 by default
checkpoint_config = dict(interval=3)
workflow = [('train', 1)]
evaluation = dict(interval=1,
                  metrics=['top_k_accuracy', 'mean_class_accuracy'],
                  topk=(1, 5))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/kinetics_tiny/slowfast/'
load_from = 'checkpoints/slowfast_r50_video_4x16x1_256e_kinetics400_rgb_20200826-f85b90c5.pth'
resume_from = None
find_unused_parameters = False
