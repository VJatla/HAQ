model = dict(type='Recognizer3D',
             backbone=dict(type='ResNet3dSlowOnly',
                           depth=50,
                           pretrained=None,
                           lateral=False,
                           conv1_kernel=(1, 7, 7),
                           conv1_stride_t=1,
                           pool1_stride_t=1,
                           inflate=(0, 0, 1, 1),
                           norm_eval=False),
             cls_head=dict(type='I3DHead',
                           in_channels=2048,
                           num_classes=2,
                           spatial_type='avg',
                           dropout_ratio=0.5))
train_cfg = None
test_cfg = dict(average_clips='prob')
dataset_type = 'VideoDataset'
data_root = 'data/tynty_one_trim_per_instance_3sec_224'  # vj
data_root_val = 'data/tynty_one_trim_per_instance_3sec_224'  # vj
ann_file_train = 'data/tynty_one_trim_per_instance_3sec_224/group_leave_one_out/exp2/C1L1W-A/trn.txt'  # vj
ann_file_val = 'data/tynty_one_trim_per_instance_3sec_224/group_leave_one_out/exp2/C1L1W-A/val.txt'  # vj
ann_file_test = ''  #vj
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=4, frame_interval=16, num_clips=1),
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
         clip_len=4,
         frame_interval=16,
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
         clip_len=4,
         frame_interval=16,
         num_clips=10,
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
data = dict(videos_per_gpu=24,
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
optimizer = dict(type='SGD', lr=0.3 / 8, momentum=0.9,
                 weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
total_epochs = 256 # 256 by default
checkpoint_config = dict(interval=1)
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
work_dir = './work_dirs/kinetics_tiny/slowonly'
load_from = 'checkpoints/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth'
resume_from = None
find_unused_parameters = False
