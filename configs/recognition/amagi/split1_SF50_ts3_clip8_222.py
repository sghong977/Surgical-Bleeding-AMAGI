speed_r, time_r, channel_r = 2,2,2
split = 2
depth=50

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='AMAGI',
        pretrained=None,
        resample_rate=speed_r,   #8,  # tau
        speed_ratio=speed_r, #8,  # alpha
        channel_ratio=8,  # beta_inv
        kfold=split,
        slow_pathway=dict(
            type='resnet3d',
            depth=depth,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='AmagiHead2',
        in_channels=2304,  # 2048+256
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5, #),
        channel_rate=channel_r,   #2
        speed_ratio=speed_r,
        time_rate=time_r,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0,class_weight=[0.6059698518933964, 2.859161549565014])),
    train_cfg = None,
    test_cfg = dict(average_clips='prob'))#, max_testing_views=8))

dataset_type = 'RawframeDataset'
data_root = 'rawframes path'
data_root_val = 'rawframes path'
ann_file_train = f'txt file that composed of lines of [folder path] [clip_len] [0 or 1 annotation]'
ann_file_val = f'txt file that composed of lines of [folder path] [clip_len] [0 or 1 annotation]'
ann_file_test = f'txt file that composed of lines of [folder path] [clip_len] [0 or 1 annotation]'
img_norm_cfg = dict(
    mean=[117.1168392375, 75.27494787, 67.37629650299999], std=[60.241352387999996, 51.261253263, 49.192591569], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=3, num_clips=1),
    dict(type='RawFrameDecode'),
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
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=4e-5)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=30)
total_epochs = 100
checkpoint_config = dict(interval=8)
workflow = [('train', 1)]
evaluation = dict(
    interval=10, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './results'
load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'  # noqa: E501
resume_from = None
find_unused_parameters = False
