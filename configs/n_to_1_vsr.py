_base_ = [
    './_base_/default_runtime.py',
]

model_name = 'test'
model_configs = dict()
loss_name = 'CharbonnierLoss'
train_iter = 300_000
val_interval = 1_000
train_on_patch = False
gt_patch_size = 256
work_dir = f'./work_dirs/{model_name}'
save_dir = './work_dirs'
scale = 4

# ----------- Train Data Delivery
batch_size = 4
num_workers = 4

# 
num_input_frames = 7
# model settings
model = dict(
    type='NaiveVSR',
    generator=dict(
        type=f'{model_name}',
        **model_configs,
        ),
    pixel_loss=dict(type=loss_name, loss_weight=1.0, reduction='mean'),
    train_cfg=dict(fix_iter=train_iter),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    *( [dict(type='PairedRandomCrop', gt_patch_size=gt_patch_size)] if train_on_patch else [] ),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], frames_per_clip=99),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PackInputs')
]

reds_root = './data/REDS'

train_dataloader = dict(
    num_workers=num_workers,
    batch_size=batch_size,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        reds_root=reds_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_train.txt',
        depth=2,
        num_input_frames=num_input_frames,
        num_output_frames=1,
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=num_workers,
    batch_size=batch_size,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        reds_root=reds_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=2,
        num_input_frames=num_input_frames,
        num_output_frames=1,
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator', 
    metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

train_cfg = dict(type='IterBasedTrainLoop', max_iters=train_iter, val_interval=val_interval)
val_cfg = dict(type='MultiValLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)),
)

# learning policy
param_scheduler = dict(type='CosineRestartLR', by_epoch=False, 
                       periods=[300_000], restart_weights=[1], eta_min=1e-7)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,
        save_optimizer=True,
        out_dir=save_dir,
        by_epoch=False))