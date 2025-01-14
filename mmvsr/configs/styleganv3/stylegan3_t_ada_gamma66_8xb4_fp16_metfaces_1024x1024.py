# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.ffhq_flip import *
    from .._base_.gen_default_runtime import *
    from .._base_.models.base_styleganv3 import *

from torch.optim import Adam

from mmvsr.engine.hooks.visualization_hook import VisualizationHook
from mmvsr.evaluation.metrics.fid import FrechetInceptionDistance
from mmvsr.models.base_models.average_model import ExponentialMovingAverage
from mmvsr.models.base_models.base_gan import BaseGAN
from mmvsr.models.editors.stylegan2.stylegan2_discriminator import (
    ADAAug, ADAStyleGAN2Discriminator)
from mmvsr.models.editors.stylegan3.stylegan3_modules import SynthesisNetwork

synthesis_cfg = {
    'type': SynthesisNetwork,
    'channel_base': 32768,
    'channel_max': 512,
    'magnitude_ema_beta': 0.999
}
r1_gamma = 6.6  # set by user
d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

load_from = 'https://download.openmmlab.com/mmediting/stylegan3/stylegan3_t_ffhq_1024_b4x8_cvt_official_rgb_20220329_235113-db6c6580.pth'  # noqa
# ada settings
aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}

ema_half_life = 10.  # G_smoothing_kimg

ema_kimg = 10
ema_nimg = ema_kimg * 1000
ema_beta = 0.5**(32 / max(ema_nimg, 1e-8))

ema_config = dict(
    type=ExponentialMovingAverage, interval=1, momentum=ema_beta, start_iter=0)

model.update(
    generator=dict(
        out_size=1024,
        img_channels=3,
        rgb2bgr=True,
        synthesis_cfg=synthesis_cfg),
    discriminator=dict(
        type=ADAStyleGAN2Discriminator,
        in_size=1024,
        input_bgr2rgb=True,
        data_aug=dict(type=ADAAug, aug_pipeline=aug_kwargs, ada_kimg=100)),
    loss_config=dict(r1_loss_weight=r1_gamma / 2.0 * d_reg_interval),
    ema_config=ema_config)

optim_wrapper.update(
    generator=dict(
        optimizer=dict(
            type=Adam, lr=0.0025 * g_reg_ratio, betas=(0, 0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type=Adam, lr=0.002 * d_reg_ratio, betas=(0, 0.99**d_reg_ratio))))

batch_size = 4
data_root = 'data/metfaces/images/'

train_dataloader.update(
    batch_size=batch_size, dataset=dict(data_root=data_root))
val_dataloader.update(batch_size=batch_size, dataset=dict(data_root=data_root))
test_dataloader.update(
    batch_size=batch_size, dataset=dict(data_root=data_root))
train_cfg.update(max_iters=160000)

# VIS_HOOK
custom_hooks = [
    dict(
        type=VisualizationHook,
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type=BaseGAN, name='fake_img')
    )  # vis_kwargs_list=dict(type='GAN', name='fake_img'))
]

# METRICS
metrics = [
    dict(
        type=FrechetInceptionDistance,
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='ema')
]
# NOTE: config for save multi best checkpoints
# default_hooks = dict(
#     checkpoint=dict(
#         save_best=['FID-Full-50k/fid', 'IS-50k/is'],
#         rule=['less', 'greater']))

default_hooks.update(checkpoint=dict(save_best='FID-Full-50k/fid'))
val_evaluator.update(metrics=metrics)
test_evaluator.update(metrics=metrics)
