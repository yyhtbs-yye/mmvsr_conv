# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .adm_ddim250_8xb32_imagenet_512x512 import *

from mmvsr.evaluation.metrics import FrechetInceptionDistance
from mmvsr.models.editors.guided_diffusion.classifier import EncoderUNetModel

model.update(
    dict(
        classifier=dict(
            type=EncoderUNetModel,
            image_size=512,
            in_channels=3,
            model_channels=128,
            out_channels=1000,
            num_res_blocks=2,
            attention_resolutions=(16, 32, 64),
            channel_mult=(0.5, 1, 1, 2, 2, 4, 4),
            use_fp16=False,
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            pool='attention')))

metrics = [
    dict(
        type=FrechetInceptionDistance,
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig',
        sample_kwargs=dict(
            num_inference_steps=250, show_progress=True, classifier_scale=1.))
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
