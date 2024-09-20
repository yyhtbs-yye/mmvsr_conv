# Copyright (c) OpenMMLab. All rights reserved.
from mmvsr.models import DataPreprocessor
from mmvsr.models.editors import SAGAN
from mmvsr.models.editors.biggan import SelfAttentionBlock
from mmvsr.models.editors.sagan import ProjDiscriminator, SNGANGenerator

model = dict(
    type=SAGAN,
    num_classes=1000,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(
        type=SNGANGenerator,
        output_scale=128,
        base_channels=64,
        attention_cfg=dict(type=SelfAttentionBlock),
        attention_after_nth_block=4,
        with_spectral_norm=True),
    discriminator=dict(
        type=ProjDiscriminator,
        input_scale=128,
        base_channels=64,
        attention_cfg=dict(type=SelfAttentionBlock),
        attention_after_nth_block=1,
        with_spectral_norm=True),
    generator_steps=1,
    discriminator_steps=1)
