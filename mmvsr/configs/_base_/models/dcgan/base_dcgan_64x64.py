# Copyright (c) OpenMMLab. All rights reserved.
from mmvsr.models import DataPreprocessor
from mmvsr.models.editors import DCGAN
from mmvsr.models.editors.dcgan import DCGANDiscriminator, DCGANGenerator

# define GAN model
model = dict(
    type=DCGAN,
    noise_size=100,
    data_preprocessor=dict(type=DataPreprocessor),
    generator=dict(type=DCGANGenerator, output_scale=64, base_channels=1024),
    discriminator=dict(
        type=DCGANDiscriminator,
        input_scale=64,
        output_scale=4,
        out_channels=1))
