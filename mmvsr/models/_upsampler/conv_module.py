# Copyright (c) Yuhang Ye. All rights reserved.
from logging import WARNING
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmvsr.models.archs import PixelShufflePack
from mmvsr.registry import MODELS

@MODELS.register_module()
class BasicVSRUpsampler(BaseModule): # BasicVSR Upsampler
    def __init__(self, in_channels, mid_channels=64):

        super().__init__()

        self.mid_channels = mid_channels

        # upsample
        self.fusion = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feats, lrs):
        
        B, T, C, H, W = feats.size()
        output = []

        for i in range(T):
            # Extract features and low-resolution images for the current time step
            current_feats = feats[:, i, :, :, :]
            current_lrs = lrs[:, i, :, :, :]

            # Apply the fusion layer on concatenated features and low-resolution images
            x = self.lrelu(self.fusion(current_feats))

            # Perform sequential upsampling
            x = self.lrelu(self.upsample1(x))
            x = self.lrelu(self.upsample2(x))
            x = self.lrelu(self.conv_hr(x))
            x = self.conv_last(x)

            # Upsample the low-resolution image for residual learning and add to the final high-resolution output
            upsampled_lrs = self.img_upsample(current_lrs)
            x += upsampled_lrs

            # Append the result for this time step
            output.append(x)

        # Stack along the time dimension
        return torch.stack(output, dim=1)
