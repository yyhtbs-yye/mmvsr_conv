import torch
import torch.nn.functional as F
from torch import nn

from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair

from einops import rearrange

from mmcv.ops import DeformConv2dPack

from .basic_modules import GuidedDeformConv2dPack

class PyramidDeformableAlignment(nn.Module):
    def __init__(self, n_channels, deform_groups):
        super(PyramidDeformableAlignment, self).__init__()

        self.offset_channels = deform_groups * 2 * 3 * 3
        self.deform_groups = deform_groups
        self.n_channels = n_channels

        self.downsample = nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=False)

        self.offset_l1 = nn.Sequential(
            nn.Conv2d(self.n_channels*2, self.n_channels, 
                      kernel_size=7, padding=3),
            DeformConv2dPack(self.n_channels, self.n_channels, 
                             3, padding=1, deform_groups=8),
            DeformConv2dPack(self.n_channels, self.offset_channels, 
                             3, padding=1, deform_groups=8),
                )
        
        self.offset_l2 = nn.Sequential(
            nn.Conv2d(self.n_channels*2, self.n_channels, 
                      kernel_size=5, padding=2),
            DeformConv2dPack(self.n_channels, self.n_channels, 
                             3, padding=1, deform_groups=8),
            DeformConv2dPack(self.n_channels, self.offset_channels, 
                             3, padding=1, deform_groups=8),
                )
        
        self.offset_l3 = nn.Sequential(
            nn.Conv2d(self.n_channels*2, self.n_channels, 
                      kernel_size=3, padding=1),
            DeformConv2dPack(self.n_channels, self.n_channels, 
                             3, padding=1, deform_groups=8),
            DeformConv2dPack(self.n_channels, self.offset_channels, 
                             3, padding=1, deform_groups=8),
                )

        self.align_l2 = GuidedDeformConv2dPack(n_channels, n_channels, 
                                               kernel_size=3, padding=1, 
                                               deform_groups=deform_groups)
        
        self.align_l1 = GuidedDeformConv2dPack(n_channels, n_channels, 
                                               kernel_size=3, padding=1, 
                                               deform_groups=deform_groups)

        self.align_fi = GuidedDeformConv2dPack(n_channels, n_channels, 
                                               kernel_size=3, padding=1, 
                                               deform_groups=deform_groups)

        self.post_deform = DeformConv2dPack(n_channels, n_channels, 
                                            kernel_size=3, padding=1, 
                                            deform_groups=deform_groups)

        constant_init(self.offset_l1, val=0, bias=0)
        constant_init(self.offset_l2, val=0, bias=0)
        constant_init(self.offset_l3, val=0, bias=0)

    def forward(self, x):
        # x1: level 1, original spatial size
        # x2: level 2, 1/2 spatial size
        # x3: level 3, 1/4 spatial size

        x1 = x
        x2 = self.downsample(x1) # Downsample x1 to half its spatial dimensions (1/2 of original)
        x3 = self.downsample(x2) # Downsample x2 to half its spatial dimensions (1/4 of original)

        b, t, c, h, w = x1.shape  # Extract shape of x1 to variables: batch size, number of frames, channels, height, width

        # Extract center frame features for each level
        feat_center_l3 = x3[:, t // 2, :, :, :]  # Feature at the exact middle frame from level 3
        feat_center_l2 = x2[:, t // 2, :, :, :]  # Feature at the exact middle frame from level 2
        feat_center_l1 = x1[:, t // 2, :, :, :]  # Feature at the exact middle frame from level 1

        aligned_feats_l1 = [] # List to store aligned features at level 1

        # Coarse Alignment using Deformable Convolution
        for i in range(0, t):
            if i == t // 2:
                # Append the center frame as-is for the middle frame
                aligned_feats_l1.append(feat_center_l1)
            else:

                # Level 3 Offset Compute
                feat_neig_l3 = x3[:, i, :, :, :].contiguous()
                offset_l3 = self.offset_l3(torch.cat([feat_neig_l3, feat_center_l3], dim=1))        # Compute offset for level 3
                offset_l3_us = self.upsample(offset_l3) * 2                                         # Upsample level 3 offset from level 3 back to level 2 size

                # Level 2 Offset Compute
                feat_neig_l2 = x2[:, i, :, :, :].contiguous()
                z_l2 = self.align_l2(feat_neig_l2, offset_l3_us)                                    # Align level 2 features using the offset from level 3

                offset_l2 = self.offset_l2(torch.cat([z_l2, feat_center_l2], dim=1)) + offset_l3_us # Compute offset for level 2
                offset_l2_us = self.upsample(offset_l2) * 2                                         # Upsample level 3offset from level 2 back to level 1 size

                # Level 1 Offset Compute
                feat_neig_l1 = x1[:, i, :, :, :].contiguous()
                z_l1 = self.align_l1(feat_neig_l1, offset_l2_us)                                    # Align level 1 features using the offset from level 2

                offset_l1 = self.offset_l1(torch.cat([z_l1, feat_center_l1], dim=1)) + offset_l2_us # Compute final offset for level 1
                aligned_feat_l1 = self.align_fi(feat_neig_l1, offset_l1)                         # Align the feature using the final offset

                # Final Output Append
                aligned_feats_l1.append(self.post_deform(aligned_feat_l1))                                            # Align the feature using the final offset

        return torch.stack(aligned_feats_l1, dim=1)                                                 # Stack the aligned features along a new dimension
