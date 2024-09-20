import torch
import torch.nn.functional as F
from torch import nn

from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair

from einops import rearrange

from .basic_blocks import ConvModule, ModulatedDCNPack

class SPyNetAlignment(nn.Module):
    def __init__(self, n_channels, deform_groups):
        super(SPyNetAlignment, self).__init__()

        self.offset_channels = deform_groups * 2 * 3 * 3
        self.deform_groups = deform_groups
        self.n_channels = n_channels

        self.downsample = nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.offset_conv_a_3 = ConvModule(n_channels * 2, n_channels, 3, padding=1)
        self.offset_conv_b_3 = ConvModule(n_channels, n_channels, 3, padding=1, no_acti=True)
        
        self.offset_conv_a_2 = ConvModule(n_channels * 3, n_channels, 3, padding=1)
        self.offset_conv_b_2 = ConvModule(n_channels, n_channels, 3, padding=1)
        self.offset_conv_c_2 = ConvModule(n_channels, n_channels, 3, padding=1, no_acti=True)
        
        self.offset_conv_a_1 = ConvModule(n_channels * 3, n_channels, 3, padding=1)
        self.offset_conv_b_1 = ConvModule(n_channels, n_channels, 3, padding=1)
        self.offset_conv_c_1 = ConvModule(n_channels, n_channels, 3, padding=1, no_acti=True)

        self.dcn_pack_2 = ModulatedDCNPack(n_channels, n_channels, 3, padding=1, deform_groups=deform_groups)
        self.dcn_pack_1 = ModulatedDCNPack(n_channels, n_channels, 3, padding=1, deform_groups=deform_groups)
        self.dcn_pack_f = ModulatedDCNPack(n_channels, n_channels, 3, padding=1, deform_groups=deform_groups)
        
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

        out_feat = [] # List to store aligned features at level 1

        # Coarse Alignment using Deformable Convolution
        for i in range(0, t):
            if i == t // 2:
                # Append the center frame as-is for the middle frame
                out_feat.append(feat_center_l1)
            else:

                # Level 3 Offset Compute ``offset3``
                feat_neig_l3 = x3[:, i, :, :, :].contiguous()

                offset3 = self.offset_conv_a_3(torch.cat([feat_neig_l3, feat_center_l3], dim=1))

                offset3 = self.offset_conv_b_3(offset3)

                u_offset3 = self.upsample(offset3) * 2          # Upsample offset from level 3 to level 2 size

                # Level 2 Offset Compute ``offset2``
                feat_neig_l2 = x2[:, i, :, :, :].contiguous()

                feat_align_l2 = self.dcn_pack_2(feat_neig_l2, u_offset3)                                    # Align level 2 features using the offset from level 3

                offset2 = self.offset_conv_a_2(torch.cat([feat_align_l2, feat_center_l2, u_offset3], dim=1))
                
                offset2 = self.offset_conv_b_2(offset2)           # Compute offset for level 2
                
                offset2 = self.offset_conv_c_2(offset2) + u_offset3

                u_offset2 = self.upsample(offset2) * 2          # Upsample offset from level 2 to level 1 size

                # Level 1 Offset Compute ``offset1``
                feat_neig_l1 = x1[:, i, :, :, :].contiguous()

                feat_align_l1 = self.dcn_pack_1(feat_neig_l1, u_offset2)                                    # Align level 2 features using the offset from level 3

                offset1 = self.offset_conv_a_1(torch.cat([feat_align_l1, feat_center_l1, u_offset2], dim=1))
                
                offset1 = self.offset_conv_b_1(offset1)           # Compute offset for level 2

                offset1 = self.offset_conv_c_1(offset1) + u_offset2

                # Final Output
                feat_align_f = self.dcn_pack_f(feat_neig_l1, offset1)                         # Align the feature using the final offset

                out_feat.append(feat_align_f)

        return torch.stack(out_feat, dim=1)                                                 # Stack the aligned features along a new dimension

