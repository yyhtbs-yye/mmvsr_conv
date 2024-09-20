import torch
import torch.nn.functional as F
from torch import nn

from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair

from einops import rearrange

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import ConvModule

class YyeAlignment(nn.Module):
    def __init__(self, n_channels, deform_groups):
        super(YyeAlignment, self).__init__()

        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)

        self.offset_channels = deform_groups * 2 * 3 * 3
        self.deform_groups = deform_groups
        self.n_channels = n_channels

        self.downsample = nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=False)

        self.offset_conv_a_3 = ConvModule(n_channels * 2, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.offset_conv_b_3 = ConvModule(n_channels, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.offset_conv_a_2 = ConvModule(n_channels * 2, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.offset_conv_b_2 = ConvModule(n_channels * 2, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.offset_conv_c_2 = ConvModule(n_channels, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.dcn_pack_2 = ModulatedDCNPack(n_channels, n_channels, 3,
                                           padding=1, deform_groups=deform_groups)
        self.feat_conv_b_2 = ConvModule(n_channels * 2, n_channels, 3,
                                        padding=1, act_cfg=act_cfg)
        self.offset_conv_a_1 = ConvModule(n_channels * 2, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.offset_conv_b_1 = ConvModule(n_channels * 2, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.offset_conv_c_1 = ConvModule(n_channels, n_channels, 3, 
                                          padding=1, act_cfg=act_cfg)
        self.dcn_pack_1 = ModulatedDCNPack(n_channels, n_channels, 3,
                                           padding=1, deform_groups=deform_groups)
        self.feat_conv_b_1 = ConvModule(n_channels * 2, n_channels, 3,
                                        padding=1, act_cfg=None)

        self.dcn_pack_f = ModulatedDCNPack(n_channels, n_channels, 3,
                                           padding=1, deform_groups=deform_groups)

        # Cascading DCN (Maybe removed for fair testing)
        self.cas_offset_conv_a = ConvModule(n_channels * 2, n_channels, 3, 
                                           padding=1, act_cfg=act_cfg)
        self.cas_offset_conv_b = ConvModule(n_channels, n_channels, 3, 
                                           padding=1, act_cfg=act_cfg)
        self.cas_dcnpack = ModulatedDCNPack(n_channels, n_channels, 3, 
                                            padding=1, deform_groups=deform_groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
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

                c_offset2 = self.offset_conv_a_2(torch.cat([feat_align_l2, feat_center_l2], dim=1))
                
                offset2 = self.offset_conv_b_2(torch.cat([c_offset2, u_offset3], dim=1))           # Compute offset for level 2
                
                offset2 = self.offset_conv_c_2(offset2) + u_offset3 + c_offset2

                u_offset2 = self.upsample(offset2) * 2          # Upsample offset from level 2 to level 1 size

                # Level 1 Offset Compute ``offset1``
                feat_neig_l1 = x1[:, i, :, :, :].contiguous()

                feat_align_l1 = self.dcn_pack_1(feat_neig_l1, u_offset2)                                    # Align level 2 features using the offset from level 3

                c_offset1 = self.offset_conv_a_1(torch.cat([feat_align_l1, feat_center_l1], dim=1))
                
                offset1 = self.offset_conv_b_1(torch.cat([c_offset1, u_offset2], dim=1))           # Compute offset for level 2

                offset1 = self.offset_conv_c_1(offset1) + u_offset2 + c_offset1

                # Final Output
                feat_align_f = self.dcn_pack_f(feat_neig_l1, offset1)                         # Align the feature using the final offset

                # Cascading
                offset = torch.cat([feat_align_f, feat_center_l1], dim=1)
                offset = self.cas_offset_conv_b(self.cas_offset_conv_a(offset))
                out_feat.append(self.lrelu(self.cas_dcnpack(feat_align_f, offset)))

        return torch.stack(out_feat, dim=1)                                                 # Stack the aligned features along a new dimension

class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        """Forward function."""
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


if __name__ == "__main__":

    from torch.profiler import profile
    import pyprof
    pyprof.init()

    n_channels = 64  # Number of channels in the input
    deform_groups = 8  # Number of deformable groups
    
    model = MultiscaleAlignment(n_channels, deform_groups).cuda()
    
    # Create a test input tensor
    batch_size = 6
    temporal_dimension = 7  # Number of frames
    height, width = 64, 64  # Spatial dimensions
    input = torch.randn(batch_size, temporal_dimension, n_channels, height, width).cuda()
    
    # Run the model
    output = model(input)
