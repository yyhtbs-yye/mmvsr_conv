
import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple
from mmcv.ops import DeformConv2d, DeformConv2dPack, deform_conv2d
from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair
from mmvsr.models.utils import default_init_weights

class ResBlockD2D(nn.Module):
    def __init__(self, mid_channels, deform_groups):
        super(ResBlockD2D, self).__init__()
        self.feat_aggregate = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1, bias=True),
            DeformConv2dPack(
                mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups),
            DeformConv2dPack(
                mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups),
        )
        self.align_1 = AugmentedDeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)
        self.align_2 = DeformConv2dPack(
            mid_channels, mid_channels, 3, padding=1, deform_groups=deform_groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lrs):

        b, t, c, h, w = lrs.size()

        lr_center = lrs[:, t // 2, :, :, :]  # LR center frame

        aligned_lrs = []

        for i in range(0, t):
            if i == t // 2:
                aligned_lrs.append(lr_center.unsqueeze(1))
            else:
                feat_neig = lrs[:, i, :, :, :].contiguous()

                feat_agg = torch.cat([lr_center, feat_neig], dim=1)
                feat_agg = self.feat_aggregate(feat_agg)

                aligned_feat = self.align_2(self.align_1(feat_neig, feat_agg))
                aligned_lrs.append(aligned_feat.unsqueeze(1))

        aligned_lrs = torch.cat(aligned_lrs, dim=1)

        return aligned_lrs

class AugmentedDeformConv2dPack(DeformConv2d): # from TDAN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
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
        offset = self.conv_offset(extra_feat)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)

class ResidualBlockNoBN(nn.Module):

    def __init__(self, mid_channels=64, kernel_size=3):
        super().__init__()
        self.res_scale = 1.0
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, kernel_size//2, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self) -> None:

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

if __name__ == "__main__":
    model = ResBlockD2D(mid_channels=64)
    scripted_model = torch.jit.script(model)
    print(scripted_model.graph)  # Optionally print the graph to see the transformed model

