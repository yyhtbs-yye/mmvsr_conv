
import torch
import torch.nn.functional as F
from torch import nn

from mmvsr.models.utils import default_init_weights
from mmcv.ops import DeformConv2d, DeformConv2dPack, deform_conv2d

class ResidualBlockNoBN(nn.Module):

    def __init__(self, n_channels=64, kernel_size=3):
        super().__init__()
        self.res_scale = 1.0
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, 1, kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, 1, kernel_size//2, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self) -> None:

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class GuidedDeformConv2dPack(DeformConv2d): # from TDAN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, offset):
        """Forward function."""
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)
