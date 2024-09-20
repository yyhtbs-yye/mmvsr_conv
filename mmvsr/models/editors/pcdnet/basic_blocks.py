import torch
import torch.nn.functional as F
from torch import nn

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

from mmengine.model.weight_init import constant_init
from torch.nn.modules.utils import _pair

class LeakyHardTanh(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyHardTanh, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x > 1, self.negative_slope * (x - 1) + 1,
                           torch.where(x < -1, self.negative_slope * (x + 1) - 1, x))

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, negative_slope=0.01, no_acti=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = LeakyHardTanh(negative_slope)
        self.no_acti = no_acti

    def forward(self, x):
        x = self.conv(x)
        if self.no_acti: 
            return x
        x = self.activation(x)
        return x

class ModulatedDCNPack(ModulatedDeformConv2d):

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
