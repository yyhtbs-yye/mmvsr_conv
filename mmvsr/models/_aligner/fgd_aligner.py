import torch
import torch.nn as nn

from mmengine.model.weight_init import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

class FirstOrderDeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2*self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 36*self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self): # Init constant offset

        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, unalign_feat, rough_aligned_feat, flows): # Forward function

        flow = flows[0]
        
        rough_aligned_feat = torch.cat([rough_aligned_feat, flow], dim=1)
        out = self.conv_offset(rough_aligned_feat)
        offset, mask = torch.chunk(out, 2, dim=1)

        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(unalign_feat, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)

# ------- # **Deprecated
class SecondOrderDeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flows):
        flow_1, flow_2 = flows[0], flows[1]
        
        """Forward function."""
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))

        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)

class AnyOrderDeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.order = kwargs.pop('order', 2)

        kwargs['deform_groups'] *= self.order

        super(AnyOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((self.order + 1) * self.out_channels + 2 * self.order, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        ) # 27 = 9(x_offset) + 9(y_offset) + 9(mask)

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flows):
        """Forward function."""
        extra_feat = torch.cat([extra_feat, *flows], dim=1)
        out = self.conv_offset(extra_feat)
        offsets, mask = torch.split(out, [18 * self.deform_groups, 9 * self.deform_groups], dim=1)

        # offset
        offsets = self.max_residue_magnitude * torch.tanh(offsets)
        
        offsets = torch.chunk(offsets, self.order, dim=1)
        out_offsets = []

        for i, flow in enumerate(flows):
            out_offsets.append(offsets[i] + flow.flip(1).repeat(1, offsets[i].size(1) // 2, 1, 1))

        offset = torch.cat(out_offsets, dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)
