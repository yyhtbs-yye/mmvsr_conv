import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

class DCNv2PackFlowGuided(ModulatedDeformConv2d):

    def __init__(self, in_channels, out_channels, 
                 kernel_size=1, stride=1, padding=0, 
                 dilation=1, groups=1, deformable_groups=1, 
                 bias=True, max_residue_magnitude=10, n_frames=2):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride, padding, dilation, groups, bias)

        self.max_residue_magnitude = max_residue_magnitude
        self.n_frames = n_frames

        self.deformable_groups = deformable_groups
        
        self.conv_offset_mask = nn.Sequential(
            nn.Conv2d((1 + self.n_frames // 2) * self.in_channels + self.n_frames, 
                      self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * 9 * self.deformable_groups, 3, 1, 1),
        )

        # Initialize offset and mask to zero.
        self.conv_offset_mask[-1].weight.data.zero_()
        self.conv_offset_mask[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset_mask(torch.cat(x_flow_warpeds + [x_current] + flows, dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.n_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)

        mask = torch.sigmoid(mask)
        
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deformable_groups)

