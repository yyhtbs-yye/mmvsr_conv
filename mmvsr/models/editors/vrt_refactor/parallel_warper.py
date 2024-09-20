import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from torch.nn.modules.utils import _pair

from mmvsr.models.utils import flow_warp

class ParallelWarper(nn.Module):

    def __init__(self, n_channels, n_frames, 
                 deformable_groups, kernel_size, 
                 padding, max_residue_magnitude):
        
        super(ParallelWarper, self).__init__()

        # Optical Flow Guided DCNv2
        self.ofg_dcn = DCNv2PackFlowGuided(n_channels, n_channels, 
                                           n_frames=n_frames, 
                                           deformable_groups=deformable_groups, kernel_size=kernel_size, 
                                           padding=padding, max_residue_magnitude=max_residue_magnitude)

    def forward(self, x, flows_backward, flows_forward):
        n = x.size(1)
        
        # backward pass
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            processed_feature = self.ofg_dcn(x_i, [x_i_warped], x[:, i - 1, ...], [flow])
            x_backward.insert(0, processed_feature)

        # forward pass
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            processed_feature = self.ofg_dcn(x_i, [x_i_warped], x[:, i + 1, ...], [flow])

            x_forward.append(processed_feature)

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]
    
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

