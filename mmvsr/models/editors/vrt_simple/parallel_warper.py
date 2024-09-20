import torch
import torch.nn as nn

from .dcn_modules import DCNv2PackFlowGuided

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
    