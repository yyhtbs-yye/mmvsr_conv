
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .frvsr_utils import backward_warp, space_to_depth
from .frvsr_modules import FNet, SRNet


class FRVSRNet(nn.Module):
    """ Frame-recurrent network: https://arxiv.org/abs/1801.04590
    Implemented in TecoGAN
    """

    def __init__(self, in_channels, out_channels, mid_channels, n_blocks, scale):
        super(FRVSRNet, self).__init__()

        self.scale = scale

        # get upsampling function according to degradation type
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # define fnet & srnet
        self.fnet = FNet(in_channels)
        self.srnet = SRNet(in_channels, out_channels, mid_channels, n_blocks, self.upsample_func, self.scale)

    def forward(self, lr_data):

        n, t, c, h, w = lr_data.size()
        hr_h, hr_w = h * self.scale, w * self.scale

        # Calculate LR optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, h, w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, h, w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # OUTPUT size: [n * (t-1), 2, h, w]

        # Upsample LR flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # Compute the first HR frame (with no context)
        hr_data = [self.srnet(lr_data[:, 0, ...],
                              torch.zeros(n, (self.scale**2) * c, h, w, 
                                          dtype=torch.float32,
                                          device=lr_data.device))]

        # compute the remaining HR data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(lr_data[:, i, ...], space_to_depth(hr_prev_warp, self.scale))

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # OUTPUT size = [n, t, c, hr_h, hr_w]

        return hr_data
