import torch
from torch.nn import functional as F

from .optical_flow import flow_warp
    
def get_aligned_image_2frames(x, flows_backward, flows_forward):
    n = x.size(1)

    # Backward warping
    x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
    for i in range(n - 1, 0, -1):
        x_i = x[:, i, ...]
        flow = flows_backward[:, i - 1, ...]
        # frame i+1 aligned towards i
        x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 
                                       'nearest4'))  

    # Forward warping
    x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
    for i in range(0, n - 1):
        x_i = x[:, i, ...]
        flow = flows_forward[:, i, ...]
        # frame i-1 aligned towards i
        x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 
                                   'nearest4'))  

    return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

def get_aligned_feature_2frames(x, flows_backward, flows_forward, pa_deform):
        # backward
        n = x.size(1)
        x_backward = [torch.zeros_like(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
            x_backward.insert(0, pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        # forward
        x_forward = [torch.zeros_like(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
            x_forward.append(pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

