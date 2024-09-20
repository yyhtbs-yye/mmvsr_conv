from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.registry import MODELS

from mmvsr.models._motion_estimator.spynet import  SPyNet
from mmvsr.models._temporal_propagator.first_order_recurrent import FirstOrderRecurrentPropagator, ResidualBlocksWithInputConv
from mmvsr.models._upsampler.conv_module import Type1Upsampler

@MODELS.register_module()
class BasicVSRImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        # Recurrent propagators
        self.back_propagator = FirstOrderRecurrentPropagator(mid_channels, num_blocks, is_reversed=True)
        self.forward_propagator = FirstOrderRecurrentPropagator(mid_channels, num_blocks)

        self.feat_upsampler = Type1Upsampler(2 * mid_channels, mid_channels)

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

    def compute_flow(self, lrs):

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        forward_flows = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows

    def forward(self, lrs):

        n, t, c, h, w = lrs.size()

        # compute optical flow
        forward_flows, backward_flows = self.compute_flow(lrs)

        # According to BasicVSRNet:
        # `back_propagator` is done first, then `forward_propagator`. 

        # Run the forward propagation with inverted order of features
        backward_feats = self.back_propagator(lrs, backward_flows, [])

        # Run the backward and forward propagation
        forward_feats = self.forward_propagator(lrs, forward_flows, [])


        out = self.feat_upsampler(torch.cat((forward_feats, backward_feats), dim=-3), lrs)

        return out


if __name__ == '__main__':
    tensor_filepath = "/workspace/mmvsr/test_input_tensor.pt"
    input_tensor = torch.load('test_input_tensor.pt') / 100
    model = BasicVSRImpl(mid_channels=4, num_blocks=1, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

    output1 = model(input_tensor)

