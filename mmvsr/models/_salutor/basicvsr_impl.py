from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.registry import MODELS

from mmvsr.models._motion_estimator.spynet import  SPyNet
from mmvsr.models._temporal_propagator.first_order_recurrent import FirstOrderRecurrentPropagator, ResidualBlocksWithInputConv, Alignment

from mmvsr.models.archs import PixelShufflePack, ResidualBlockNoBN

@MODELS.register_module()
class BasicVSRImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        # Recurrent propagators
        self.backward_propagator = FirstOrderRecurrentPropagator(mid_channels, is_reversed=True)
        self.forward_propagator = FirstOrderRecurrentPropagator(mid_channels)

        self.aligner = Alignment()
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

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
        final_prop_feats = []
        backward_prop_feats = []
        forward_prop_feats = []

        backward_prop_feats = self.backward_propagator(lrs, backward_flows, [])

        # feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        # feat_indices = list(range(-1, -t - 1, -1))
        # for i in range(t):
        #     if i > 0:  # no warping required for the last timestep
        #         flow = backward_flows[:, feat_indices[i - 1], :, :, :]
        #         feat_prop = self.aligner(feat_prop, flow.permute(0, 2, 3, 1))

        #     feat_prop = torch.cat([lrs[:, feat_indices[i], :, :, :], feat_prop], dim=1)
        #     feat_prop = self.backward_resblocks(feat_prop)

        #     backward_prop_feats.append(feat_prop)
        # backward_prop_feats = backward_prop_feats[::-1]

        forward_prop_feats = self.forward_propagator(lrs, forward_flows, [])

        # feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        # feat_indices = list(range(0, t, 1))
        # for i in range(t):
        #     if i > 0:  # no warping required for the first timestep
        #         flow = forward_flows[:, feat_indices[i - 1], :, :, :]
        #         feat_prop = self.aligner(feat_prop, flow.permute(0, 2, 3, 1))

        #     feat_prop = torch.cat([lrs[:, feat_indices[i], :, :, :], feat_prop], dim=1)
        #     feat_prop = self.forward_resblocks(feat_prop)

        #     forward_prop_feats.append(feat_prop)

        for i in range(t):

            # Extract features and low-resolution images for the current time step
            current_feats = torch.cat((backward_prop_feats[i], forward_prop_feats[i]), dim=1) 
            current_lrs = lrs[:, i, :, :, :]

            # Apply the fusion layer on concatenated features and low-resolution images
            x = self.lrelu(self.fusion(current_feats))

            # Perform sequential upsampling
            x = self.lrelu(self.upsample1(x))
            x = self.lrelu(self.upsample2(x))
            x = self.lrelu(self.conv_hr(x))
            x = self.conv_last(x)

            # Upsample the low-resolution image for residual learning and add to the final high-resolution output
            upsampled_lrs = self.img_upsample(current_lrs)
            x += upsampled_lrs

            # Append the result for this time step
            final_prop_feats.append(x)

        return torch.stack(final_prop_feats, dim=1)


if __name__ == '__main__':
    tensor_filepath = "/workspace/mmvsr/test_input_tensor.pt"
    input_tensor = torch.load('test_input_tensor.pt') / 100
    model = BasicVSRImpl(mid_channels=4, num_blocks=1, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

    output1 = model(input_tensor)

