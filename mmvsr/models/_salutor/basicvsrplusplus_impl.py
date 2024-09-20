from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.registry import MODELS

from mmvsr.models._motion_estimator.spynet import  SPyNet
from mmvsr.models._temporal_propagator.second_order_recurrent import SecondOrderRecurrentPropagator, ResidualBlocksWithInputConv
from mmvsr.models._aligner.fgd_aligner import SecondOrderDeformableAlignment
from mmvsr.models.archs import PixelShufflePack, ResidualBlockNoBN

@MODELS.register_module()
class BasicVSRPlusPlusImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, max_residue_magnitude=10, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        aligner_def = SecondOrderDeformableAlignment
        aligner_args = dict(in_channels=2*mid_channels, out_channels=mid_channels, 
                                kernel_size=3, padding=1, deform_groups=16,
                                max_residue_magnitude=max_residue_magnitude)
        fextor_def = ResidualBlocksWithInputConv
        fextor_args_b1 = dict(in_channels=2*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        fextor_args_f1 = dict(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        fextor_args_b2 = dict(in_channels=4*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        fextor_args_f2 = dict(in_channels=5*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)

        self.spatial_fextor = ResidualBlocksWithInputConv(3, mid_channels, 5)

        # Recurrent propagators
        self.backward_propagator1 = SecondOrderRecurrentPropagator(mid_channels, 
                                                                   aligner_def=aligner_def,
                                                                   aligner_args=aligner_args,
                                                                   fextor_def=fextor_def,
                                                                   fextor_args=fextor_args_b1,
                                                                   is_reversed=True)
        self.forward_propagator1  = SecondOrderRecurrentPropagator(mid_channels, 
                                                                   aligner_def=aligner_def,
                                                                   aligner_args=aligner_args,
                                                                   fextor_def=fextor_def,
                                                                   fextor_args=fextor_args_f1,)
        self.backward_propagator2 = SecondOrderRecurrentPropagator(mid_channels, 
                                                                   aligner_def=aligner_def,
                                                                   aligner_args=aligner_args,
                                                                   fextor_def=fextor_def,
                                                                   fextor_args=fextor_args_b2,
                                                                   is_reversed=True)
        self.forward_propagator2  = SecondOrderRecurrentPropagator(mid_channels, 
                                                                   aligner_def=aligner_def,
                                                                   aligner_args=aligner_args,
                                                                   fextor_def=fextor_def,
                                                                   fextor_args=fextor_args_f2,)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
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

        feats_ = self.spatial_fextor(lrs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)

        # compute optical flow
        forward_flows, backward_flows = self.compute_flow(lrs)

        feats1 = self.backward_propagator1(feats_, backward_flows, [])

        feats2 = self.forward_propagator1(feats1, forward_flows, [feats_])

        feats3 = self.backward_propagator2(feats2, backward_flows, [feats_, feats1])

        feats4 = self.forward_propagator2(feats3, forward_flows, [feats_, feats1, feats2])

        return self.upsample(lrs, [feats_, feats1, feats2, feats3, feats4])

    def upsample(self, lrs, feats):

        outputs = []

        for i in range(0, lrs.size(1)):
            hr = torch.cat([it[:, i, ...] for it in feats], dim=1)
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lrs[:, i, :, :, :])

            outputs.append(hr)

        return torch.stack(outputs, dim=1)


if __name__ == '__main__':
    tensor_filepath = "/workspace/mmvsr/test_input_tensor.pt"
    input_tensor = torch.load('test_input_tensor.pt') / 100
    model = BasicVSRPlusPlusImpl(mid_channels=4, num_blocks=1, spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
                     'basicvsr/spynet_20210409-c6c1bd09.pth')

    output1 = model(input_tensor)

