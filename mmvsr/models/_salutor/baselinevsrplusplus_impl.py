from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.registry import MODELS

from mmvsr.models._motion_estimator.spynet import  SPyNet
from mmvsr.models._temporal_propagator.second_order_window import SecondOrderWindowPropagator, ResidualBlocksWithInputConv

from mmvsr.models.archs import PixelShufflePack, ResidualBlockNoBN

@MODELS.register_module()
class BaselineVSRPlusPlusImpl(BaseModule):

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
        fextor_args_f1 = dict(in_channels=2*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        fextor_args_b2 = dict(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        fextor_args_f2 = dict(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)

        self.spatial_fextor = ResidualBlocksWithInputConv(3, mid_channels, 5)

        # Sliding Window Propagators
        self.backward_propagator1 = SecondOrderWindowPropagator(mid_channels, 
                                                                aligner_def=aligner_def,
                                                                aligner_args=aligner_args,
                                                                fextor_def=fextor_def,
                                                                fextor_args=fextor_args_b1,
                                                                is_reversed=True)
        self.forward_propagator1  = SecondOrderWindowPropagator(mid_channels, 
                                                                aligner_def=aligner_def,
                                                                aligner_args=aligner_args,
                                                                fextor_def=fextor_def,
                                                                fextor_args=fextor_args_f1,)
        self.backward_propagator2 = SecondOrderWindowPropagator(mid_channels, 
                                                                aligner_def=aligner_def,
                                                                aligner_args=aligner_args,
                                                                fextor_def=fextor_def,
                                                                fextor_args=fextor_args_b2,
                                                                is_reversed=True)
        self.forward_propagator2  = SecondOrderWindowPropagator(mid_channels, 
                                                                aligner_def=aligner_def,
                                                                aligner_args=aligner_args,
                                                                fextor_def=fextor_def,
                                                                fextor_args=fextor_args_f2,)

        self.bifusor1 = nn.Conv2d(2*mid_channels, mid_channels, 1)
        self.bifusor2 = nn.Conv2d(2*mid_channels, mid_channels, 1)

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

        feats1b = self.backward_propagator1(feats_, backward_flows, [])

        feats1f = self.forward_propagator1(feats_, forward_flows, [])

        feats1 = self.bifusor1(torch.cat((feats1b, feats1f), dim=-3).view(-1, feats1b.shape[-3]*2, feats1b.shape[-2], feats1b.shape[-1]))

        feats1 = feats1.view(n, t, *feats1.shape[-3:])

        feats2b = self.backward_propagator2(feats1, backward_flows, [feats_])

        feats2f = self.forward_propagator2(feats1, forward_flows, [feats_])

        return self.upsample(lrs, [feats_, feats1b, feats1f, feats2b, feats2f])

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

from mmengine.model.weight_init import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

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

    def forward(self, x, extra_feat, flow_n12):
        flow_1, flow_2 = flow_n12[0], flow_n12[1]
        """Forward function."""
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)
