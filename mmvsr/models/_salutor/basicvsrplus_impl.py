from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.registry import MODELS

from mmvsr.models._motion_estimator.spynet import  SPyNet
from mmvsr.models._aligner.fgd_aligner import AnyOrderDeformableAlignment as Aligner
from mmvsr.models._feature_extractor.conv_module import ResidualBlocksWithInputConv as Extractor
from mmvsr.models._temporal_propagator.recurrent import FirstOrderUnidirectionalRecurrentPropagator as Propagator

from mmvsr.models._upsampler.conv_module import BasicVSRUpsampler
from mmvsr.models._spatial_processor.conv_module import BasicVSRPlusPlusSpatial

@MODELS.register_module()
class BasicVSRPlusImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None, max_residue_magnitude=10):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        self.preproc = BasicVSRPlusPlusSpatial(in_channels=3, mid_channels=mid_channels)

        self.fextor_l1_args = dict(in_channels=2*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l2_args = dict(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l3_args = dict(in_channels=4*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l4_args = dict(in_channels=5*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)

        self.aligner_args = dict(in_channels=mid_channels, out_channels=mid_channels, 
                                 kernel_size=3, padding=1, deform_groups=16,
                                 max_residue_magnitude=max_residue_magnitude, order=1)

        self.propagator_l1 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l1_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args, is_reversed=True)
        self.propagator_l2 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l2_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args)
        self.propagator_l3 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l3_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args, is_reversed=True)
        self.propagator_l4 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l4_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args)

        self.upsample = BasicVSRUpsampler(in_channels=5*mid_channels, mid_channels=64)

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
        flows = self.compute_flow(lrs)

        feats_ = self.preproc(lrs)

        feats_l1 = self.propagator_l1(feats_, flows, [])
        feats_l2 = self.propagator_l2(feats_l1, flows, [feats_])
        feats_l3 = self.propagator_l3(feats_l2, flows, [feats_, feats_l1])
        feats_l4 = self.propagator_l4(feats_l3, flows, [feats_, feats_l1, feats_l2])

        return self.upsample(torch.cat([feats_, feats_l1, feats_l2, feats_l3, feats_l4], dim=-3), lrs)
