from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.registry import MODELS

from mmvsr.models._motion_estimator.spynet import  SPyNet

from mmvsr.models._aligner.fgd_aligner import AnyOrderDeformableAlignment as Aligner
from mmvsr.models._feature_extractor.conv_module import ResidualBlocksWithInputConv as Extractor
from mmvsr.models._temporal_propagator.feedforward import AnyOrderBidirectionalFullyConnectedPropagator as Propagator
from mmvsr.models._upsampler.conv_module import BasicVSRUpsampler
from mmvsr.models._spatial_processor.conv_module import BasicVSRPlusPlusSpatial

@MODELS.register_module()
class BorisVSRImpl(BaseModule):

    def __init__(self, mid_channels=64, num_blocks=30, spynet_pretrained=None, max_residue_magnitude=10):

        super().__init__()

        self.mid_channels = mid_channels
        self.num_blocks = num_blocks

        self.preproc = BasicVSRPlusPlusSpatial(in_channels=3, mid_channels=mid_channels)

        self.fextor_l1_args = dict(in_channels=2*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l2_args = dict(in_channels=3*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l3_args = dict(in_channels=4*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        self.fextor_l4_args = dict(in_channels=5*mid_channels, out_channels=mid_channels, num_blocks=num_blocks)

        self.aligner_args = dict(in_channels=6*mid_channels, out_channels=mid_channels, 
                                 kernel_size=3, padding=1, deform_groups=8,
                                 max_residue_magnitude=max_residue_magnitude, order=6)

        self.propagator_l1 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l1_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args)
        self.propagator_l2 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l2_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args)
        self.propagator_l3 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l3_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args)
        self.propagator_l4 = Propagator(mid_channels=mid_channels, 
                                        fextor_def=Extractor, fextor_args=self.fextor_l4_args,
                                        aligner_def=Aligner, aligner_args=self.aligner_args)

        self.upsample = BasicVSRUpsampler(in_channels=5*mid_channels, mid_channels=64)

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

    def compute_flow(self, lrs):

        n, t, c, h, w = lrs.size()

        # Initialize a flow matrix of zeros
        flow_matrix = torch.zeros(n, t, t, 2, h, w, device=lrs.device)

        # Compute optical flow for each pair of frames
        for i in range(t):
            for j in range(t):
                if i == j:
                    # Skip self-flow (optical flow between the same frame is zero)
                    continue
                
                # Prepare frame pairs
                ref = lrs[:, i, :, :, :]
                supp = lrs[:, j, :, :, :]
                
                # Compute flow from frame_i to frame_j
                flow_ij = self.spynet(ref, supp)  # Shape: (n, 2, h, w)
                
                # Store the computed flow in the flow matrix
                flow_matrix[:, i, j, :, :, :] = flow_ij

        return flow_matrix

    def compute_flow_fast(self, lrs):  # Needs more GPU ram than the vanilla approach

        n, t, c, h, w = lrs.size()

        # Reshape to create all frame pairs
        lrs_1 = lrs.unsqueeze(2).expand(-1, -1, t, -1, -1, -1)  # Shape: (n, t, t, c, h, w)
        lrs_2 = lrs.unsqueeze(1).expand(-1, t, -1, -1, -1, -1)  # Shape: (n, t, t, c, h, w)

        # Flatten the first two dimensions to compute the flows in a batch
        lrs_1 = lrs_1.reshape(-1, c, h, w)  # Shape: (n*t, c, h, w)
        lrs_2 = lrs_2.reshape(-1, c, h, w)  # Shape: (n*t, c, h, w)

        # Compute the flows in a batch using spynet
        flow_ij = self.spynet(lrs_1, lrs_2)  # Shape: (n*t*t, 2, h, w)

        # Reshape the flow matrix back to (n, t, t, 2, h, w)
        flow_matrix = flow_ij.view(n, t, t, 2, h, w)

        # Set diagonal elements to zero (self-flow)
        i_indices = torch.arange(t, device=lrs.device)
        flow_matrix[:, i_indices, i_indices, :, :, :] = 0

        return flow_matrix


    def forward(self, lrs):

        n, t, c, h, w = lrs.size()

        # compute optical flow
        matrix_flows = self.compute_flow(lrs)

        feats_ = self.preproc(lrs)

        feats_l1 = self.propagator_l1(feats_, matrix_flows, [])
        feats_l2 = self.propagator_l2(feats_l1, matrix_flows, [feats_])
        feats_l3 = self.propagator_l3(feats_l2, matrix_flows, [feats_, feats_l1])
        feats_l4 = self.propagator_l4(feats_l3, matrix_flows, [feats_, feats_l1, feats_l2])

        return self.upsample(torch.cat([feats_, feats_l1, feats_l2, feats_l3, feats_l4], dim=-3), lrs)


