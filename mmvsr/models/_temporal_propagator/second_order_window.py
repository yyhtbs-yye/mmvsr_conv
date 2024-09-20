from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.models.archs import ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class SecondOrderWindowPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, n_frames=7,
                 fextor_def=None, fextor_args=None,
                 aligner_def=None, aligner_args=None, 
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels

        self.is_reversed = is_reversed

        if fextor_def is None:
            fextor_def = ResidualBlocksWithInputConv
            fextor_args = dict(in_channels=mid_channels+3, out_channels=mid_channels, num_blocks=30)
        if aligner_def is None:
            aligner_def = SecondOrderAligner
            aligner_args = dict()

        # Function definitions or classes to create fextor and warper
        self.fextor = fextor_def(**fextor_args)
        self.warper = Warper()
        self.aligner = aligner_def(**aligner_args)
        self.feat_indices = list(range(-1, -n_frames - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(n_frames))

    def forward(self, curr_feats, flows, prev_feats=[]):

        n, t, c, h, w = curr_feats.size()

        out_feats = list()
        # prop_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)
        # align_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            curr_feat = curr_feats[:, self.feat_indices[i], ...]
            # ----------------------Initialization-----------------------
            n1_flow = torch.zeros_like(flows[:, 0, ...])
            n1_feat = torch.zeros_like(curr_feat)
            n1_cond = torch.zeros_like(curr_feat)
            # ----------------------Initialization-----------------------
            n2_flow = torch.zeros_like(n1_flow)
            n2_feat = torch.zeros_like(curr_feat)
            n2_cond = torch.zeros_like(curr_feat)

            if i > 0:
                n1_flow = flows[:, self.feat_indices[i - 1], ...]
                n1_feat = curr_feats[:, self.feat_indices[i - 1], ...] 
                n1_cond = self.warper(n1_feat, n1_flow.permute(0, 2, 3, 1))

                if i > 1:
                    n2_flow = flows[:, self.feat_indices[i - 2], ...]
                    n2_flow = n1_flow + self.warper(n2_flow, n1_flow.permute(0, 2, 3, 1))
                    n2_feat = curr_feats[:, self.feat_indices[i - 2], ...]
                    n2_cond = self.warper(n2_feat, n2_flow.permute(0, 2, 3, 1))

            n12c_cond = torch.cat([n1_cond, n2_cond, curr_feat], dim=1)
            n12_feat = torch.cat([n1_feat, n2_feat], dim=1)
            align_feat = self.aligner(n12_feat, n12c_cond, [n1_flow, n2_flow])
            cat_feat = torch.cat([curr_feat, align_feat, *[it[:, self.feat_indices[i], ...] for it in prev_feats]], dim=C_DIM)
            prop_feat = self.fextor(cat_feat)

            out_feats.append(prop_feat.clone())

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)

class SecondOrderAligner(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, unalign_feat, rough_aligned_feat, n12_flow):
        return rough_aligned_feat


class Warper(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, feat, flow):
        return flow_warp(feat, flow)

class ResidualBlocksWithInputConv(BaseModule):

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        )

    def forward(self, feat):
        return self.main(feat)
