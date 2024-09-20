from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.models.archs import ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class FirstOrderRecurrentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, n_frames=7,
                 fextor_def=None, aligner_def=None,
                 fextor_args=None,
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels

        self.is_reversed = is_reversed

        if fextor_def is None:
            fextor_def = ResidualBlocksWithInputConv
            fextor_args = dict(in_channels=mid_channels+3, out_channels=mid_channels, num_blocks=30)
        if aligner_def is None:
            aligner_def = Alignment

        # Function definitions or classes to create fextor and aligner
        self.fextor = fextor_def(**fextor_args)
        self.aligner = aligner_def()
        self.feat_indices = list(range(-1, -n_frames - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(n_frames))

    def forward(self, curr_feats, flows, prev_feats=[]):

        n, t, c, h, w = curr_feats.size()

        outputs = list()
        feat_prop = curr_feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            curr_feat = curr_feats[:, self.feat_indices[i], :, :, :]
            if i > 0:
                flow = flows[:, self.feat_indices[i - 1], :, :, :]
                feat_prop = self.aligner(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([curr_feat, feat_prop, *[it[:, self.feat_indices[i], :, :, :] 
                                                                for it in prev_feats]], dim=C_DIM)
            
            feat_prop = self.fextor(feat_prop)

            outputs.append(feat_prop.clone())

        if self.is_reversed:
            outputs = outputs[::-1]

        return outputs

class Alignment(BaseModule):
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
