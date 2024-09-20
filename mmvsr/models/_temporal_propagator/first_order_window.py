from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmvsr.models.archs import ResidualBlockNoBN
from mmvsr.models.utils import flow_warp, make_layer
from mmvsr.registry import MODELS

C_DIM = -3

@MODELS.register_module()
class FirstOrderWindowPropagator(BaseModule):
    
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
            aligner_def = FirstOrderAligner
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

            if i > 0:
                n1_flow = flows[:, self.feat_indices[i - 1], ...]
                n1_feat = curr_feats[:, self.feat_indices[i - 1], ...] 
                n1_cond = self.warper(n1_feat, n1_flow.permute(0, 2, 3, 1))

            n1c_cond = torch.cat([n1_cond, curr_feat], dim=1)
            align_feat = self.aligner(n1_feat, n1c_cond, [n1_flow])
            cat_feat = torch.cat([curr_feat, align_feat, *[it[:, self.feat_indices[i], ...] for it in prev_feats]], dim=C_DIM)
            prop_feat = self.fextor(cat_feat)

            out_feats.append(prop_feat.clone())

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)

class FirstOrderAligner(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, unalign_feat, rough_aligned_feat, n1_flow):
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

from mmengine.model.weight_init import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

class FirstOrderDeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 3, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 18 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self): # Init constant offset
        
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, unalign_feat, rough_aligned_feat, n1_flow): # Forward function
        
        rough_aligned_feat = torch.cat([rough_aligned_feat, n1_flow], dim=1)
        out = self.conv_offset(rough_aligned_feat)
        offset, mask = torch.chunk(out, 2, dim=1)

        offset = offset + n1_flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(unalign_feat, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding, self.dilation, 
                                       self.groups, self.deform_groups)
