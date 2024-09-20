from logging import WARNING

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmvsr.registry import MODELS
from mmvsr.models._aligner.warper import Warper


C_DIM = -3

@MODELS.register_module()
class SecondOrderUnidirectionalRecurrentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, n_frames=7,
                 warper_def=None, warper_args=None,
                 aligner_def=None, aligner_args=None,
                 fextor_def=None, fextor_args=None,
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels
        self.is_reversed = is_reversed

        self.fextor_def = fextor_def
        if fextor_def:
            self.fextor = fextor_def(**fextor_args)

        self.warper_def = warper_def
        if warper_def:  # This enables 1) no pre warp, 2) patch/block warp
            self.warper = self.warper_def(**warper_args)
        else:
            self.warper = Warper()

        self.aligner_def = aligner_def
        if aligner_def:
            self.aligner = aligner_def(**aligner_args)

        self.easy_indices = list(range(-1, -n_frames - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(n_frames))

    def forward(self, now_feats, flows, prev_feats=[]):

        n, t, c, h, w = now_feats.size()

        out_feats = list()
        prop_feat = now_feats.new_zeros(n, self.mid_channels, h, w)
        align_feat = now_feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            
            now_feat = now_feats[:, self.easy_indices[i], ...]
            od1_cond = torch.zeros_like(now_feat)
            od2_cond = torch.zeros_like(now_feat)
            od1_flow = now_feats.new_zeros(n, 2, h, w)
            od2_flow = now_feats.new_zeros(n, 2, h, w)
            # there is no ``od1_feat`` which is de facto ``prop_feat``
            od2_feat = torch.zeros_like(prop_feat)

            if i > 0:
                od1_flow = flows[:, self.easy_indices[i - 1], ...]
                od1_cond = self.warper(prop_feat, od1_flow.permute(0, 2, 3, 1))
                
                if i > 1:
                    od2_flow = flows[:, self.easy_indices[i - 2], :, :, :]
                    # Compute second-order optical flow using first-order flow.
                    od2_flow = od1_flow + self.warper(od2_flow, od1_flow.permute(0, 2, 3, 1))
                    od2_feat = out_feats[-2] # [-2]: last-last out feat; [-1]: last out feat ``prop_feat``
                    od2_cond = self.warper(od2_feat, od2_flow.permute(0, 2, 3, 1))

            # Concatenate conditions for deformable convolution.
            cond = torch.cat([od1_cond, od2_cond, now_feat], dim=1)
            # Concatenate features for deformable convolution.
            feat = torch.cat([prop_feat, od2_feat], dim=1)
            # Use deformable convolution to refine the offset (coarse='od1_flow','od2_flow'),
            # then apply it to align 'prop_feat'
            # Refined Alignment:  ``fwd_feat`` and ``bwd_feat`` are refined aligned features
            if self.aligner_def:
                align_feat = self.aligner(feat, cond, [od1_flow, od2_flow])
            else:               # In the case there is no refined alignment
                align_feat = (od1_cond + od2_cond + now_feat) / 3

            # Feature Extraction: ``cat_feat`` contains refined aligned features and data from previous layers
            if self.fextor_def:
                cat_feat = torch.cat([now_feat, align_feat, 
                                    *[it[:, self.easy_indices[i], ...] for it in prev_feats]], dim=C_DIM)
                prop_feat = self.fextor(cat_feat)
            else:               # In the case there is no feature extraction, use aggregation instead
                prop_feat = (now_feat + align_feat) / 2

            out_feats.append(prop_feat)

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)
    
@MODELS.register_module()
class FirstOrderUnidirectionalRecurrentPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, n_frames=7,
                 warper_def=None, warper_args=None,
                 aligner_def=None, aligner_args=None,
                 fextor_def=None, fextor_args=None,
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels
        self.is_reversed = is_reversed

        self.fextor_def = fextor_def
        if fextor_def:
            self.fextor = fextor_def(**fextor_args)

        self.warper_def = warper_def
        if warper_def:  # This enables 1) no pre warp, 2) patch/block warp
            self.warper = self.warper_def(**warper_args)
        else:
            self.warper = Warper()

        self.aligner_def = aligner_def
        if aligner_def:
            self.aligner = aligner_def(**aligner_args)

        self.easy_indices = list(range(-1, -n_frames - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(n_frames))

    def forward(self, now_feats, flows, prev_feats=[]):

        flows = flows[0]

        n, t, c, h, w = now_feats.size()

        out_feats = list()
        prop_feat = now_feats.new_zeros(n, self.mid_channels, h, w)
        align_feat = now_feats.new_zeros(n, self.mid_channels, h, w)

        for i in range(0, t):
            
            now_feat = now_feats[:, self.easy_indices[i], ...]
            od1_cond = torch.zeros_like(now_feat)
            od1_flow = now_feats.new_zeros(n, 2, h, w)

            if i > 0:
                od1_flow = flows[:, self.easy_indices[i - 1], ...]
                od1_cond = self.warper(prop_feat, od1_flow.permute(0, 2, 3, 1))

            cond = torch.cat([od1_cond, now_feat], dim=1)

            if self.aligner_def:
                align_feat = self.aligner(prop_feat, cond, [od1_flow])
            else:               # In the case there is no refined alignment
                align_feat = (od1_cond + now_feat) / 2

            # Feature Extraction: ``cat_feat`` contains refined aligned features and data from previous layers
            if self.fextor_def:
                cat_feat = torch.cat([now_feat, align_feat, 
                                    *[it[:, self.easy_indices[i], ...] for it in prev_feats]], dim=C_DIM)
                prop_feat = self.fextor(cat_feat)
            else:               # In the case there is no feature extraction, use aggregation instead
                prop_feat = (now_feat + align_feat) / 2

            out_feats.append(prop_feat)

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)