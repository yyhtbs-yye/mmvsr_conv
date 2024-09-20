import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmvsr.registry import MODELS
from mmvsr.models._aligner.warper import Warper

C_DIM = -3

@MODELS.register_module()
class FirstOrderBidirectionalSlidingWindowPropagator(BaseModule):

    def __init__(self, mid_channels=64, n_frames=7,
                 warper_def=None, warper_args=None,
                 aligner_def=None, aligner_args=None,
                 fextor_def=None, fextor_args=None):

        super().__init__()

        self.mid_channels = mid_channels

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
        
    def forward(self, in_feats, flows, dense_feats=None):

        fwd_flows, bwd_flows = flows[0], flows[1]

        if dense_feats is None: dense_feats = []

        n, t, c, h, w = in_feats.size()
        out_feats = []

        for i in range(t):
            now_feat = in_feats[:, i, ...]

            if i == 0:
                fwd_feat = now_feat
                fwd_flow = torch.zeros_like(fwd_flows[:, 0, ...])
                bwd_feat = in_feats[:, i+1, ...]
                bwd_flow = bwd_flows[:, i,   ...]
            elif i == t - 1:
                bwd_feat = now_feat
                bwd_flow = torch.zeros_like(bwd_flows[:, 0, ...])
                fwd_feat = in_feats[:, i-1, ...] 
                fwd_flow = fwd_flows[:, i-1, ...]
            elif i > 0 and i < t-1:
                fwd_feat = in_feats[:, i-1, ...] 
                bwd_feat = in_feats[:, i+1, ...]
                fwd_flow = fwd_flows[:, i-1, ...]
                bwd_flow = bwd_flows[:, i,   ...]
            
            # Coarse Alignment: ``fwd_cond``, and ``bwd_cond`` are coarsely aligned features
            fwd_cond = self.warper(fwd_feat, fwd_flow.permute(0, 2, 3, 1))
            bwd_cond = self.warper(bwd_feat, bwd_flow.permute(0, 2, 3, 1)) 

            feat = torch.cat([fwd_feat, bwd_feat], dim=C_DIM)
            cond = torch.cat([fwd_cond, bwd_cond], dim=C_DIM)

            # Refined Alignment:  ``fwd_feat`` and ``bwd_feat`` are refined aligned features
            if self.aligner_def:
                align_feat = self.aligner(feat, torch.cat([cond, now_feat], dim=C_DIM), [fwd_flow, bwd_flow])
            else:               # In the case there is no refined alignment
                align_feat = (fwd_cond + bwd_cond + now_feat) / 3

            # Feature Extraction: ``cat_feat`` contains refined aligned features and data from previous layers
            if self.fextor_def:
                cat_feat = torch.cat([now_feat, align_feat,
                                      *[it[:, i, ...] for it in dense_feats]], dim=C_DIM)
                prop_feat = self.fextor(cat_feat)
            else:               # In the case there is no feature extraction, use aggregation instead
                prop_feat = (now_feat + align_feat) / 2

            out_feats.append(prop_feat)

        return torch.stack(out_feats, dim=1)

@MODELS.register_module()
class FirstOrderUnidirectionalSlidingWindowPropagator(BaseModule):

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

    def forward(self, in_feats, flows, dense_feats=None):

        if dense_feats is None: dense_feats = []

        n, t, c, h, w = in_feats.size()
        out_feats = []

        for i in range(t):
            now_feat = in_feats[:, self.easy_indices[i], ...]

            if i == 0:
                fwd_feat = now_feat
                fwd_flow = torch.zeros_like(flows[:, 0, ...])
            else:
                fwd_feat = in_feats[:, self.easy_indices[i-1], ...] 
                fwd_flow = flows[:, self.easy_indices[i-1], ...]

            cond = self.warper(fwd_feat, fwd_flow.permute(0, 2, 3, 1))

            if self.aligner_def:
                align_feat = self.aligner(fwd_feat, torch.cat([cond, now_feat], dim=C_DIM), [fwd_flow])
            else:
                align_feat = (cond + now_feat) / 2

            if self.fextor_def:
                cat_feat = torch.cat([now_feat, align_feat,
                                      *[it[:, i, ...] for it in dense_feats]], dim=C_DIM)
                prop_feat = self.fextor(cat_feat)
            else: # Fallback, 1x1 conv2d to aggregate, no dense connections from previous layers
                prop_feat = (now_feat + align_feat) / 2

            out_feats.append(prop_feat)

        return torch.stack(out_feats, dim=1)

@MODELS.register_module()
class SecondOrderUnidirectionalSlidingWindowPropagator(BaseModule):
    
    def __init__(self, mid_channels=64, n_frames=7,
                 fextor_def=None, fextor_args=None,
                 aligner_def=None, aligner_args=None, 
                 is_reversed=False):

        super().__init__()

        self.mid_channels = mid_channels

        self.is_reversed = is_reversed

        self.aligner_def = aligner_def

        if aligner_def:
            self.aligner = aligner_def(**aligner_args)

        self.fextor_def = fextor_def
        if fextor_def:
            self.fextor = fextor_def(**fextor_args)

        self.warper = Warper()

        self.easy_indices = list(range(-1, -n_frames - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(n_frames))

    def fwd(self, in_feats, flows, dense_feats=[]):

        n, t, c, h, w = in_feats.size()

        out_feats = list()

        for i in range(0, t):
            now_feat = in_feats[:, self.easy_indices[i], ...]
            # ----------------------Initialization-----------------------
            n1_flow = torch.zeros_like(flows[:, 0, ...])
            n1_feat = torch.zeros_like(now_feat)
            n1_cond = torch.zeros_like(now_feat)
            # ----------------------Initialization-----------------------
            n2_flow = torch.zeros_like(n1_flow)
            n2_feat = torch.zeros_like(now_feat)
            n2_cond = torch.zeros_like(now_feat)

            if i > 0:
                n1_flow = flows[:, self.easy_indices[i - 1], ...]
                n1_feat = in_feats[:, self.easy_indices[i - 1], ...] 
                n1_cond = self.warper(n1_feat, n1_flow.permute(0, 2, 3, 1))

                if i > 1:
                    n2_flow = flows[:, self.easy_indices[i - 2], ...]
                    n2_flow = n1_flow + self.warper(n2_flow, n1_flow.permute(0, 2, 3, 1))
                    n2_feat = in_feats[:, self.easy_indices[i - 2], ...]
                    n2_cond = self.warper(n2_feat, n2_flow.permute(0, 2, 3, 1))

            conds = torch.cat([n1_cond, n2_cond, now_feat], dim=1)
            feats = torch.cat([n1_feat, n2_feat], dim=1)

            if self.aligner_def:
                align_feat = self.aligner(feats, conds, [n1_flow, n2_flow])
            else:
                align_feat = (n1_cond + n2_cond + now_feat) / 3

            cat_feat = torch.cat([now_feat, align_feat, *[it[:, self.easy_indices[i], ...] for it in dense_feats]], dim=C_DIM)

            if self.fextor_def:
                prop_feat = self.fextor(cat_feat)
            else:
                prop_feat = (now_feat + align_feat) / 2

            out_feats.append(prop_feat.clone())

        if self.is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)
    

@MODELS.register_module()
class AnyOrderBidirectionalFullyConnectedPropagator(BaseModule):

    def __init__(self, mid_channels=64, n_frames=7,
                 warper_def=None, warper_args=None,
                 aligner_def=None, aligner_args=None,
                 fextor_def=None, fextor_args=None,):

        super().__init__()

        self.mid_channels = mid_channels

        self.fextor_def = fextor_def
        if fextor_def:
            self.fextor = fextor_def(**fextor_args)

        self.warper_def = warper_def
        if warper_def:  
            self.warper = self.warper_def(**warper_args)
        else:
            self.warper = Warper()

        self.aligner_def = aligner_def
        if aligner_def:
            self.aligner = aligner_def(**aligner_args)

    def forward(self, in_feats, matrix_flows, dense_feats=None):

        if dense_feats is None: dense_feats = []

        n, t, c, h, w = in_feats.size()
        out_feats = []

        for i in range(t):

            now_feat = in_feats[:, i, ...]

            # Alignment using global flow matrix
            conds = []
            flows = []
            feats = []
            for j in range(t):
                if j == i:
                    continue  # Skip self-alignment
                
                feat = in_feats[:, j, ...]
                flow = matrix_flows[:, i, j, ...]  # Assume matrix_flows holds pairwise flows
                cond = self.warper(feat, flow.permute(0, 2, 3, 1))
                conds.append(cond)
                flows.append(flow)
                feats.append(feat)

            # Refine features using aligner if available
            if self.aligner_def:
                align_feat = self.aligner(torch.cat(feats, dim=C_DIM), 
                                          torch.cat([*conds, now_feat], dim=C_DIM), 
                                          flows)
            else:
                align_feat = torch.mean(torch.stack([*conds, now_feat], dim=0), dim=0)

            # Feature Extraction
            if self.fextor_def:
                cat_feat = torch.cat([now_feat, align_feat,
                                      *[it[:, i, ...] for it in dense_feats]], dim=C_DIM)
                prop_feat = self.fextor(cat_feat)
            else:
                prop_feat = (now_feat + align_feat) / 2

            out_feats.append(prop_feat)

        return torch.stack(out_feats, dim=1)