# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init

from mmvsr.models.archs import PixelShufflePack
from mmvsr.models.utils import flow_warp
from mmvsr.registry import MODELS
from ..basicvsr.basicvsr_net import ResidualBlocksWithInputConv, SPyNet

@MODELS.register_module()
class BaselineVSRPlusPlusNet(BaseModule):
    """BaselineVSRPlusPlus network structure.

    Support either x4 upsampling or same size output.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 num_prop_layers=4):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input

        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        self.num_prop_layers = num_prop_layers

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.tagg = nn.ModuleDict()

        for layer_id in range(self.num_prop_layers):
            module_name = str(layer_id)
            self.deform_align[module_name] = FirstOrderDeformableAlignment(
                mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            
            self.tagg[module_name] = nn.Conv2d(
                3 * mid_channels, mid_channels, kernel_size=1)

            ### NOTE: The backbone below will have input size
            # layer 0: 2*64 for prop_feat, curr_feat
            # layer 1: 1*64 + 2*64 for prop_feat, curr_feat + a frame
            # layer 2: 2*64 + 2*64 for prop_feat, curr_feat + a frame + another frame
            
            # self.backbone[module] = ResidualBlocksWithInputConv(
            #     (2 + i) * mid_channels, mid_channels, num_blocks)

            self.backbone[module_name] = ResidualBlocksWithInputConv(
                (layer_id + 1) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        # self.reconstruction = ResidualBlocksWithInputConv(
        #     5 * mid_channels, mid_channels, 5)
        
        self.reconstruction = ResidualBlocksWithInputConv(
            (num_prop_layers + 1) * mid_channels, mid_channels, 5)

        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
#-------# compute spatial features
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
        #     'The height and width of low-res inputs must be at least 64, '
        #     f'but got {h} and {w}.')
        
#-------# compute optical flow using the low-res inputs
        flows_forwards, flows_backwards = self.compute_flow(lqs_downsample)

#-------# feature propagation, 
        for layer_id in range(self.num_prop_layers):
            module_name = str(layer_id)
            feats = self.propagate(feats, flows_forwards, flows_backwards, layer_id)

        return self.upsample(lqs, feats)

    def propagate(self, feats, forward_flows, backward_flows, layer_id):

        n, t, _, h, w = forward_flows.size()

        module_name = str(layer_id)
        pre_module_name = str(layer_id - 1) if layer_id > 0 else 'spatial'

        for idx in range(t + 1):
            feat_raw_f = [
                safe_get_feat(feats[pre_module_name], idx - 2),  # Second-to-last frame
                safe_get_feat(feats[pre_module_name], idx - 1),  # Previous frame
                safe_get_feat(feats[pre_module_name], idx)       # Current frame
            ]
            
            flows_f = [
                safe_get_flow(forward_flows, idx - 2),  # Flow from second-to-last to last frame
                safe_get_flow(forward_flows, idx - 1)   # Flow from last to current frame
            ]
            cond_f_n = [None, None, feat_raw_f[-1]]

            for j in range(1, -1, -1): # index = [1, 0]
                if flows_f[j] is None: 
                    # All other flows are None, return previous warp align
                    for jj in range(j, -1, -1):
                        cond_f_n[jj] = cond_f_n[j + 1]
                    break
                if j + 1 < len(flows_f):
                    flows_f[j] = flows_f[j + 1] + flow_warp(flows_f[j], flows_f[j + 1].permute(0, 2, 3, 1))

                cond_f_n[j] = flow_warp(feat_raw_f[j], flows_f[j].permute(0, 2, 3, 1))

            cond_f = torch.cat([it for it in cond_f_n], dim=1)
            feat_raw_f = torch.cat(feat_raw_f[:-1], dim=1)
            feat_align_f = self.deform_align[module_name](feat_raw_f, cond_f, flows_f)

            feat_raw_b = [
                safe_get_feat(feats[pre_module_name], idx + 2),  # Second-to-last frame
                safe_get_feat(feats[pre_module_name], idx + 1),  # Previous frame
                safe_get_feat(feats[pre_module_name], idx)       # Current frame
            ]
            
            flows_b = [
                safe_get_flow(backward_flows, idx + 2),  # Flow from second-to-last to last frame
                safe_get_flow(backward_flows, idx + 1)   # Flow from last to current frame
            ]
            cond_b_n = [None, None, feat_raw_b[-1]]

            for j in range(1, -1, -1): # index = [1, 0]
                if flows_b[j] is None: 
                    # All other flows are None, return previous warp align
                    for jj in range(j, -1, -1):
                        cond_b_n[jj] = cond_b_n[j + 1]
                    break
                if j + 1 < len(flows_b):
                    flows_b[j] = flows_b[j + 1] + flow_warp(flows_b[j], flows_b[j + 1].permute(0, 2, 3, 1))

                cond_b_n[j] = flow_warp(feat_raw_b[j], flows_b[j].permute(0, 2, 3, 1))

            cond_b = torch.cat([it for it in cond_b_n], dim=1)
            feat_raw_b = torch.cat(feat_raw_b[:-1], dim=1)
            feat_align_b = self.deform_align[module_name](feat_raw_b, cond_b, flows_b)


            feat_tagg = [feat_align_f, feat_align_b, feats[pre_module_name][idx]]

            feat_dense = [feat_tagg] + [feats[k][idx]
                                        for k in feats if k not in ['spatial', module_name]]

            feat_dense = torch.cat(feat_dense, dim=1)

            # Run through a backbone network to update popagated features ([B*2 or B*3, C, H, W])
            feat_prop = self.backbone[module_name](feat_dense)

            if module_name not in feats: feats[module_name] = []
            feats[module_name].append(feat_prop)

        return feats
    
    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            outputs.append(hr)

        return torch.stack(outputs, dim=1)


    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

class FirstOrderDeformableAlignment(ModulatedDeformConv2d):
    """First-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue. Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(FirstOrderDeformableAlignment, self).__init__(*args, **kwargs)

        # Change the channel size in the offset layer to accommodate only one flow input
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Initialize constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        """Forward function."""
        # 'extra_feat' is to derive input-aware offset, 
        # so only using 'x' or another frame like x[t+1] 
        # will not work!
        extra_feat = torch.cat([extra_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        offset, mask = torch.split(out, [out.shape[1]//3*2, out.shape[1]//3], dim=1)

        # Calculate offset based on the first order flow only
        offset = self.max_residue_magnitude * torch.tanh(offset)
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

def safe_get_feat(feats, idx):
    """ Returns feature with replicate padding. """
    max_index = len(feats) - 1
    if idx < 0:
        idx = 0
    elif idx > max_index:
        idx = max_index
    return feats[idx]

def safe_get_flow(flows, idx):
    """ Returns flow with None padding for out of bounds. """
    if 0 <= idx < len(flows):
        return flows[idx]
    else:
        return None  # No flow data available for this index

