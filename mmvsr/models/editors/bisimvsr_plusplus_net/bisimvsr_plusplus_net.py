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
class BisimVSRPlusPlusNet(BaseModule):

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None):

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

        # propagation branches
        self.deform_align_f, self.deform_align_b = nn.ModuleDict(), nn.ModuleDict()
        self.backbone_f, self.backbone_b = nn.ModuleDict(), nn.ModuleDict()

        # bi-direction aggregation module. 
        self.bagg = nn.ModuleDict()

        modules = ['0', '1']
        for i, module in enumerate(modules):
            self.deform_align_f[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3,
                padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.deform_align_b[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3,
                padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone_f[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)
            self.backbone_b[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)
            self.bagg[module] = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1)
        
        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            (len(modules) + 1) * mid_channels, mid_channels, 5)
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

        # both flows_forward and flows_backward are of index 0, 1, 2, ... t-1
        return flows_forward, flows_backward

    def propagate(self, feats, forward_flows, backward_flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            forward_flows (tensor): Optical forward_flows with shape (n, t - 1, 2, h, w).
            backward_flows (tensor): Optical backward_flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = forward_flows.size()

        # Initialize indices for frames and forward_flows
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))

        # Initialize a tensor to store propagated features. The size is determined by
        # the mid-channel dimension of the network architecture.

        prev_feats = [feats[k] for k in feats if k not in ['spatial', module_name]]
        
        feat_new = []
#-------# Forward iterate (Recurrent!!!) through each frame index to propagate features.
        feat_prop_f = forward_flows.new_zeros(n, self.mid_channels, h, w)
        feats_f = []
        for i, idx in enumerate(frame_idx):
            # Retrieve current frame features.
            feat_current = feats['spatial'][idx]
            # If not the first frame, calculate deformable alignments.
                # second-order deformable alignment
            if i > 0:
                flow_n1 = forward_flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop_f, flow_n1.permute(0, 2, 3, 1))

                feat_n2, flow_n2, cond_n2 = torch.zeros_like(feat_prop_f), torch.zeros_like(flow_n1), torch.zeros_like(cond_n1)

                if i > 1:  # Compute second-order features if beyond the second frame.
                    feat_n2 = feats_f[-2]
                    flow_n2 = forward_flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop_f = torch.cat([feat_prop_f, feat_n2], dim=1)
                feat_prop_f = self.deform_align_f[module_name](feat_prop_f, cond, flow_n1, flow_n2)

            feat_f = [feat_current] + [it[idx] for it in prev_feats] + [feat_prop_f]
            feat_f = torch.cat(feat_f, dim=1)
            feat_prop_f = feat_prop_f + self.backbone_f[module_name](feat_f)
            feats_f.append(feat_prop_f)

#-------# Backward iterate (Recurrent!!!) through each frame index to propagate features.
        feat_prop_b = forward_flows.new_zeros(n, self.mid_channels, h, w)
        feats_b = []
        for i in reversed(frame_idx):
            feat_current = feats['spatial'][i]
            if i < t:
                flow_n1 = backward_flows[:, flow_idx[i + 1], :, :, :]
                cond_n1 = flow_warp(feat_prop_b, flow_n1.permute(0, 2, 3, 1))

                feat_n2, flow_n2, cond_n2 = torch.zeros_like(feat_prop_b), torch.zeros_like(flow_n1), torch.zeros_like(cond_n1)

                if i < t - 1:
                    feat_n2 = feats_f[i + 2]
                    flow_n2 = backward_flows[:, flow_idx[i + 2], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop_b = torch.cat([feat_prop_b, feat_n2], dim=1)
                feat_prop_b = self.deform_align_b[module_name](feat_prop_b, cond, flow_n1, flow_n2)

            feat_b = [feat_current] + [it[i] for it in prev_feats] + [feat_prop_b]
            feat_b = torch.cat(feat_b, dim=1)
            feat_prop_b = feat_prop_b + self.backbone_b[module_name](feat_b)
            feats_b.append(feat_prop_b)

#-------# Aggregatopm of Backward and Forward Aligned Features
        for i in range(t + 1):
            feat_new.append(self.bagg[module_name](torch.cat([feats_f[i], feats_b[i]], dim=1)))

        feats[module_name] = feat_new
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

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

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

        feats_ = self.feat_extract(lqs.view(-1, c, h, w))
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propagation
        for iter_ in [0, 1]:
            module_name = str(iter_)

            feats = self.propagate(feats, flows_forward, flows_backward, module_name)

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

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
            residue (Eq. 6 in paper). Default: 10.
    """

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

    def forward(self, x, extra_feat, flow_1, flow_2):
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
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)
