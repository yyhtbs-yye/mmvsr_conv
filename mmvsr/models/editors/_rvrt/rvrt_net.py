import torch
import torch.nn as nn
import torch.nn.functional as F


from mmengine.model import BaseModule
from mmvsr.registry import MODELS

from einops.layers.torch import Rearrange

from .rvrt_modules import *
from .rvrt_utils import *
from .swinir_modules import *
from .gda_modules import *

@MODELS.register_module()
class RVRTNet(nn.Module):
    """ Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT).
            A PyTorch impl of : `Recurrent Video Restoration Transformer with Guided Deformable Attention`  -
              https://arxiv.org/pdf/22054.00000

        Args:
            upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
            clip_size (int): Size of clip in recurrent restoration transformer.
            img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
            window_size (int | tuple(int)): Window size. Default: (2,8,8).
            num_blocks (list[int]): Number of RSTB blocks in each stage.
            depths (list[int]): Depths of each RSTB.
            embed_dims (list[int]): Number of linear projection output channels.
            num_heads (list[int]): Number of attention head of each stage.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
            inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
            spynet_path (str): Pretrained SpyNet model path.
            deformable_groups (int): Number of deformable groups in deformable attention. Default: 12.
            attention_heads (int): Number of attention heads in deformable attention. Default: 12.
            attention_window (list[int]): Attention window size in aeformable attention. Default: [3, 3].
            nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
            use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
            use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
            no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
            no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
            cpu_cache_length: (int): Maximum video length without cpu caching. Default: 100.
        """

    def __init__(self,
                 upscale=4,
                 clip_size=2,
                 img_size=[2, 64, 64],
                 window_size=[2, 8, 8],
                 num_blocks=[1, 2, 1],
                 depths=[2, 2, 2],
                 embed_dims=[144, 144, 144],
                 num_heads=[6, 6, 6],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 inputconv_groups=[1, 1, 1, 1, 1, 1],
                 spynet_path=None,
                 max_residue_magnitude=10,
                 deformable_groups=12,
                 attention_heads=12,
                 attention_window=[3, 3],
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 cpu_cache_length=100
                 ):

        super().__init__()
        self.upscale = upscale
        self.clip_size = clip_size
        self.nonblind_denoising = nonblind_denoising
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in range(100)]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in range(100)]
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # shallow feature extraction
        if self.upscale == 4:
            # video sr
            self.feat_extract = RSTBWithInputConv(in_channels=3,
                                                  kernel_size=(1, 3, 3),
                                                  groups=inputconv_groups[0],
                                                  num_blocks=num_blocks[0],
                                                  dim=embed_dims[0],
                                                  input_resolution=[1, img_size[1], img_size[2]],
                                                  depth=depths[0],
                                                  num_heads=num_heads[0],
                                                  window_size=[1, window_size[1], window_size[2]],
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  norm_layer=norm_layer,
                                                  use_checkpoint_attn=[False],
                                                  use_checkpoint_ffn=[False]
                                                  )
        else:
            # video deblurring/denoising
            self.feat_extract = nn.Sequential(Rearrange('n d c h w -> n c d h w'),
                                              nn.Conv3d(4 if self.nonblind_denoising else 3, embed_dims[0], (1, 3, 3),
                                                        (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              nn.Conv3d(embed_dims[0], embed_dims[0], (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              Rearrange('n c d h w -> n d c h w'),
                                              RSTBWithInputConv(
                                                                in_channels=embed_dims[0],
                                                                kernel_size=(1, 3, 3),
                                                                groups=inputconv_groups[0],
                                                                num_blocks=num_blocks[0],
                                                                dim=embed_dims[0],
                                                                input_resolution=[1, img_size[1], img_size[2]],
                                                                depth=depths[0],
                                                                num_heads=num_heads[0],
                                                                window_size=[1, window_size[1], window_size[2]],
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                norm_layer=norm_layer,
                                                                use_checkpoint_attn=[False],
                                                                use_checkpoint_ffn=[False]
                                                               )
                                              )

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        # recurrent feature refinement
        self.backbone = nn.ModuleDict()
        self.deform_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            # deformable attention
            self.deform_align[module] = GuidedDeformAttnPack(embed_dims[1],
                                                             embed_dims[1],
                                                             attention_window=attention_window,
                                                             attention_heads=attention_heads,
                                                             deformable_groups=deformable_groups,
                                                             clip_size=clip_size,
                                                             max_residue_magnitude=max_residue_magnitude)

            # feature propagation
            self.backbone[module] = RSTBWithInputConv(
                                                     in_channels=(2 + i) * embed_dims[0],
                                                     kernel_size=(1, 3, 3),
                                                     groups=inputconv_groups[i + 1],
                                                     num_blocks=num_blocks[1],
                                                     dim=embed_dims[1],
                                                     input_resolution=img_size,
                                                     depth=depths[1],
                                                     num_heads=num_heads[1],
                                                     window_size=window_size,
                                                     mlp_ratio=mlp_ratio,
                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                     norm_layer=norm_layer,
                                                     use_checkpoint_attn=[use_checkpoint_attns[i]],
                                                     use_checkpoint_ffn=[use_checkpoint_ffns[i]]
                                                     )

        # reconstruction
        self.reconstruction = RSTBWithInputConv(
                                               in_channels=5 * embed_dims[0],
                                               kernel_size=(1, 3, 3),
                                               groups=inputconv_groups[5],
                                               num_blocks=num_blocks[2],

                                               dim=embed_dims[2],
                                               input_resolution=[1, img_size[1], img_size[2]],
                                               depth=depths[2],
                                               num_heads=num_heads[2],
                                               window_size=[1, window_size[1], window_size[2]],
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               norm_layer=norm_layer,
                                               use_checkpoint_attn=[False],
                                               use_checkpoint_ffn=[False]
                                               )
        self.conv_before_upsampler = nn.Sequential(
                                                  nn.Conv3d(embed_dims[-1], 64, kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                                  )
        self.upsampler = Upsample(4, 64)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

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

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def propagate(self, feats, flows, module_name, updated_flows=None):
        """Propagate the latent clip features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, clip_size, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            updated_flows dict(list[tensor]): Each component is a list of updated
                optical flows with shape (n, clip_size, 2, h, w).

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        # Determine dimensions from the input flow tensor.
        n, t, _, h, w = flows.size()

        # Decide the order of processing clips and flows based on the module name indicating forward or backward propagation.
        if 'backward' in module_name:
            flow_idx = range(0, t + 1)[::-1]
            clip_idx = range(0, (t + 1) // self.clip_size)[::-1]
        else:
            flow_idx = range(-1, t)
            clip_idx = range(0, (t + 1) // self.clip_size)

        # Initialize or add to a list for updated flows for '_1' modules.
        if '_1' in module_name:
            updated_flows[f'{module_name}_n1'] = []
            updated_flows[f'{module_name}_n2'] = []

        # Initialize a tensor to hold propagated features.
        feat_prop = torch.zeros_like(feats['shallow'][0])
        if self.cpu_cache:
            feat_prop = feat_prop.cuda()

        # Processing for each clip based on the indices determined.
        last_key = list(feats)[-2] # Take the second last key for reference feature extraction.
        for i in range(0, len(clip_idx)):
            idx_c = clip_idx[i]             # 'idx_c' is the current clip index.
            if i > 0:
                # Handling the first type of propagation involving computation of new flow fields from existing ones.
                if '_1' in module_name:
                    # Calculate intermediary flows by warping and summing existing flows.
                    flow_n01 = flows[:, flow_idx[self.clip_size * i - 1], :, :, :]
                    flow_n12 = flows[:, flow_idx[self.clip_size * i], :, :, :]
                    flow_n23 = flows[:, flow_idx[self.clip_size * i + 1], :, :, :]
                    flow_n02 = flow_n12 + flow_warp(flow_n01, flow_n12.permute(0, 2, 3, 1))
                    flow_n13 = flow_n23 + flow_warp(flow_n12, flow_n23.permute(0, 2, 3, 1))
                    flow_n03 = flow_n23 + flow_warp(flow_n02, flow_n23.permute(0, 2, 3, 1))
                    flow_n1 = torch.stack([flow_n02, flow_n13], 1)
                    flow_n2 = torch.stack([flow_n12, flow_n03], 1)
                else:
                    # For '_2' modules, reuse updated flows from corresponding '_1' modules.
                    module_name_old = module_name.replace('_2', '_1')
                    flow_n1 = updated_flows[f'{module_name_old}_n1'][i - 1]
                    flow_n2 = updated_flows[f'{module_name_old}_n2'][i - 1]

                # Prepare feature maps for propagation
                if 'backward' in module_name:
                    feat_q = feats[last_key][idx_c].flip(1)
                    feat_k = feats[last_key][clip_idx[i - 1]].flip(1)
                else:
                    feat_q = feats[last_key][idx_c]
                    feat_k = feats[last_key][clip_idx[i - 1]]

                # Warp the propagated features based on the computed optical flows
                feat_prop_warped1 = flow_warp(feat_prop.flatten(0, 1),
                                           flow_n1.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
                feat_prop_warped2 = flow_warp(feat_prop.flip(1).flatten(0, 1),
                                           flow_n2.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

                # Align the features and update the flows
                if '_1' in module_name:
                    feat_prop, flow_n1, flow_n2 = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                                                 [feat_prop_warped1, feat_prop_warped2],
                                                                                 [flow_n1, flow_n2],
                                                                                 True)
                    updated_flows[f'{module_name}_n1'].append(flow_n1)
                    updated_flows[f'{module_name}_n2'].append(flow_n2)
                else:
                    feat_prop = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                               [feat_prop_warped1, feat_prop_warped2],
                                                               [flow_n1, flow_n2],
                                                               False)
            # Gather the features for the current clip index
            if 'backward' in module_name:
                feat = [feats[k][idx_c].flip(1) for k in feats if k not in [module_name]] + [feat_prop]
            else:
                feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]

            # Update the propagated features with the backbone network
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=2))
            feats[module_name].append(feat_prop)

        # Reverse the order of features for backward propagation
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            feats[module_name] = [f.flip(1) for f in feats[module_name]]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        feats['shallow'] = torch.cat(feats['shallow'], 1)
        feats['backward_1'] = torch.cat(feats['backward_1'], 1)
        feats['forward_1'] = torch.cat(feats['forward_1'], 1)
        feats['backward_2'] = torch.cat(feats['backward_2'], 1)
        feats['forward_2'] = torch.cat(feats['forward_2'], 1)

        if self.cpu_cache:
            outputs = []
            for i in range(0, feats['shallow'].shape[1]):
                hr = torch.cat([feats[k][:, i:i + 1, :, :, :] for k in feats], dim=2)
                hr = self.reconstruction(hr.cuda())
                hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
                hr += torch.nn.functional.interpolate(lqs[:, i:i + 1, :, :, :].cuda(), size=hr.shape[-3:],
                                                      mode='trilinear', align_corners=False)
                hr = hr.cpu()
                outputs.append(hr)
                torch.cuda.empty_cache()

            return torch.cat(outputs, dim=1)

        else:
            hr = torch.cat([feats[k] for k in feats], dim=2)
            hr = self.reconstruction(hr)
            hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
            hr += torch.nn.functional.interpolate(lqs, size=hr.shape[-3:], mode='trilinear', align_corners=False)

            return hr

    def forward(self, lqs):
        """Forward function for RVRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, _, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.upscale == 4:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(lqs[:, :, :3, :, :].view(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
                .view(n, t, 3, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        # shallow feature extractions
        feats = {}
        if self.cpu_cache:
            feats['shallow'] = []
            for i in range(0, t // self.clip_size):
                feat = self.feat_extract(lqs[:, i * self.clip_size:(i + 1) * self.clip_size, :, :, :]).cpu()
                feats['shallow'].append(feat)
            flows_forward, flows_backward = self.compute_flow(lqs_downsample)

            lqs = lqs.cpu()
            lqs_downsample = lqs_downsample.cpu()
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
            torch.cuda.empty_cache()
        else:
            feats['shallow'] = list(torch.chunk(self.feat_extract(lqs), t // self.clip_size, dim=1))
            flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # recurrent feature refinement
        updated_flows = {}
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(feats, flows, module_name, updated_flows)

        # reconstruction
        return self.upsample(lqs[:, :, :3, :, :], feats)