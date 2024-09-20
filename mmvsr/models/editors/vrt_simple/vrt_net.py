# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmvsr.registry import MODELS
# from mmvsr.models.basicsr_archs.optical_flow_archs import SpyNetMultiScale
# from mmvsr.models.basicsr_archs.upsample import Conv3dPixelShuffle as Upsample
from mmvsr.models.utils import flow_warp

from einops import rearrange
from einops.layers.torch import Rearrange

from .unet_vrt_arch import UnetBlock

from .tmsa_modules import RTMSA

@MODELS.register_module()
class VRTNet(BaseModule):

    def __init__(self,
                 upscale=4,
                 in_channels=3,
                 out_channels=3,
                 window_size=[4, 8, 8],
                 unet_block_depths = 4, 
                 unet_block_channels = 32,
                 unet_n_heads=4,  
                 lnet_n_blocks=4,
                 lnet_block_depths = 4, 
                 lnet_block_channels = 32,
                 lnet_n_heads = 4,  
                 lnet_indep_layers=[2, 3],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 n_frames=2,
                 deformable_groups=8,
                 spynet_path=None
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale
        self.n_frames = n_frames

        # conv_first
        self.conv_first = nn.Conv3d(in_channels * (1 + 2 * 4), 
                                    unet_block_channels, 
                                    kernel_size=(1, 3, 3), 
                                    padding=(0, 1, 1))

        self.spynet = SpyNetMultiScale(pretrained=spynet_path, 
                                     return_levels=[2, 3, 4, 5])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7*unet_block_depths+lnet_n_blocks*lnet_block_depths)]  # stochastic depth decay rule
        
        self.unet = nn.ModuleList([
            UnetBlock(
                in_channels=unet_block_channels,
                out_channels=unet_block_channels,
                n_mma_blocks = unet_block_depths // 4 * 3,
                n_msa_blocks = unet_block_depths // 4 * 1,
                # Confgures for TMSAG
                n_heads=unet_n_heads, window_size=window_size, 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop_path=dpr[i * unet_block_depths : (i + 1) * unet_block_depths],
                norm_layer=norm_layer,
                # Confgures for PW
                deformable_groups=deformable_groups,  
                kernel_size=3, padding=1, 
                max_residue_magnitude=10/scale,
                n_frames=n_frames,
                reshape=reshape,
            ) for i, (reshape, scale) in enumerate(zip(['none', 'down', 'down', 'down', 'up', 'up', 'up'], 
                                                       [1, 2, 4, 8, 4, 2, 1]))
        ])

        self.ffn = nn.Sequential(
                Rearrange('n c d h w ->  n d h w c'),
                nn.LayerNorm(unet_block_channels),
                nn.Linear(unet_block_channels, lnet_block_channels),
                Rearrange('n d h w c -> n c d h w')
        )

        self.lnet = nn.ModuleList([
            RTMSA(n_channels=lnet_block_channels,
                n_blocks=lnet_block_depths,
                n_heads=lnet_n_heads,
                window_size=[1, window_size[1], window_size[2]] \
                    if i in lnet_indep_layers else window_size, # T=1 or T=all
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_path=dpr[7 * unet_block_depths + i * lnet_block_depths : 
                              7 * unet_block_depths + (i + 1) * lnet_block_depths],
                norm_layer=norm_layer,
                )
            for i in range(lnet_n_blocks)
        ])

        self.norm = norm_layer(lnet_block_channels)
        self.shape_back = nn.Linear(lnet_block_channels, unet_block_channels)

        # for video sr
        r_channels = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(unet_block_channels, r_channels, 
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True)
        
        )
        self.upsample = Upsample(upscale, r_channels)

        self.conv_last = nn.Conv3d(r_channels, out_channels, 
                                   kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        # x: (N, D, C, H, W)

        b, t, c, h, w = x.shape

        x_lq = x.clone()

        # calculate flows
        flows_backward, flows_forward = self.compute_flow(x)

        # Initial Image Space Alignment
        x_backward, x_forward = self.apply_flow(x, flows_backward[0], flows_forward[0])
        
        x = torch.cat([x, x_backward, x_forward], 2)

        # video sr
        x = self.conv_first(x.transpose(1, 2))

        u = self.apply_unet(x, flows_backward, flows_forward)
        u = self.ffn(u)
        u = self.apply_lnet(u)

        u = rearrange(u, 'n c d h w -> n d h w c')
        u = self.norm(u)
        u = rearrange(u, 'n d h w c -> n c d h w')

        x = x + self.shape_back(u.transpose(1, 4)).transpose(1, 4)
        
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)

        B, T, C, H, W = x.shape

        return x + torch.nn.functional.interpolate(x_lq.view(-1, c, h, w), 
                                                   size=(H, W), mode='bilinear', 
                                                   align_corners=False).view(B, T, C, H, W)


    def apply_unet(self, x, flows_backward, flows_forward):
        '''Main u-net network for feature extraction.'''

        x1 = self.unet[0](x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.unet[1](x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.unet[2](x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.unet[3](x3, flows_backward[3::4], flows_forward[3::4])
        x = self.unet[4](x4, flows_backward[2::4], flows_forward[2::4])
        x = self.unet[5](x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.unet[6](x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1

        return x

    def apply_lnet(self, x):
        '''Subsequent l-net network for feature refine.'''

        for layer in self.lnet:
            x = layer(x)
        return x
    
    def compute_flow(self, x):
        b, n, c, h, w = x.size()
        lrs_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # Calculate backward flow
        flows_backward = self.spynet(lrs_1, lrs_2)
        flows_backward = [flow.view(b, n - 1, 2, 
                                    h // (2 ** i), 
                                    w // (2 ** i)) 
                        for flow, i in zip(flows_backward, range(4))]

        # Calculate forward flow
        flows_forward = self.spynet(lrs_2, lrs_1)
        flows_forward = [flow.view(b, n - 1, 2, 
                                h // (2 ** i), 
                                w // (2 ** i)) 
                        for flow, i in zip(flows_forward, range(4))]

        return flows_backward, flows_forward
    
    def apply_flow(self, x, flows_backward, flows_forward):
        n = x.size(1)

        # Backward warping
        x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            # frame i+1 aligned towards i
            x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 
                                        'nearest4'))  

        # Forward warping
        x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            # frame i-1 aligned towards i
            x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 
                                    'nearest4'))  

        return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]
