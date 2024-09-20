# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmvsr.registry import MODELS


from einops import rearrange
from einops.layers.torch import Rearrange

from .vrt_utils import flow_warp, get_flow_between_frames, get_aligned_image_2frames, get_aligned_feature_2frames

from .vrt_modules import SpyNet, Mlp_GEGLU, Upsample, DCNv2PackFlowGuided
from .vrt_tmsa import RTMSA, TMSAG

@MODELS.register_module()
class VRTNet(BaseModule):
    r""" VRT
        Video Restoration Transformer (VRT),
        A PyTorch impl of : `VRT: A Video Restoration Transformer` 
        https://arxiv.org/pdf/2201.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        out_chans (int): Number of output image channels. Default: 3.
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
        no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
    """

    def __init__(self,
                 upscale=4,
                 in_chans=3,
                 out_chans=3,
                 window_size=[6, 16, 16],
                 depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                 indep_reconsts=[11, 12],
                 embed_dims=[64, 64, 64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96],
                 num_heads=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 spynet_path=None,
                 pa_frames=2,
                 deformable_groups=16,
                 recal_all_flows=False,
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows

        # conv_first
        conv_first_in_chans = in_chans*(1+2*4)
        
        self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # main body
        self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        reshapes = ['none', 'down', 'down', 'down', 'up', 'up', 'up']
        
        scales = [1, 2, 4, 8, 4, 2, 1]

        # stage 1- 7
        for i in range(7):
            setattr(self, f'stage{i + 1}',
                    Stage(
                        in_dim=embed_dims[i - 1],
                        dim=embed_dims[i],
                        depth=depths[i],
                        num_heads=num_heads[i],
                        mul_attn_ratio=mul_attn_ratio,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                        norm_layer=norm_layer,
                        pa_frames=pa_frames,
                        deformable_groups=deformable_groups,
                        reshape=reshapes[i],
                        max_residue_magnitude=10 / scales[i],
                        )
                    )

        # stage 8
        self.stage8 = nn.ModuleList(
            [nn.Sequential(
                Rearrange('n c d h w ->  n d h w c'),
                nn.LayerNorm(embed_dims[6]),
                nn.Linear(embed_dims[6], embed_dims[7]),
                Rearrange('n d h w c -> n c d h w')
            )]
        )
        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(dim=embed_dims[i],
                      depth=depths[i],
                      num_heads=num_heads[i],
                      window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                      norm_layer=norm_layer,
                      )
            )

        self.norm = norm_layer(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])

        # for video sr
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(embed_dims[0], num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv3d(num_feat, out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def reflection_pad2d(self, x, pad=1):

        x = torch.cat([torch.flip(x[:, :, 1:pad+1, :], [2]), x, torch.flip(x[:, :, -pad-1:-1, :], [2])], 2)
        x = torch.cat([torch.flip(x[:, :, :, 1:pad+1], [3]), x, torch.flip(x[:, :, :, -pad-1:-1], [3])], 3)
        
        return x

    def forward(self, x):
        # x: (N, D, C, H, W)

        b, t, c, h, w = x.shape

        x_lq = x.clone()

        # calculate flows
        flows_backward, flows_forward = self.get_flows(x)

        # warp input
        x_backward, x_forward = get_aligned_image_2frames(x,  flows_backward[0], flows_forward[0])
        x = torch.cat([x, x_backward, x_forward], 2)

        # video sr
        x = self.conv_first(x.transpose(1, 2))
        x = x + self.conv_after_body(
            self.forward_features(x, flows_backward, flows_forward).transpose(1, 4)).transpose(1, 4)
        
        x = self.conv_last(self.upsample(self.conv_before_upsample(x))).transpose(1, 2)

        B, T, C, H, W = x.shape

        return x + torch.nn.functional.interpolate(x_lq.view(-1, c, h, w), size=(H, W), mode='bilinear', align_corners=False).view(B, T, C, H, W)


    def get_flows(self, x):
        ''' Get flows for 2 frames, 4 frames or 6 frames.'''

        if self.pa_frames == 2:
            flows_backward, flows_forward = get_flow_between_frames(x, self.spynet)
        
        # elif self.pa_frames == 4:
        #     flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
        #     flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
        #     flows_backward = flows_backward_2frames + flows_backward_4frames
        #     flows_forward = flows_forward_2frames + flows_forward_4frames
    
        # elif self.pa_frames == 6:
        #     flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
        #     flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(flows_forward_2frames, flows_backward_2frames)
        #     flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames)
        #     flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
        #     flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames

        return flows_backward, flows_forward

    # def get_flow_2frames(self, x):
    #     '''Get flow between frames t and t+1 from x.'''

    #     b, n, c, h, w = x.size()
    #     x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
    #     x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

    #     # backward
    #     flows_backward = self.spynet(x_1, x_2)
    #     flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
    #                       zip(flows_backward, range(4))]

    #     # forward
    #     flows_forward = self.spynet(x_2, x_1)
    #     flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
    #                      zip(flows_forward, range(4))]

    #     return flows_backward, flows_forward

    # def get_flow_4frames(self, flows_forward, flows_backward):
    #     '''Get flow between t and t+2 from (t,t+1) and (t+1,t+2).'''

    #     # backward
    #     d = flows_forward[0].shape[1]
    #     flows_backward2 = []
    #     for flows in flows_backward:
    #         flow_list = []
    #         for i in range(d - 1, 0, -1):
    #             flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
    #             flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
    #             flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+2 to i
    #         flows_backward2.append(torch.stack(flow_list, 1))

    #     # forward
    #     flows_forward2 = []
    #     for flows in flows_forward:
    #         flow_list = []
    #         for i in range(1, d):
    #             flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
    #             flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
    #             flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-2 to i
    #         flows_forward2.append(torch.stack(flow_list, 1))

    #     return flows_backward2, flows_forward2

    # def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
    #     '''Get flow between t and t+3 from (t,t+2) and (t+2,t+3).'''

    #     # backward
    #     d = flows_forward2[0].shape[1]
    #     flows_backward3 = []
    #     for flows, flows2 in zip(flows_backward, flows_backward2):
    #         flow_list = []
    #         for i in range(d - 1, 0, -1):
    #             flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
    #             flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
    #             flow_list.insert(0, flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i+3 to i
    #         flows_backward3.append(torch.stack(flow_list, 1))

    #     # forward
    #     flows_forward3 = []
    #     for flows, flows2 in zip(flows_forward, flows_forward2):
    #         flow_list = []
    #         for i in range(2, d + 1):
    #             flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
    #             flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
    #             flow_list.append(flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1)))  # flow from i-3 to i
    #         flows_forward3.append(torch.stack(flow_list, 1))

    #     return flows_backward3, flows_forward3

    # def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
    #     '''Parallel feature warping for 2 frames.'''

    #     # backward
    #     n = x.size(1)
    #     x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
    #     for i in range(n - 1, 0, -1):
    #         x_i = x[:, i, ...]
    #         flow = flows_backward[:, i - 1, ...]
    #         x_backward.insert(0, flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i+1 aligned towards i

    #     # forward
    #     x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
    #     for i in range(0, n - 1):
    #         x_i = x[:, i, ...]
    #         flow = flows_forward[:, i, ...]
    #         x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), 'nearest4')) # frame i-1 aligned towards i

    #     return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    def forward_features(self, x, flows_backward, flows_forward):
        '''Main network for feature extraction.'''

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

class Stage(nn.Module):

    def __init__(self,
                 in_dim,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mul_attn_ratio=0.75,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 pa_frames=2,
                 deformable_groups=16,
                 reshape=None,
                 max_residue_magnitude=10,
                 ):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames

        # reshape the tensor
        if reshape == 'none':
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                                         nn.LayerNorm(dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                                         nn.LayerNorm(4 * in_dim), nn.Linear(4 * in_dim, dim),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                                         nn.LayerNorm(in_dim // 4), nn.Linear(in_dim // 4, dim),
                                         Rearrange('n d h w c -> n c d h w'))

        # mutual and self attention
        self.residual_group1 = TMSAG(dim=dim,
                                     depth=int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=(2, window_size[1], window_size[2]),
                                     mut_attn=True,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     )
        self.linear1 = nn.Linear(dim, dim)

        # only self attention
        self.residual_group2 = TMSAG(dim=dim,
                                     depth=depth - int(depth * mul_attn_ratio),
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mut_attn=False,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     )
        self.linear2 = nn.Linear(dim, dim)

        # parallel warping
        self.pa_deform = DCNv2PackFlowGuided(dim, dim, 3, padding=1, deformable_groups=deformable_groups,
                                                max_residue_magnitude=max_residue_magnitude, pa_frames=pa_frames)
        self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def forward(self, x, flows_backward, flows_forward):
        # print(x.shape)
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        if self.pa_frames:
            x = x.transpose(1, 2)
            if self.pa_frames == 2:
                x_backward, x_forward = get_aligned_feature_2frames(x, flows_backward, flows_forward, self.pa_deform)
            # elif self.pa_frames == 4:
            #     x_backward, x_forward = self.get_aligned_feature_4frames(x, flows_backward, flows_forward)
            # elif self.pa_frames == 6:
            #     x_backward, x_forward = self.get_aligned_feature_6frames(x, flows_backward, flows_forward)

            x = self.pa_fuse(torch.cat([x, x_backward, x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        return x

    # def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
    #     '''Parallel feature warping for 2 frames.'''

    #     # backward
    #     n = x.size(1)
    #     x_backward = [torch.zeros_like(x[:, -1, ...])]
    #     for i in range(n - 1, 0, -1):
    #         x_i = x[:, i, ...]
    #         flow = flows_backward[0][:, i - 1, ...]
    #         x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
    #         x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

    #     # forward
    #     x_forward = [torch.zeros_like(x[:, 0, ...])]
    #     for i in range(0, n - 1):
    #         x_i = x[:, i, ...]
    #         flow = flows_forward[0][:, i, ...]
    #         x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
    #         x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

    #     return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    # def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
    #     '''Parallel feature warping for 4 frames.'''

    #     # backward
    #     n = x.size(1)
    #     x_backward = [torch.zeros_like(x[:, -1, ...])]
    #     for i in range(n, 1, -1):
    #         x_i = x[:, i - 1, ...]
    #         flow1 = flows_backward[0][:, i - 2, ...]
    #         if i == n:
    #             x_ii = torch.zeros_like(x[:, n - 2, ...])
    #             flow2 = torch.zeros_like(flows_backward[1][:, n - 3, ...])
    #         else:
    #             x_ii = x[:, i, ...]
    #             flow2 = flows_backward[1][:, i - 2, ...]

    #         x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
    #         x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
    #         x_backward.insert(0,
    #             self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]))

    #     # forward
    #     x_forward = [torch.zeros_like(x[:, 0, ...])]
    #     for i in range(-1, n - 2):
    #         x_i = x[:, i + 1, ...]
    #         flow1 = flows_forward[0][:, i + 1, ...]
    #         if i == -1:
    #             x_ii = torch.zeros_like(x[:, 1, ...])
    #             flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
    #         else:
    #             x_ii = x[:, i, ...]
    #             flow2 = flows_forward[1][:, i, ...]

    #         x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
    #         x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
    #         x_forward.append(
    #             self.pa_deform(torch.cat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]))

    #     return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]

    # def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
    #     '''Parallel feature warping for 6 frames.'''

    #     # backward
    #     n = x.size(1)
    #     x_backward = [torch.zeros_like(x[:, -1, ...])]
    #     for i in range(n + 1, 2, -1):
    #         x_i = x[:, i - 2, ...]
    #         flow1 = flows_backward[0][:, i - 3, ...]
    #         if i == n + 1:
    #             x_ii = torch.zeros_like(x[:, -1, ...])
    #             flow2 = torch.zeros_like(flows_backward[1][:, -1, ...])
    #             x_iii = torch.zeros_like(x[:, -1, ...])
    #             flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
    #         elif i == n:
    #             x_ii = x[:, i - 1, ...]
    #             flow2 = flows_backward[1][:, i - 3, ...]
    #             x_iii = torch.zeros_like(x[:, -1, ...])
    #             flow3 = torch.zeros_like(flows_backward[2][:, -1, ...])
    #         else:
    #             x_ii = x[:, i - 1, ...]
    #             flow2 = flows_backward[1][:, i - 3, ...]
    #             x_iii = x[:, i, ...]
    #             flow3 = flows_backward[2][:, i - 3, ...]

    #         x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i+1 aligned towards i
    #         x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i+2 aligned towards i
    #         x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i+3 aligned towards i
    #         x_backward.insert(0,
    #                           self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
    #                                          x[:, i - 3, ...], [flow1, flow2, flow3]))

    #     # forward
    #     x_forward = [torch.zeros_like(x[:, 0, ...])]
    #     for i in range(0, n - 1):
    #         x_i = x[:, i, ...]
    #         flow1 = flows_forward[0][:, i, ...]
    #         if i == 0:
    #             x_ii = torch.zeros_like(x[:, 0, ...])
    #             flow2 = torch.zeros_like(flows_forward[1][:, 0, ...])
    #             x_iii = torch.zeros_like(x[:, 0, ...])
    #             flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
    #         elif i == 1:
    #             x_ii = x[:, i - 1, ...]
    #             flow2 = flows_forward[1][:, i - 1, ...]
    #             x_iii = torch.zeros_like(x[:, 0, ...])
    #             flow3 = torch.zeros_like(flows_forward[2][:, 0, ...])
    #         else:
    #             x_ii = x[:, i - 1, ...]
    #             flow2 = flows_forward[1][:, i - 1, ...]
    #             x_iii = x[:, i - 2, ...]
    #             flow3 = flows_forward[2][:, i - 2, ...]

    #         x_i_warped = flow_warp(x_i, flow1.permute(0, 2, 3, 1), 'bilinear')  # frame i-1 aligned towards i
    #         x_ii_warped = flow_warp(x_ii, flow2.permute(0, 2, 3, 1), 'bilinear')  # frame i-2 aligned towards i
    #         x_iii_warped = flow_warp(x_iii, flow3.permute(0, 2, 3, 1), 'bilinear')  # frame i-3 aligned towards i
    #         x_forward.append(self.pa_deform(torch.cat([x_i, x_ii, x_iii], 1), [x_i_warped, x_ii_warped, x_iii_warped],
    #                                         x[:, i + 1, ...], [flow1, flow2, flow3]))

    #     return [torch.stack(x_backward, 1), torch.stack(x_forward, 1)]