import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .tmsa_modules import TMSAG

# from mmvsr.models.basicsr_archs.x_attentions import GLUAttention

from .parallel_warper import ParallelWarper

class UnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 n_mma_blocks, n_msa_blocks,
                 n_heads, window_size, 
                 mlp_ratio, qkv_bias, qk_scale, 
                 drop_path, norm_layer,
                 deformable_groups, kernel_size, padding, 
                 max_residue_magnitude,
                 n_frames,
                 reshape=None,
                 ):
        super(UnetBlock, self).__init__()

        self.n_frames = n_frames

        # reshape the tensor
        if reshape == 'none':
            self.reshape = nn.Sequential(Rearrange('n c d h w -> n d h w c'),
                                         nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'down':
            self.reshape = nn.Sequential(Rearrange('n c d (h neih) (w neiw) -> n d h w (neiw neih c)', neih=2, neiw=2),
                                         nn.LayerNorm(4 * in_channels), nn.Linear(4 * in_channels, out_channels),
                                         Rearrange('n d h w c -> n c d h w'))
        elif reshape == 'up':
            self.reshape = nn.Sequential(Rearrange('n (neiw neih c) d h w -> n d (h neih) (w neiw) c', neih=2, neiw=2),
                                         nn.LayerNorm(in_channels // 4), nn.Linear(in_channels // 4, out_channels),
                                         Rearrange('n d h w c -> n c d h w'))

        # mutual and self attention
        self.residual_group1 = TMSAG(n_channels=out_channels, n_blocks=n_mma_blocks, 
                                     n_heads=n_heads, window_size=(2, window_size[1], window_size[2]), 
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                     drop_path=drop_path, norm_layer=norm_layer, use_mutual=True,
        )
        
        self.linear1 = nn.Linear(out_channels, out_channels)

        # only self attention
        self.residual_group2 = TMSAG(n_channels=out_channels, n_blocks=n_msa_blocks, 
                                     n_heads=n_heads, window_size=window_size, 
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                     drop_path=drop_path, norm_layer=norm_layer, use_mutual=False,
        )
        self.linear2 = nn.Linear(out_channels, out_channels)

        # parallel warping
        self.pw = ParallelWarper(out_channels, n_frames=n_frames, 
                                 deformable_groups=deformable_groups, 
                                 kernel_size=kernel_size, padding=padding, 
                                 max_residue_magnitude=max_residue_magnitude)

        self.pa_fuse = GLUAttention(out_channels * (1 + 2), out_channels * (1 + 2), out_channels)

    def forward(self, x, flows_backward, flows_forward):
        # print(x.shape)
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).transpose(1, 4)).transpose(1, 4) + x
        x = self.linear2(self.residual_group2(x).transpose(1, 4)).transpose(1, 4) + x

        x = x.transpose(1, 2)
        x_backward, x_forward = self.pw(x, flows_backward, flows_forward)
        x = self.pa_fuse(torch.cat([x, 
                                    x_backward, 
                                    x_forward], 2).permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)

        return x
