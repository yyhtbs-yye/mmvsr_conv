import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from .attention_modules import WindowMutualSelfAttention

from mmvsr.models.basicsr_archs.x_attentions import GLUAttention
from mmcv.cnn.bricks.drop import DropPath
from mmvsr.models.basicsr_utils.vidt_proc import window_partition, window_reverse, get_window_size, compute_mask

class TMSA(nn.Module): # Temporal Mutual Self Attention

    def __init__(self, n_channels, n_heads,
                 window_size=(6, 8, 8),
                 shift_size=(0, 0, 0), 
                 mlp_ratio=2.,                          # Ratio of mlp hidden n_channels to embedding n_channels.
                 qkv_bias=True, qk_scale=None,          # Override default qk scale of head_dim ** -0.5 if set.
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_mutual=True,
                 ):
        
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.prenorm = norm_layer(n_channels)

        self.attention = WindowMutualSelfAttention(n_channels, window_size=self.window_size, 
                                                   n_heads=n_heads, qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale, use_mutual=use_mutual)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(n_channels)
        
        self.mlp = GLUAttention(in_dim=n_channels, h_dim=int(n_channels * mlp_ratio), act_layer=act_layer)

    def _forward(self, x, mask):

        B, D, H, W, C = x.shape

        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        # Layer Norm input ``x``
        x = self.prenorm(x)                                                                                   

        # The ``tensor`` is padded to match multiples of the window size, ensuring 
        # that attention can be applied evenly across the data.

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape

        # Depending on the ``shift_size``, the tensor may be cyclically shifted to align 
        # different parts of the data under the attention mechanism, enhancing the model's 
        # ability to learn from various spatial relationships. [SWIN]

        if any(i > 0 for i in shift_size): # if exist any i in shift_size > 0
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            mask = None

        # The shifted and padded tensor is then split into smaller blocks or windows.
        x_windows = window_partition(shifted_x, window_size)  # ``x_windows`` is token of patches of size [B*nW, Wd*Wh*Ww, C]
        
        # The attention mechanism is applied within these windows.
        # Either self-attention or a combination of mutual and self-attention
        attn_windows = self.attention(x_windows, mask=mask)  # B*nW, Wd*Wh*Ww, C

        # After processing through the attention mechanism, the tensor 
        # blocks are reassembled.
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # If there was a cyclic shift applied initially, it is reversed to 
        # bring the tensor back to its original alignment.
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove the Paddings (may contains something?)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        x = self.drop_path(x)

        return x

    def forward(self, x, mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask: Attention mask for cyclic shift.
        """

        x = x + self._forward(x, mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class TMSAG(nn.Module):                                     # Temporal Mutual Self Attention Group (TMSAG).

    def __init__(self, n_channels, n_blocks, n_heads,
                 window_size=[6, 8, 8], shift_size=None,    # Shift size for mutual and self attention. Default: None.
                 use_mutual=True, mlp_ratio=2.,               # Ratio of mlp hidden n_channels to embedding n_channels. Default: 2.
                 qkv_bias=False, qk_scale=None,             # Override default qk scale of head_dim ** -0.5 if set 
                 drop_path=0., norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            TMSA(
                n_channels=n_channels,
                n_heads=n_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                use_mutual=use_mutual,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(n_blocks)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for block in self.blocks:
            x = block(x, mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RTMSA(nn.Module):                                     # Residual TMSA. Only used in stage 8.

    def __init__(self, n_channels, n_blocks, n_heads, 
                 window_size, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 ):
        super(RTMSA, self).__init__()
        self.n_channels = n_channels

        self.residual_group = TMSAG(n_channels=n_channels, n_blocks=n_blocks, n_heads=n_heads,
                                    window_size=window_size, use_mutual=False,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                    qk_scale=qk_scale, drop_path=drop_path,
                                    norm_layer=norm_layer,
                                    )

        self.linear = nn.Linear(n_channels, n_channels)

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)
