import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmvsr.models.basicsr_utils.positional_encoding import get_sine_position_encoding, get_position_index
from mmengine.model.weight_init import trunc_normal_

from .attention_fcns import swin_attention

class WindowMutualSelfAttention(nn.Module):
    """ Window based multi-head mutual attention and self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        n_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        use_mutual (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(self, dim, window_size, n_heads, 
                 qkv_bias=False, qk_scale=None, 
                 use_mutual=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        head_dim = dim // n_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_mutual = use_mutual

        self.n_heads = n_heads

        # self attention with relative position bias
        self.rpe_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        n_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("rpe_index", get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # mutual attention with sine position encoding
        if self.use_mutual:
            self.register_buffer("pe",
                                 get_sine_position_encoding(window_size[1:], dim // 2, normalize=True))
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.rpe_table, std=.02)

    def forward(self, x, mask=None): # x: input features of shape (nW*B, N, C); mask: (0/-inf) of shape (nW, N, N) or None

        w_B, N, C = x.shape

        qkv = self.qkv_self(x).reshape(w_B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # each size = w_B, nH, N, C/nH

        x_out = swin_attention(q, k, v, mask, self.scale, self.rpe_table, self.rpe_index)

        if self.use_mutual: # MMA
            # Mutual Attention MMA, the window size is always [2, *, *], 
            # MMA will have more num_windows 'nW' than MSA (e.g., MMA has 3 times windows 
            # compared than the MSA of window size [6, *, *]). 
            # ALSO: Position Encoding is with Input not with QKV. 
            qkv = self.qkv_mut(x + self.pe.repeat(1, 2, 1))\
                                                          .reshape(w_B, N, 3, self.n_heads, C // self.n_heads)\
                                                              .permute(2, 0, 3, 1, 4)
            
            # w_B, nH, N/2, C, not sure why it guarantee the first frame and the second frame. 
            (q1, q2) = torch.chunk(qkv[0], 2, dim=2) 
            (k1, k2) = torch.chunk(qkv[1], 2, dim=2) 
            (v1, v2) = torch.chunk(qkv[2], 2, dim=2)  
            
            # No RPE enabled for MMA
            x1_aligned = swin_attention(q2, k1, v1, mask, self.scale, None, None)
            x2_aligned = swin_attention(q1, k2, v2, mask, self.scale, None, None)

            # concat(MMA, MSA) -> x_out
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

        # projection
        x = self.proj(x_out)

        return x
    