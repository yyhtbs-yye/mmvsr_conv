import torch
from mmengine.model.weight_init import trunc_normal_
import torch.nn as nn

from mmvsr.models.basicsr_utils.positional_encoding import get_sine_position_encoding, get_position_index
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
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_mutual = use_mutual

        # self attention with relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        n_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # mutual attention with sine position encoding
        if self.use_mutual:
            self.register_buffer("position_bias",
                                 get_sine_position_encoding(window_size[1:], dim // 2, normalize=True))
            self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(2 * dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        w_B, N, C = x.shape
        qkv = self.qkv_self(x).reshape(w_B, N, 3, 
                                       self.n_heads, 
                                       C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each size = w_B, nH, N, C/nH

        x_out = self.attention(q, k, v, mask, (w_B, N, C), 
                               relative_position_encoding=True)

        # mutual attention
        if self.use_mutual:
            qkv = self.qkv_mut(x + self.position_bias.repeat(1, 2, 1)).reshape(w_B, N, 3, self.n_heads,
                                                                               C // self.n_heads).permute(2, 0, 3, 1,
                                                                                                            4)
            # w_B, nH, N/2, C
            (q1, q2) = torch.chunk(qkv[0], 2, dim=2) 
            (k1, k2) = torch.chunk(qkv[1], 2, dim=2) 
            (v1, v2) = torch.chunk(qkv[2], 2, dim=2)  
            
            x1_aligned = self.attention(q2, k1, v1, mask, (w_B, N // 2, C), relative_position_encoding=False)
            x2_aligned = self.attention(q1, k2, v2, mask, (w_B, N // 2, C), relative_position_encoding=False)
            x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):

        w_B, N, C = x_shape

        beta = (q * self.scale) @ k.transpose(-2, -1)

        if relative_position_encoding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)     # Wd*Wh*Ww, Wd*Wh*Ww,nH
            beta = beta + relative_position_bias.permute(2, 0, 1).unsqueeze(0)          # w_B, nH, N, N

        if mask is None:
            alpha = self.softmax(beta)
        else:
            nW = mask.shape[0]
            alpha = alpha.view(w_B // nW, nW, self.n_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            alpha = alpha.view(-1, self.n_heads, N, N)
            alpha = self.softmax(alpha)

        x = (alpha @ v).transpose(1, 2).reshape(w_B, N, C)

        return x
