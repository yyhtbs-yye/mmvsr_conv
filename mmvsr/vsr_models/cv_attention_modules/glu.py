import torch
from torch import nn as nn
from torch.nn import functional as F

class GLUAttention(nn.Module):
    """ Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_dim, h_dim=None, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        h_dim = h_dim or in_dim

        self.fc11 = nn.Linear(in_dim, h_dim)
        self.fc12 = nn.Linear(in_dim, h_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x
