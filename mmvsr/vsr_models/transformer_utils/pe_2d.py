
import torch
import math

def get_sine_position_encoding(HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    """ Get sine position encoding """

    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")

    if scale is None:
        scale = 2 * math.pi

    not_mask = torch.ones([1, HW[0], HW[1]])
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    # BxCxHxW
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos_embed.flatten(2).permute(0, 2, 1).contiguous()
