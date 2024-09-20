import torch.nn as nn
import torch

class VanillaFRBlock(nn.Module):
    def __init__(self, ):
        super(VanillaFRBlock, self).__init__()
        self.prelu = nn.PReLU()
        self.SpatioConv = nn.Conv3d(3, 3, (1, 3, 3), 1, (0, 1, 1))
        self.TemporalConv = nn.Conv3d(3, 3, (3, 1, 1), 1, (1, 0, 0))

    def forward(self, x):
        f = self.prelu(x)
        f = self.SpatioConv(f)
        f = self.TemporalConv(f)
        return f + x
    
class PokerFRBlock(nn.Module):
    def __init__(self, in_channel=3, in_frames=5, mid_channels=None, has_relu=True):
        super(PokerFRBlock, self).__init__()

        self.in_channel = in_channel
        self.in_frames = in_frames

        if mid_channels is None: 
            self.mid_channels = in_channel
        else: 
            self.mid_channels = mid_channels
        
        self.prelu = nn.PReLU()
        self.SpatioConv_3 = nn.Conv3d(in_channel, mid_channels, (1, 3, 3), 1, (0, 1, 1))
        self.SpatioConv_5 = nn.Conv3d(in_channel, mid_channels, (1, 5, 5), 1, (0, 2, 2))
        self.TemporalConv_3 = nn.Conv3d(mid_channels, mid_channels, (3, 1, 1), 1, (1, 0, 0))
        self.TemporalConv_5 = nn.Conv3d(mid_channels, mid_channels, (5, 1, 1), 1, (2, 0, 0))
        self.concatConv = nn.Conv3d(in_frames*4, in_frames, (1, 1, 1), 1, (0, 0, 0))

    def forward(self, x):

        u = self.prelu(x)

        x_3 = self.SpatioConv_3(u)
        x_5 = self.SpatioConv_5(u)

        # It is like a GoogleNet multi-resolution analysis
        x_3_3 = self.TemporalConv_3(x_3)
        x_3_5 = self.TemporalConv_5(x_3)
        x_5_3 = self.TemporalConv_3(x_5)
        x_5_5 = self.TemporalConv_5(x_5)
        x_cat = torch.cat([x_3_3, x_3_5, x_5_3, x_5_5], 2) # on dimension T so [B, C, T*4, H, W]

        # The code below is to aggregate temporal information 20 frames 
        # [5, 5, 5, 5] -> [5]
        x_cat = x_cat.permute(0, 2, 1, 3, 4)    # [B, T*4, C, H, W]
        f = self.concatConv(x_cat)              # [B, T, C, H, W]
        f = f.permute(0, 2, 1, 3, 4)            # [B, C, T, H, W]

        return x + f