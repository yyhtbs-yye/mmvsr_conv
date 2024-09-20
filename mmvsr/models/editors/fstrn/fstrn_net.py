import torch
import torch.nn as nn
import torch.nn.functional as F

from .fstrn_modules import PokerFRBlock  # Assuming this module is defined elsewhere

class FSTRNet(nn.Module):
    def __init__(self, model_para, upscale_factor=4, frb_num=5):
        super(FSTRNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.model_para = model_para
        # self.lfe = self.create_bottle_net(has_relu=False)
        self.lfe = nn.Conv3d(3, 3, (3, 3, 3), 1, (1, 1, 1))

        # Using a list to store FRB blocks instead of nn.Sequential
        self.frb_num = frb_num
        self.FRB_blocks = nn.Sequential(
            [PokerFRBlock(has_relu=True) for _ in range(self.frb_num)]
        )

        self.prelu = nn.PReLU()
        
        self.lsr = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.ConvTranspose2d(3, 3, 8, 4, 2, 0),
            nn.Conv2d(3, 3, 3, 1, 1)
        )

    def forward(self, x):
        # Added by Yuhang, for 3D conv, the input is shaped in BCTHW not BTCHW (default mmvsr Input), 
        # so convert BTCHW -> BCTHW
        if x.size(2) == 3:
            x = torch.permute(x, (0, 2, 1, 3, 4))

        residual = F.interpolate(x[:, :, n // 2, :, :], 
                                    scale_factor=self.upscale_factor, 
                                    mode='bilinear', 
                                    align_corners=False)

        b, c, n, h, w = x.size()

        f_0 = self.lfe(x)

        f = self.FRB_blocks(f_0) + f_0

        f = self.prelu(f)

        f = F.dropout(f, 0.3)

        # Sum along the temporal channels (assuming temporal channel is at dim=2)
        f = torch.sum(f, 2)

        f = residual + self.lsr(f)

        return f