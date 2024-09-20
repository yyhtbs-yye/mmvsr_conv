import torch
import torch.nn.functional as F
from torch import nn

from mmvsr.models.editors.d3dnet.d3d_modules import DeformConvPack_d

class FeaturePyramidNetworks(nn.Module):
    def __init__(self, in_channel=3, nf=64):
        super(FeaturePyramidNetworks, self).__init__()
        self.in_channel = in_channel

        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.upsample1 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.upsample2 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.downsample1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
        self.downsample2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)

        self.resd3d1 = ResBlock_d3d(nf)
        self.resd3d2 = ResBlock_d3d(nf)
        self.resd3d3 = ResBlock_d3d(nf)

        self.resd3d4 = ResBlock_d3d(nf)
        self.resd3d5 = ResBlock_d3d(nf)
        self.resd3d6 = ResBlock_d3d(nf)

    def forward(self, x):
        input = self.input(x)

        resout1 = self.resd3d1(input)
        downout1 = self.downsample1(resout1)
        resout2 = self.resd3d2(downout1[0])
        downout2 = self.downsample2(resout2)
        resout3 = self.resd3d3(downout2[0])

        resout4 = self.resd3d4(resout3)
        s1 = downout1[0].size()
        upout1 = self.upsample1(resout4, downout2[1], (s1[0], s1[1], s1[2], s1[3], s1[4]))
        resout5 = self.resd3d5(upout1 + resout2)
        s2 = x.size()
        upout2 = self.upsample2(resout5, downout1[1], (s2[0], s2[1], s2[2], s2[3], s2[4]))
        resout6 = self.resd3d6(upout2 + resout1)

        out = resout6 + input

        out = self.output(out)

        return out


class ResBlock_d3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_d3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class ResBlock_c2d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_c2d, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x
