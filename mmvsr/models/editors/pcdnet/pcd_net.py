import torch
import torch.nn.functional as F
from torch import nn

from mmvsr.registry import MODELS
from mmvsr.models.utils import make_layer
from mmvsr.models.archs import PixelShufflePack
from mmcv.cnn import ConvModule

from .basic_modules import ResidualBlockNoBN

from .mini_pyramid_align import TripleScaleAlignment, DoubleScaleAlignment, SingleScaleAlignment
from .basic_pyramid_align import PyramidDeformableAlignment
from .edvrpcd_align import PCDAlignment
from .spynet_align import SPyNetAlignment
from .spynet_pw_align import SPyNetPwAlignment
from .pwcnet_align import PWCNetAlignment


@MODELS.register_module()
class De3QNet(nn.Module):
    def __init__(self, upscale_factor=4, in_frames=7,
                 in_channels=3, out_channels=3, pyramid_depth=4, 
                 preproc_config={'n_blocks': 3, 'n_channels': 256},
                 align_config={'n_channels': 256, 'deform_groups': 8},
                 postproc_config={'n_blocks': 6, 'n_channels': 128, 'kernel_size': 5}, 
                ):
        super(De3QNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels

        self.pre_feat_extract = nn.Sequential(
            ConvModule(in_channels, preproc_config['n_channels'], 1, padding=0),
            make_layer(
                ResidualBlockNoBN,
                preproc_config['n_blocks'],
                n_channels=preproc_config['n_channels'],
                kernel_size=3,
            ),
            # The line below is a shape transformer
            nn.Conv2d(preproc_config['n_channels'], align_config['n_channels'], 1, padding=0),
        )

        if pyramid_depth == 3:
            self.temporal_alignment = TripleScaleAlignment(n_channels=align_config['n_channels'], 
                                                           deform_groups=align_config['deform_groups'])
        elif pyramid_depth == 2:
            self.temporal_alignment = DoubleScaleAlignment(n_channels=align_config['n_channels'], 
                                                           deform_groups=align_config['deform_groups'])
        elif pyramid_depth == 1:
            self.temporal_alignment = SingleScaleAlignment(n_channels=align_config['n_channels'], 
                                                           deform_groups=align_config['deform_groups'])
        elif pyramid_depth == 4:
            self.temporal_alignment = PyramidDeformableAlignment(n_channels=align_config['n_channels'], 
                                                                 deform_groups=align_config['deform_groups'])
        elif pyramid_depth == "PCD": # Originally it is "5"
            self.temporal_alignment = PCDAlignment(n_channels=align_config['n_channels'], 
                                                                 deform_groups=align_config['deform_groups'])
        elif pyramid_depth == "SPyNet":
            self.temporal_alignment = SPyNetAlignment(n_channels=align_config['n_channels'], 
                                                                 deform_groups=align_config['deform_groups'])
        elif pyramid_depth == "PWCNet":
            self.temporal_alignment = PWCNetAlignment(n_channels=align_config['n_channels'], 
                                                                 deform_groups=align_config['deform_groups'])
        elif pyramid_depth == "SPyPwNet":
            self.temporal_alignment = SPyNetPwAlignment(n_channels=align_config['n_channels'], 
                                                                 deform_groups=align_config['deform_groups'])

        
        self.temporal_aggregation = nn.Conv2d(in_frames * align_config['n_channels'], 
                                              postproc_config['n_channels'], 
                                              1, 1, bias=True)
        
        self.post_feat_enhancement = nn.Sequential(
            make_layer(ResidualBlockNoBN,
                postproc_config['n_blocks'],
                n_channels=postproc_config['n_channels'],
                kernel_size=postproc_config['kernel_size'],
            )
        )
        
        self.upscale = nn.Sequential(
            PixelShufflePack(postproc_config['n_channels'], postproc_config['n_channels'], 
                             2, upsample_kernel=3),
            PixelShufflePack(postproc_config['n_channels'], postproc_config['n_channels'], 
                             2, upsample_kernel=3),
            nn.Conv2d(postproc_config['n_channels'], out_channels, 3, 1, 1, bias=False)
        )


    def forward(self, x):

        b, t, c, h, w = x.size()

        residual = F.interpolate(x[:, t // 2, :, :, :], 
                                 scale_factor=self.upscale_factor, 
                                 mode='bilinear',
                                 align_corners=False)

        out = self.pre_feat_extract(x.view(b * t, c, h, w)).view(b, t, -1, h, w)

        out = self.temporal_alignment(out)
        out = self.temporal_aggregation(out.view(b, -1, h, w))  # B, C, H, W
        out = self.post_feat_enhancement(out)

        out = self.upscale(out)
        out = torch.add(out, residual)

        return out 

