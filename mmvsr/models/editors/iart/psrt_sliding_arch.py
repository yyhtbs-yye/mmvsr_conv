import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .swin_T import SwinTransformerBasicLayer

from .iart_modules import PatchEmbed, PatchMerging, PatchUnEmbed

from .iart_utils import compute_mask

class RSTB(nn.Module):
    """Multi-frame Self-attention Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=(1, 1),
                 resi_connection='1conv',
                 num_frames=5):
        super(RSTB, self).__init__()

        self.dim = dim  
        self.input_resolution = input_resolution  
        self.num_frames=num_frames
        self.residual_group = SwinTransformerBasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            num_frames=num_frames)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size,attn_mask):
        n, c, t, h, w = x.shape
        x_ori = x
        x = self.residual_group(x, x_size,attn_mask)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x = self.conv(x)
        x = x.view(n, t, -1, h, w)
        x = self.patch_embed(x)
        x = x + x_ori
        return x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.num_frames * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        #flops += self.patch_unembed.flops()

        return flops
    
class SwinIRFM(nn.Module):

    def __init__(self,
                 img_size=64, patch_size=1, in_chans=3, embed_dim=96, 
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=(2, 7, 7), 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 num_frames=3,
                 **kwargs):
        super(SwinIRFM, self).__init__()
        num_in_ch = in_chans  #3
        num_out_ch = in_chans  #3
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        self.window_size = window_size
        self.shift_size = (window_size[0], window_size[1] // 2, window_size[2] // 2)
        self.num_frames = num_frames

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_first_feat = nn.Conv2d(num_feat, embed_dim, 3, 1, 1)
        #self.feature_extraction = make_layer(ResidualBlockNoBN, num_blocks_extraction, mid_channels=embed_dim)

        self.num_layers = len(depths)  
        self.embed_dim = embed_dim  
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim  
        self.mlp_ratio = mlp_ratio  

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            num_frames=num_frames,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  #64*64
        patches_resolution = self.patch_embed.patches_resolution  #[64,64]
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                num_frames=num_frames)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            # self.conv_before_upsample = nn.Sequential(
            #     nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
            #self.conv_before_recurrent_upsample = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            #self.upsample = Upsample(upscale, num_feat)
            #self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):

        x_size = (x.shape[3], x.shape[4])  #180,320
        h, w = x_size
        #print("x_size:",x_size)
        x = self.patch_embed(x)  #n,embed_dim,t,h,w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        attn_mask = compute_mask(self.num_frames,x_size,tuple(self.window_size),self.shift_size,x.device)
        for layer in self.layers:
            x = layer(x.contiguous(), x_size , attn_mask)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)  # b seq_len c

        x = x.permute(0, 1, 4, 2, 3).contiguous()

        return x

    def forward(self, x, ref=None):
        n, t, c, h, w = x.size()

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            if c == 3:
                x = x.view(-1, c, h, w)
                x = self.conv_first(x)
                #x = self.feature_extraction(x)
                x = x.view(n, t, -1, h, w)

            if c == 64:
                x = x.view(-1, c, h, w)
                x = self.conv_first_feat(x)
                x = x.view(n, t, -1, h, w)

            x_center = x[:, t // 2, :, :, :].contiguous()
            feats = self.forward_features(x)

            x = self.conv_after_body(feats[:, t // 2, :, :, :]) + x_center
            if ref:
                x = self.conv_before_upsample(x)
            #x = self.conv_last(self.upsample(x))

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        #flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i,layer in enumerate(self.layers):
            layer_flop=layer.flops()
            flops += layer_flop
            print(i,layer_flop / 1e9)


        flops += h * w * self.num_frames * self.embed_dim
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        #flops += self.upsample.flops()
        return flops

if __name__ == '__main__':
    upscale = 4
    window_size = (2, 8, 8)
    height = (1024 // upscale // window_size[1] + 1) * window_size[1]
    width = (1024 // upscale // window_size[2] + 1) * window_size[2]

    model = SwinIRFM(
        img_size=height,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=window_size,
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=2,
        img_range=1.,
        upsampler='pixelshuffle',
    )

    print(model)
    #print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 5, 3, height, width))
    x = model(x)
    print(x.shape)