from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from models.vssm import ChannelAttention, Downsample, OverlapPatchEmbed, Upsample


class ChannelAttentionBlock(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class DynamicDepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, num_groups=1, bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."

        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias

        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size),
                                   requires_grad=True)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, 1),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()

        b, c, h, w = x.shape

        scale = self.proj(self.pool(x))
        scale = scale.reshape(b, self.num_groups, c, self.K, self.K)
        scale = torch.softmax(scale, dim=1)

        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = scale.reshape(b, self.num_groups, c)
            scale = torch.softmax(scale, dim=1)

            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(
            x.reshape(1, -1, h, w),
            weight=weight,
            padding=self.K // 2,
            groups=b * c,
            bias=bias,
        )

        return x.reshape(b, c, h, w)


class DDCB(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        kernel_size: int = 3,
        num_groups: int = 2,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = DynamicDepthwiseConv(dim=hidden_dim, kernel_size=kernel_size,
                                                   num_groups=hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ChannelAttentionBlock(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        b, l, c = input.shape
        input = input.view(b, *x_size, c).contiguous()
        x = self.ln_1(input)
        x_dconv = self.drop_path(self.self_attention(x))
        x_dconv = x_dconv.permute(0, 2, 3, 1).contiguous()
        x = input * self.skip_scale + x_dconv
        x = x * self.skip_scale2 + self.conv_blk(
            self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1).contiguous()
        return x.view(b, -1, c).contiguous()


class CGNet(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(1, 1, 1, 1),
        num_refinement_blocks=4,
        drop_path_rate=0.0,
        bias=False,
        kernel_size_array=(3, 3, 3, 3),
        num_groups_array=(4, 4, 4, 4),
    ):
        super().__init__()

        self.haze_aware = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.feature_validator = nn.Sequential(
            nn.Conv2d(out_channels + 1, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        def build_ddcg(channels, count):
            return nn.ModuleList([
                DDCB(
                    hidden_dim=channels,
                    kernel_size=kernel_size_array[i],
                    num_groups=num_groups_array[i],
                    drop_path=drop_path_rate,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(count)
            ])

        self.encoder_level1 = build_ddcg(dim, num_blocks[0])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = build_ddcg(int(dim * 2 ** 1), num_blocks[1])
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = build_ddcg(int(dim * 2 ** 2), num_blocks[2])
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = build_ddcg(int(dim * 2 ** 3), num_blocks[3])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1,
                                            bias=bias)
        self.decoder_level3 = build_ddcg(int(dim * 2 ** 2), num_blocks[2])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = build_ddcg(int(dim * 2 ** 1), num_blocks[1])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = build_ddcg(int(dim * 2 ** 1), num_blocks[0])

        self.refinement = build_ddcg(int(dim * 2 ** 1), num_refinement_blocks)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1,
                                bias=bias)

    def encode(self, inp_img):
        haze_map = self.haze_aware(inp_img)
        _, _, h, w = inp_img.shape

        out_enc_level1 = self.patch_embed(inp_img)
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [h, w])

        out_enc_level2 = self.down1_2(out_enc_level1, h, w)
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [h // 2, w // 2])

        out_enc_level3 = self.down2_3(out_enc_level2, h // 2, w // 2)
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [h // 4, w // 4])

        latent = self.down3_4(out_enc_level3, h // 4, w // 4)
        for layer in self.latent:
            latent = layer(latent, [h // 8, w // 8])

        return latent, out_enc_level1, out_enc_level2, out_enc_level3, haze_map, h, w

    def decode(self, latent, out_enc_level1, out_enc_level2, out_enc_level3, h, w):
        inp_dec_level3 = self.up4_3(latent, h // 8, w // 8)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w",
                                   h=h // 4, w=w // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()

        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [h // 4, w // 4])

        inp_dec_level2 = self.up3_2(out_dec_level3, h // 4, w // 4)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w",
                                   h=h // 2, w=w // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()

        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [h // 2, w // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, h // 2, w // 2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)

        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [h, w])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [h, w])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        return self.output(out_dec_level1)

    def forward(self, inp_img):
        latent, enc1, enc2, enc3, haze_map, h, w = self.encode(inp_img)
        dehazed = self.decode(latent, enc1, enc2, enc3, h, w)

        fusion_weight = self.feature_validator(torch.cat([dehazed, haze_map], dim=1))
        guidance = dehazed * fusion_weight + inp_img * (1 - fusion_weight) * (1 - haze_map)

        return guidance, dehazed.to(torch.float32)
