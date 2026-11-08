import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

from models.vssm import ChannelAttention, Downsample, OverlapPatchEmbed, Upsample, VSSM


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)


class MSACA(nn.Module):
    def __init__(self, dim, bias=False, **kwargs):
        super().__init__()
        self.dwconv3x3 = nn.Conv2d(dim, dim, kernel_size=3, dilation=1, stride=1, padding=1,
                                   groups=dim, bias=bias)
        self.dwconv5x5 = nn.Conv2d(dim, dim, kernel_size=3, dilation=2, stride=1, padding=2,
                                   groups=dim, bias=bias)
        self.dwconv7x7 = nn.Conv2d(dim, dim, kernel_size=3, dilation=3, stride=1, padding=3,
                                   groups=dim, bias=bias)

        self.dwconv3 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2,
                                 bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.ca = ChannelAttention(dim, squeeze_factor=30)

    def forward(self, x):
        x5 = self.dwconv5x5(x)
        x3_5 = x5 + self.dwconv3x3(x)
        x5_7 = x5 + self.dwconv7x7(x)

        x = self.dwconv3(torch.cat([x5_7, x3_5], dim=1))
        x = self.project_out(x)
        return self.ca(x)


class VSMDCB(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        expand: float = 2.0,
        **kwargs,
    ):
        super().__init__()

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VSSM(d_model=hidden_dim, d_state=d_state, expand=expand,
                                   dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = MSACA(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.temb_proj = nn.Linear(512, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

    def forward(self, input, x_size, timestep):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            self.temb_proj(timestep)
        ).chunk(6, dim=1)

        b, l, c = input.shape
        input = input.view(b, *x_size, c).contiguous()

        x = self.ln_1(input)
        x = input * self.skip_scale + gate_msa.unsqueeze(1).unsqueeze(1) * self.drop_path(
            self.self_attention(modulate(x, shift_msa, scale_msa))
        )
        x = x * self.skip_scale2 + gate_mlp.unsqueeze(1).unsqueeze(1) * self.conv_blk(
            modulate(self.ln_2(x), shift_mlp, scale_mlp).permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1).contiguous()

        return x.view(b, -1, c).contiguous()


class VSMDCBStage(nn.Module):
    def __init__(self, hidden_dim, drop_path, norm_layer, attn_drop_rate, expand, d_state,
                 num_blocks):
        super().__init__()
        self.vss_blocks = nn.ModuleList([
            VSMDCB(
                hidden_dim=hidden_dim,
                drop_path=drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                expand=expand,
                d_state=d_state,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x, x_size, time):
        for layer in self.vss_blocks:
            x = layer(x, x_size, time)
        return x


class NoiseEstimationNetwork(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(1, 1, 1, 1, 1, 1, 1),
        num_refinement_blocks=1,
        mlp_ratio=1.5,
        drop_path_rate=0.0,
        bias=False,
    ):
        super().__init__()
        base_d_state = 4

        self.patch_embed = OverlapPatchEmbed(inp_channels * 2, dim)
        self.time_embed = TimestepEmbedder(512)

        def build_stage(channels, block_count):
            return nn.ModuleList([
                VSMDCBStage(
                    hidden_dim=channels,
                    drop_path=drop_path_rate,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    expand=mlp_ratio,
                    d_state=int(base_d_state * 2 ** 2),
                    num_blocks=block_count,
                )
            ])

        self.encoder_level1 = build_stage(dim, num_blocks[0])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = build_stage(int(dim * 2 ** 1), num_blocks[1])
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = build_stage(int(dim * 2 ** 2), num_blocks[2])
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = build_stage(int(dim * 2 ** 3), num_blocks[3])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1,
                                            bias=bias)
        self.decoder_level3 = build_stage(int(dim * 2 ** 2), num_blocks[4])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1,
                                            bias=bias)
        self.decoder_level2 = build_stage(int(dim * 2 ** 1), num_blocks[5])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = build_stage(int(dim * 2 ** 1), num_blocks[6])

        self.refinement = build_stage(int(dim * 2 ** 1), num_refinement_blocks)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1,
                                bias=bias)

    def forward(self, x, t, cond):
        _, _, h, w = x.shape
        time = self.time_embed(t)

        out_enc_level1 = self.patch_embed(torch.cat([x, cond], dim=1))
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [h, w], time)

        out_enc_level2 = self.down1_2(out_enc_level1, h, w)
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [h // 2, w // 2], time)

        out_enc_level3 = self.down2_3(out_enc_level2, h // 2, w // 2)
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, [h // 4, w // 4], time)

        latent = self.down3_4(out_enc_level3, h // 4, w // 4)
        for layer in self.latent:
            latent = layer(latent, [h // 8, w // 8], time)

        inp_dec_level3 = self.up4_3(latent, h // 8, w // 8)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 2)
        inp_dec_level3 = rearrange(inp_dec_level3, "b (h w) c -> b c h w", h=h // 4, w=w // 4).contiguous()
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = rearrange(inp_dec_level3, "b c h w -> b (h w) c").contiguous()

        out_dec_level3 = inp_dec_level3
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, [h // 4, w // 4], time)

        inp_dec_level2 = self.up3_2(out_dec_level3, h // 4, w // 4)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=h // 2, w=w // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()

        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [h // 2, w // 2], time)

        inp_dec_level1 = self.up2_1(out_dec_level2, h // 2, w // 2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)

        out_dec_level1 = inp_dec_level1
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [h, w], time)

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [h, w], time)

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        return self.output(out_dec_level1) + x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))
