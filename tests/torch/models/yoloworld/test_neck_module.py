# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
import torch.nn as nn
from typing import List,Any,Optional,Tuple
from torch import Tensor
from typing import Any
from third_party.tt_forge_models.tools.utils import get_file
from third_party.tt_forge_models.yoloworld.pytorch import ModelVariant
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs
from loguru import logger

class ReduceLayers(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.reduce_layers = nn.ModuleList(
            [nn.Identity() for _ in range(num_layers)]
        )

    def forward(self, x, idx: int = 0):
        return self.reduce_layers[idx](x)

    def __getitem__(self, idx):
        return self.reduce_layers[idx]

class UpsampleLayers(nn.Module):
    def __init__(self, num_layers: int = 2, scale_factor: int = 2):
        super().__init__()
        self.upsample_layers = nn.ModuleList(
            [
                nn.Upsample(scale_factor=scale_factor, mode="nearest")
                for _ in range(num_layers)
            ]
        )

    def __getitem__(self, idx):
        return self.upsample_layers[idx]
    def forward(self, x, idx: int):
        return self.upsample_layers[idx](x)

class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
        activation=True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        self.batch_norm2d = nn.BatchNorm2d(
            out_channels,
            eps=0.001,
            momentum=0.03,
            affine=True,
            track_running_stats=True
        )

        self.activate = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm2d(x)
        x = self.activate(x)
        return x

class DarknetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, add_identity=True):
        super().__init__()
        self.add_identity = add_identity and (in_channels == out_channels)

        # Bottleneck conv layers
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out  

class MaxSigmoidAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_heads: int = 1,
        with_scale: bool = False,
        use_einsum: bool = True,
    ) -> None:
        super().__init__()

        assert (
            out_channels % num_heads == 0
            and embed_channels % num_heads == 0
        ), "out_channels and embed_channels should be divisible by num_heads."

        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        # Optional embedding conv
        self.embed_conv = (
            ConvModule(
                in_channels,
                embed_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=False,
            )
            if embed_channels != in_channels
            else None
        )

        # Guide projection
        self.guide_fc = nn.Linear(guide_channels, embed_channels)

        # Attention bias & scale
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.scale = (
            nn.Parameter(torch.ones(1, num_heads, 1, 1))
            if with_scale
            else 1.0
        )

        # Output projection (always ConvModule)
        self.project_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            activation=False,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        B, _, H, W = x.shape

        # Guide embedding
        guide = self.guide_fc(guide)
        guide = guide.reshape(
            B, -1, self.num_heads, self.head_channels
        )

        # Feature embedding
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(
            B, self.num_heads, self.head_channels, H, W
        )

        # Attention computation
        if self.use_einsum:
            attn_weight = torch.einsum(
                "bmchw,bnmc->bmhwn", embed, guide
            )
        else:
            batch, m, channel, height, width = embed.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(
                batch, m, height, width, -1
            )

        # Max over guide dimension
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels ** 0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        # Apply attention
        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)

        return x

class CSPLayerWithTwoConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
    ) -> None:
        super().__init__()

        self.mid_channels = int(out_channels * expand_ratio)

        # First split conv
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )

        # Bottleneck blocks
        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                add_identity=add_identity,
            )
            for _ in range(num_blocks)
        )

        # Final fusion conv
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.main_conv(x)

        # Split into two parts
        x_main = list(
            x_main.split((self.mid_channels, self.mid_channels), dim=1)
        )

        # Sequential bottlenecks
        x_main.extend(block(x_main[-1]) for block in self.blocks)

        # Fuse
        return self.final_conv(torch.cat(x_main, dim=1))

class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        num_heads: int = 1,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        with_scale: bool = False,
        add_identity: bool = True,
        use_einsum: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
        )

        self.final_conv = ConvModule(
            (3 + num_blocks) * self.mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=True,
        )

        # Attention block
        self.attn_block = MaxSigmoidAttnBlock(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            use_einsum=use_einsum,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        x_main = self.main_conv(x)

        # Split
        x_main = list(
            x_main.split((self.mid_channels, self.mid_channels), dim=1)
        )

        # Bottlenecks
        x_main.extend(block(x_main[-1]) for block in self.blocks)

        # Attention branch
        x_main.append(self.attn_block(x_main[-1], guide))

        # Fuse
        return self.final_conv(torch.cat(x_main, dim=1))

def make_round(value, factor):
    return max(int(round(value * factor)), 1)

def make_divisible(value, factor):
    return max(int(round(value * factor)), 1)

class YOLOWorldPAFPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        guide_channels: int,
        embed_channels: List[int],
        num_heads: List[int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,  
    ) -> None:
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor

        num_levels = len(in_channels)

        self.num_blocks = make_round(num_csp_blocks, deepen_factor)
        self.upsample_feats_cat_first = True

        # --------------------
        # Reduce / Upsample
        # --------------------
        self.reduce_layers = ReduceLayers(num_levels)
        self.upsample_layers = UpsampleLayers(num_levels - 1)

        # --------------------
        # Top-down layers
        # --------------------
        self.top_down_layers = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            in_ch = make_divisible(
                in_channels[idx] + in_channels[idx - 1],
                widen_factor,
            )
            out_ch = make_divisible(
                out_channels[idx - 1],
                widen_factor,
            )

            self.top_down_layers.append(
                MaxSigmoidCSPLayerWithTwoConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    guide_channels=guide_channels,
                    embed_channels=make_round(
                        embed_channels[idx - 1], widen_factor
                    ),
                    num_heads=make_round(
                        num_heads[idx - 1], widen_factor
                    ),
                    num_blocks=self.num_blocks,
                    add_identity=False,
                )
            )

        # --------------------
        # Downsample layers
        # --------------------
        self.downsample_layers = nn.ModuleList()
        for i in range(num_levels - 1):
            ch = make_divisible(out_channels[i], widen_factor)
            self.downsample_layers.append(
                ConvModule(
                    ch,
                    ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )

        # --------------------
        # Bottom-up layers
        # --------------------
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(num_levels - 1):
            in_ch = make_divisible(
                out_channels[idx] + out_channels[idx + 1],
                widen_factor,
            )
            out_ch = make_divisible(
                out_channels[idx + 1],
                widen_factor,
            )

            self.bottom_up_layers.append(
                MaxSigmoidCSPLayerWithTwoConv(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    guide_channels=guide_channels,
                    embed_channels=make_round(
                        embed_channels[idx + 1], widen_factor
                    ),
                    num_heads=make_round(
                        num_heads[idx + 1], widen_factor
                    ),
                    num_blocks=self.num_blocks,
                    add_identity=False,
                )
            )

        # --------------------
        # Output layers
        # --------------------
        self.out_layers = nn.ModuleList(
            [nn.Identity() for _ in range(num_levels)]
        )

    def forward(
        self, img_feats: List[Tensor], txt_feats: Tensor = None
    ) -> Tuple[Tensor, ...]:

        assert len(img_feats) == len(self.in_channels)

        # --------------------
        # Reduce
        # --------------------
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # --------------------
        # Top-down path
        # --------------------
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]

            upsample_feat = self.upsample_layers[
                len(self.in_channels) - 1 - idx
            ](feat_high)

            if self.upsample_feats_cat_first:
                top_down_in = torch.cat([upsample_feat, feat_low], dim=1)
            else:
                top_down_in = torch.cat([feat_low, upsample_feat], dim=1)

            inner_out = self.top_down_layers[
                len(self.in_channels) - 1 - idx
            ](top_down_in, txt_feats)

            inner_outs.insert(0, inner_out)

        # --------------------
        # Bottom-up path
        # --------------------
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], dim=1),
                txt_feats,
            )
            outs.append(out)

        # --------------------
        # Output
        # --------------------
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)

variants = [
    ModelVariant.SMALL_1,
    ModelVariant.SMALL_2,
    ModelVariant.MEDIUM_1,
    ModelVariant.MEDIUM_2,
    ModelVariant.LARGE_1,
    ModelVariant.LARGE_2,
    ModelVariant.XLARGE_1
    ]
@pytest.mark.parametrize("variant",variants)
def test_yoloworld_neck_module(variant,guide_channels=512,in_channels = [256, 512, 1024],out_channels = [256, 512, 1024],embed_channels = [128, 256, 512],num_heads = [4,8,16],deepen_factor=0.33,widen_factor=0.5):
    """Test neck_module  for real inputs"""

    class Neck(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.neck = model

        def forward(self, f1, f2, f3, txt_feats):
            img_feats = [f1, f2, f3]
            return self.neck(img_feats, txt_feats)

    print("cvariant is ",variant,type(variant),type(variant.value))
    # if variant.value in {"small_640","small_1280"}:
    #     in_channels[3] = 1024
    #     out_channels = [256, 512, 1024]
    #     embed_channels = [128, 256, 512]
    #     num_heads = [4,8,16]
    #     deepen_factor=0.33
    #     widen_factor=0.5
    if variant.value in {"medium_640","medium_1280"}:
        in_channels[2] = 768
        out_channels[2] = 768
        embed_channels[2] =  384
        num_heads[2] = 12
        deepen_factor=0.67
        widen_factor=0.75
    elif variant.value in {"large_640","large_1280","xlarge_640"}:
        in_channels[2] = 512
        out_channels[2] = 512
        embed_channels[2] =  256
        num_heads[2] = 8
        deepen_factor=1.0
        if variant.value=="xlarge_640":
            widen_factor=1.25
        else:
            widen_factor=1.0
    # elif variant.value =="xlarge_640":
    #     in_channels = [256, 512, 512]
    #     out_channels = [256, 512, 512]
    #     embed_channels = [128, 256, 256]
    #     num_heads = [4,8,8]
    #     deepen_factor=1.0
    #     widen_factor=1.25

    custom_model = YOLOWorldPAFPN(in_channels=in_channels,out_channels=out_channels,guide_channels=guide_channels,embed_channels=embed_channels, num_heads=num_heads,deepen_factor=deepen_factor,widen_factor=widen_factor)
    custom_model.eval()
    checkpoint = torch.load(str(get_file(f"test_files/pytorch/yoloworld/{variant}.pth")))
    neck_state_dict = {
    k.replace("neck.", "", 1).replace(".bn.", ".batch_norm2d."): v
    for k, v in checkpoint['state_dict'].items()
    if k.startswith("neck.")
    }
    custom_model.load_state_dict(neck_state_dict,strict=False)
    custom_model.to(torch.bfloat16)
    print("inputs shapes are",torch.load(f"{variant}_inputs.pt")["img_feats_0"].shape,torch.load(f"{variant}_inputs.pt")["img_feats_1"].shape,torch.load(f"{variant}_inputs.pt")["img_feats_2"].shape,torch.load(f"{variant}_inputs.pt")["txt_feats"].shape)
    run_op_test_with_saved_inputs(
        Neck(custom_model),
        [
        torch.load(f"{variant}_inputs.pt")["img_feats_0"],  # f1
        torch.load(f"{variant}_inputs.pt")["img_feats_1"],  # f2
        torch.load(f"{variant}_inputs.pt")["img_feats_2"],  # f3
        torch.load(f"{variant}_inputs.pt")["txt_feats"],       # txt_feats
        ],
        framework=Framework.TORCH,
    )