# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helpers for Z-Image VAE decoder submodule / stage sanity tests.

The full decoder OOM (see zimage_logs/decoder.log) is a single ``ttnn::subtract``
allocating ~3.77 GiB DRAM (314 MiB × 12 banks) during execution — equivalent to a
float32 tensor shaped [1, 1024, 1280, 720] at pipeline resolution, not model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    DTYPE,
    SEED,
    load_vae,
    make_latent_inputs,
)


@dataclass(frozen=True)
class StageSpec:
    """One isolatable decoder stage for TT vs CPU graph tests."""

    name: str
    input_key: str
    build_module: Callable[[], torch.nn.Module]


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class LatentPreprocessWrapper(torch.nn.Module):
    """Scaling/shift (+ optional post_quant_conv) — matches VAEDecoderWrapper / decode()."""

    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.scaling_factor = vae.config.scaling_factor
        self.shift_factor = vae.config.shift_factor
        self.vae_dtype = vae.dtype

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(dtype=self.vae_dtype)
        z = (z / self.scaling_factor) + self.shift_factor
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return z


class DecoderHeadWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder.conv_norm_out(x)
        x = self.decoder.conv_act(x)
        return self.decoder.conv_out(x)


class GroupNormOnlyWrapper(torch.nn.Module):
    """Isolate ``nn.GroupNorm`` for Playground-style d2 bisect."""

    def __init__(self, norm: torch.nn.GroupNorm):
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class DecoderPrefixWrapper(torch.nn.Module):
    """Run decoder from post_quant output through mid_block and N up_blocks."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        num_up_blocks: int,
        *,
        include_head: bool,
    ):
        super().__init__()
        if num_up_blocks < 0 or num_up_blocks > len(decoder.up_blocks):
            raise ValueError(f"num_up_blocks must be in [0, {len(decoder.up_blocks)}]")
        self.decoder = decoder
        self.num_up_blocks = num_up_blocks
        self.include_head = include_head

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder.conv_in(z)
        h = self.decoder.mid_block(h)
        for i in range(self.num_up_blocks):
            h = self.decoder.up_blocks[i](h)
        if self.include_head:
            h = self.decoder.conv_norm_out(h)
            h = self.decoder.conv_act(h)
            h = self.decoder.conv_out(h)
        return h


@torch.no_grad()
def trace_decoder_stages(vae: torch.nn.Module, latents: torch.Tensor) -> dict[str, torch.Tensor]:
    """CPU golden activations at each decoder boundary (bf16)."""
    z = latents.to(dtype=vae.dtype)
    z = (z / vae.config.scaling_factor) + vae.config.shift_factor
    stages: dict[str, torch.Tensor] = {"latents": latents, "latents_scaled": z}

    if vae.post_quant_conv is not None:
        z = vae.post_quant_conv(z)
    stages["decoder_in"] = z

    dec = vae.decoder
    h = dec.conv_in(z)
    stages["conv_in"] = h

    h = dec.mid_block(h)
    stages["mid_block"] = h

    for i, up_block in enumerate(dec.up_blocks):
        h = up_block(h)
        stages[f"up_block_{i}"] = h

    # Input to up_blocks[3].resnets[0].norm2 (after norm1 → silu → conv1).
    r0 = dec.up_blocks[3].resnets[0]
    _t = r0.norm1(stages["up_block_2"])
    _t = r0.nonlinearity(_t)
    stages["up3_resnet0_after_conv1"] = r0.conv1(_t)

    out = dec.conv_norm_out(h)
    out = dec.conv_act(out)
    out = dec.conv_out(out)
    stages["decoded"] = out
    return stages


def load_vae_and_latents(
    dtype: torch.dtype = DTYPE,
) -> tuple[torch.nn.Module, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(SEED)
    vae = load_vae(dtype).eval()
    latents = make_latent_inputs(dtype)
    stages = trace_decoder_stages(vae, latents)
    return vae, latents, stages


def build_stage_specs(vae: torch.nn.Module) -> list[StageSpec]:
    dec = vae.decoder
    specs: list[StageSpec] = [
        StageSpec(
            name="latent_preprocess",
            input_key="latents",
            build_module=lambda: LatentPreprocessWrapper(vae),
        ),
        StageSpec(
            name="conv_in",
            input_key="decoder_in",
            build_module=lambda: ModuleWrapper(dec.conv_in),
        ),
        StageSpec(
            name="mid_block",
            input_key="conv_in",
            build_module=lambda: ModuleWrapper(dec.mid_block),
        ),
    ]
    for i in range(len(dec.up_blocks)):
        block = dec.up_blocks[i]
        specs.append(
            StageSpec(
                name=f"up_block_{i}",
                input_key="mid_block" if i == 0 else f"up_block_{i - 1}",
                build_module=(lambda b=block: lambda: ModuleWrapper(b))(),
            )
        )
    specs.append(
        StageSpec(
            name="decoder_head",
            input_key=f"up_block_{len(dec.up_blocks) - 1}",
            build_module=lambda: DecoderHeadWrapper(dec),
        )
    )
    for n in range(len(dec.up_blocks) + 1):
        specs.append(
            StageSpec(
                name=f"prefix_through_up_{n}",
                input_key="decoder_in",
                build_module=(lambda n=n: lambda: DecoderPrefixWrapper(
                    dec, num_up_blocks=n, include_head=False
                ))(),
            )
        )
    specs.append(
        StageSpec(
            name="prefix_full_decoder",
            input_key="decoder_in",
            build_module=lambda: DecoderPrefixWrapper(
                dec, num_up_blocks=len(dec.up_blocks), include_head=True
            ),
        )
    )
    return specs


# Playground #4710 bisect: https://github.com/tenstorrent/tt-xla/issues/4710
UP_BLOCK_NORM_BISECT_INDEX = 3
RESNET_NORM_BISECT_INDEX = 0


def build_d1_skip_before_up3_resnet0_norm1(decoder: torch.nn.Module) -> torch.nn.Module:
    """d1: conv_in + mid + up_blocks[0..2]; return tensor entering up_blocks[3].resnets[0].norm1."""
    return DecoderPrefixWrapper(
        decoder,
        num_up_blocks=UP_BLOCK_NORM_BISECT_INDEX,
        include_head=False,
    )


def build_d2_up3_resnet0_norm1_only(decoder: torch.nn.Module) -> torch.nn.Module:
    """d2: only up_blocks[3].resnets[0].norm1 (GroupNorm 32, 256 @ 1280x720)."""
    norm1 = decoder.up_blocks[UP_BLOCK_NORM_BISECT_INDEX].resnets[
        RESNET_NORM_BISECT_INDEX
    ].norm1
    return GroupNormOnlyWrapper(norm1)


class DecoderPrefixThenNorm1Wrapper(torch.nn.Module):
    """d3: one graph — prefix through up_blocks[0..2] then up_blocks[3].resnets[0].norm1."""

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.prefix = DecoderPrefixWrapper(
            decoder,
            num_up_blocks=UP_BLOCK_NORM_BISECT_INDEX,
            include_head=False,
        )
        self.norm1 = decoder.up_blocks[UP_BLOCK_NORM_BISECT_INDEX].resnets[
            RESNET_NORM_BISECT_INDEX
        ].norm1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.prefix(z)
        return self.norm1(h)


# Stops inside up_blocks[3].resnets[0] for cumulative bisect after d3 passes.
RESNET0_STOP_NORM1 = "norm1"
RESNET0_STOP_POST_SILU = "post_silu"
RESNET0_STOP_CONV1 = "conv1"
RESNET0_STOP_NORM2 = "norm2"
RESNET0_STOP_FULL = "full"

RESNET0_CUMULATIVE_STOPS = (
    RESNET0_STOP_CONV1,
    RESNET0_STOP_NORM2,
    RESNET0_STOP_FULL,
)


class ResnetBlock2DPartialWrapper(torch.nn.Module):
    """Run ResnetBlock2D forward through ``stop`` (inclusive endpoint)."""

    def __init__(self, resnet: torch.nn.Module, stop: str):
        super().__init__()
        if stop not in (
            RESNET0_STOP_NORM1,
            RESNET0_STOP_POST_SILU,
            RESNET0_STOP_CONV1,
            RESNET0_STOP_NORM2,
            RESNET0_STOP_FULL,
        ):
            raise ValueError(f"Unknown stop: {stop}")
        self.resnet = resnet
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.resnet
        hidden_states = x
        input_tensor = x

        hidden_states = r.norm1(hidden_states)
        if self.stop == RESNET0_STOP_NORM1:
            return hidden_states

        hidden_states = r.nonlinearity(hidden_states)
        if self.stop == RESNET0_STOP_POST_SILU:
            return hidden_states

        hidden_states = r.conv1(hidden_states)
        if self.stop == RESNET0_STOP_CONV1:
            return hidden_states

        hidden_states = r.norm2(hidden_states)
        if self.stop == RESNET0_STOP_NORM2:
            return hidden_states

        hidden_states = r.nonlinearity(hidden_states)
        hidden_states = r.dropout(hidden_states)
        hidden_states = r.conv2(hidden_states)
        if r.conv_shortcut is not None:
            input_tensor = r.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / r.output_scale_factor


class DecoderPrefixThenUpBlock3Wrapper(torch.nn.Module):
    """Prefix through up_blocks[0..2], then part or all of up_blocks[3]."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        num_resnets: int,
        resnet0_stop: str = RESNET0_STOP_FULL,
    ):
        super().__init__()
        up3 = decoder.up_blocks[UP_BLOCK_NORM_BISECT_INDEX]
        if num_resnets < 0 or num_resnets > len(up3.resnets):
            raise ValueError(f"num_resnets must be in [0, {len(up3.resnets)}]")
        self.prefix = DecoderPrefixWrapper(
            decoder,
            num_up_blocks=UP_BLOCK_NORM_BISECT_INDEX,
            include_head=False,
        )
        self.up3 = up3
        self.num_resnets = num_resnets
        self.resnet0_stop = resnet0_stop

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.prefix(z)
        for i in range(self.num_resnets):
            resnet = self.up3.resnets[i]
            if i == RESNET_NORM_BISECT_INDEX and self.resnet0_stop != RESNET0_STOP_FULL:
                h = ResnetBlock2DPartialWrapper(resnet, self.resnet0_stop)(h)
            else:
                h = resnet(h, temb=None)
        return h


def build_d3_prefix_up3_then_norm1(decoder: torch.nn.Module) -> torch.nn.Module:
    """d3: prefix + norm1 only (passed — not sufficient to repro OOM)."""
    return DecoderPrefixThenNorm1Wrapper(decoder)


def build_d3b_prefix_up3_resnet0_stop(
    decoder: torch.nn.Module, stop: str
) -> torch.nn.Module:
    """Prefix + resnet[0] through ``stop`` (conv1, norm2, or full)."""
    return DecoderPrefixThenUpBlock3Wrapper(
        decoder, num_resnets=1, resnet0_stop=stop
    )


def build_d4_prefix_up3_resnet0_full(decoder: torch.nn.Module) -> torch.nn.Module:
    """Prefix + full up_blocks[3].resnets[0] (256→128 @ 1280×720)."""
    return DecoderPrefixThenUpBlock3Wrapper(
        decoder, num_resnets=1, resnet0_stop=RESNET0_STOP_FULL
    )


def build_d5_prefix_up3_resnets01(decoder: torch.nn.Module) -> torch.nn.Module:
    """Prefix + up_blocks[3].resnets[0] and resnets[1]."""
    return DecoderPrefixThenUpBlock3Wrapper(decoder, num_resnets=2)


def build_d6_prefix_up3_block_full(decoder: torch.nn.Module) -> torch.nn.Module:
    """Prefix + entire up_blocks[3] (3 resnets); should match prefix_through_up_4 OOM."""
    return DecoderPrefixThenUpBlock3Wrapper(decoder, num_resnets=3)


def d1_input_key() -> str:
    return "decoder_in"


def d2_input_key() -> str:
    return f"up_block_{UP_BLOCK_NORM_BISECT_INDEX - 1}"


def d3_input_key() -> str:
    return "decoder_in"


def build_norm2_only(decoder: torch.nn.Module) -> torch.nn.Module:
    """Isolated up_blocks[3].resnets[0].norm2 — GroupNorm(32, 128) @ 1280×720."""
    norm2 = decoder.up_blocks[UP_BLOCK_NORM_BISECT_INDEX].resnets[
        RESNET_NORM_BISECT_INDEX
    ].norm2
    return GroupNormOnlyWrapper(norm2)


def build_confirm_prefix_through_conv1(decoder: torch.nn.Module) -> torch.nn.Module:
    """Cumulative through conv1 (gate before norm2); must PASS."""
    return build_d3b_prefix_up3_resnet0_stop(decoder, RESNET0_STOP_CONV1)


def build_confirm_prefix_through_norm2(decoder: torch.nn.Module) -> torch.nn.Module:
    """Cumulative through norm2; must OOM (minimal repro for full decoder failure)."""
    return build_d3b_prefix_up3_resnet0_stop(decoder, RESNET0_STOP_NORM2)


def norm2_isolated_input_key() -> str:
    return "up3_resnet0_after_conv1"
