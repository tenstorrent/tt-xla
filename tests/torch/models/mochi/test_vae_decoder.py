# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi VAE decoder component test at original Mochi-1 resolution (0.36B)."""

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from diffusers.models.autoencoders.autoencoder_kl_mochi import MochiChunkedGroupNorm3D
from infra import Framework, run_graph_test
from infra.testers.compiler_config import CompilerConfig
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.mochi.pytorch.src.utils import (
    load_vae_decoder_inputs_full_res,
)


def _replicate_pad_in_native_dtype(self, inputs, conv_cache=None):
    """CogVideoXCausalConv3d replicate padding without the f32 round trip.

    torch has no bf16 replication_pad3d kernel, so the stock
    F.pad(..., mode="replicate") upcasts. XLA keeps that upcast in the graph,
    and the permute plus three pad concats that follow all run at f32 - each
    one twice the DRAM it needs, on the largest tensors in the decoder.

    Replication is clamp(index) per axis and therefore separable, so padding
    one axis at a time with slice + expand + cat gives identical values while
    staying in the input's own dtype. Order across axes doesn't matter.
    """
    if self.pad_mode != "replicate":
        return _STOCK_FAKE_CONTEXT_PARALLEL_FORWARD(self, inputs, conv_cache)

    x = inputs
    if self.time_pad:
        # Causal: leading pad only, replicating frame 0.
        x = torch.cat([x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1), x], dim=2)
    if self.height_pad:
        pad = self.height_pad
        x = torch.cat(
            [
                x[:, :, :, :1].expand(-1, -1, -1, pad, -1),
                x,
                x[:, :, :, -1:].expand(-1, -1, -1, pad, -1),
            ],
            dim=3,
        )
    if self.width_pad:
        pad = self.width_pad
        x = torch.cat(
            [
                x[..., :1].expand(-1, -1, -1, -1, pad),
                x,
                x[..., -1:].expand(-1, -1, -1, -1, pad),
            ],
            dim=4,
        )
    return x


_STOCK_FAKE_CONTEXT_PARALLEL_FORWARD = (
    CogVideoXCausalConv3d.fake_context_parallel_forward
)


def _group_norm_with_affine_on_channels(self, x: torch.Tensor = None) -> torch.Tensor:
    """MochiChunkedGroupNorm3D with the affine applied outside F.group_norm.

    F.group_norm normalizes in group space, [chunks*groups, C/groups*H*W], where
    a per-channel weight varies along the *inner* axis and so cannot broadcast
    implicitly. Since up_block.proj was sharded the compiler keeps the affine in
    that space, and each norm materializes its weight and bias as full
    1x8x128x480x848 f32 tensors - 1.70 GB apiece, via ttnn.repeat.

    Applying the affine ourselves on the [chunk, C, H, W] output puts channel
    back on a real dimension, where the broadcast is implicit and free. Same
    arithmetic: group_norm scales by weight[c] and shifts by bias[c] after
    normalizing, which is exactly what this does.

    The normalize is also open-coded so it can run in the input's dtype.
    F.group_norm upcasts everything to f32, which costs 1.67 GB per
    1x256x1628160 intermediate. Only the reductions actually need f32 - their
    outputs are tiny - so mean and variance are computed in f32 and the
    full-size centre-and-scale runs in bf16. That is a real precision
    reduction: the centred values are O(1) so bf16 carries them fine, but the
    subtraction rounds the mean to bf16 first, which costs accuracy in
    proportion to mean/std.
    """
    batch_size = x.size(0)
    norm = self.norm_layer

    x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

    normalized_chunks = []
    for chunk in x.split(self.chunk_size, dim=0):
        grouped = chunk.reshape(chunk.shape[0], norm.num_groups, -1)

        stats = grouped.float()
        mean = stats.mean(dim=2, keepdim=True)
        # Biased variance, matching F.group_norm.
        var = stats.var(dim=2, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + norm.eps)

        centred = grouped - mean.to(grouped.dtype)
        normalized_chunks.append((centred * rstd.to(grouped.dtype)).reshape_as(chunk))

    output = torch.cat(normalized_chunks, dim=0)
    if norm.affine:
        output = output * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)

    return output.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)


def test_torch_mochi_vae_decoder_inference():
    # run_graph_test runs the decoder on CPU as a golden reference before
    # comparing against TT, and the CPU pass takes > 50 min at full
    # resolution - https://github.com/tenstorrent/tt-xla/issues/4885
    xr.set_device_type("TT")
    torch.manual_seed(42)

    CogVideoXCausalConv3d.fake_context_parallel_forward = _replicate_pad_in_native_dtype
    MochiChunkedGroupNorm3D.forward = _group_norm_with_affine_on_channels

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="vae")
    vae = loader.load_model(dtype_override=torch.bfloat16)
    decoder = vae.decoder.eval()

    # Megatron-style sharding; run_graph_test enables SPMD and applies the
    # shard specs once the weights are on the XLA device.
    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    [latent] = load_vae_decoder_inputs_full_res(dtype=torch.bfloat16)

    # Const-eval is off because the chunked GroupNorm's affine params get
    # hoisted into full-activation-size f32 broadcasts (a 128-element bias
    # becomes a 1.67 GB 1x256x1628160 constant), and const-eval'd results stay
    # pinned in DRAM for the whole execution - 14 GiB of them, against 12.8 GB
    # of DRAM.
    compiler_config = CompilerConfig(
        optimization_level=1,
        experimental_enable_dram_space_saving_optimization=True,
        enable_const_eval=False,
    )

    run_graph_test(
        decoder,
        [latent],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
        compiler_config=compiler_config,
    )
