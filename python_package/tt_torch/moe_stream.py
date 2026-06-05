# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight preparation for the b1-stream low-batch MoE path.

Lays GPT-OSS expert weights out the way b1's ``DRAMStreamingExpertsMatmul``
kernel expects: per-expert ``[K, N]`` WIDTH_SHARDED across DRAM banks, with each
shard's tiles reordered row-major -> column-major so the K tiles of one output
column are contiguous (streamable). Experts are stacked along K so expert ``e``
lives at ``base + e * K`` rows; the kernel addresses it as
``base_addr + expert_idx * expert_size``.

GPT-OSS already stores experts stacked as a single parameter, so no gather is
needed:
  - ``gate_up_proj`` : [num_experts, hidden(=K), 2*intermediate(=N)]  -> e.g. [32, 2880, 5760]
  - ``down_proj``    : [num_experts, intermediate(=K), hidden(=N)]    -> e.g. [32, 2880, 2880]
Both are already ``[E, K, N]`` (K = contraction dim), used as ``x @ W[e]`` with
no transpose, so prep is just reshape + per-expert tile shuffle.

NOTE on the activation: GPT-OSS uses a clamped GLU (``alpha=1.702``,
``limit=7.0``) with gate/up **interleaved** in the 2*intermediate axis
(gate = ``[..., ::2]``, up = ``[..., 1::2]``). b1's kernel fuses plain SiLU, so
the clamped GLU must run as a separate step between the gate_up and down
streaming matmuls (not fused into the kernel). See ``gptoss_clamped_glu``.
"""

from __future__ import annotations

import torch

TILE_W = 32
# GPT-OSS clamped-GLU constants (transformers GptOssExperts).
GPTOSS_GLU_ALPHA = 1.702
GPTOSS_GLU_LIMIT = 7.0


def _pad_to_dram_banks(n: int, lcm: int) -> int:
    """Pad N up to a multiple of ``tile_w * num_banks`` (per-bank tile alignment)."""
    remainder = n % lcm
    return n if remainder == 0 else n + (lcm - remainder)


def shuffle_tensor_tiles(tensor: torch.Tensor, tile_size: int, num_banks: int) -> torch.Tensor:
    """Reorder tiles within each WIDTH_SHARDED bank from row-major to column-major.

    TTNN stores tiles row-major; the streaming kernel wants the ``K`` tiles of a
    given output column contiguous in physical memory so it can stream a K x 1
    stick per output tile. Mirrors b1's proven host helper.

    Args:
        tensor: ``[*, K, N]`` (batch dims allowed).
        tile_size: square tile dim (32).
        num_banks: number of DRAM banks / shards.
    Returns:
        ``[*, K, N]`` with per-shard tiles rearranged (same shape).
    """
    orig_shape = tensor.shape
    K, N = orig_shape[-2], orig_shape[-1]

    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    tensor = tensor.reshape(-1, K, N)
    batch = tensor.shape[0]
    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    # [batch, K, num_banks, per_N] -> [batch, num_banks, K, per_N] -> [batch*banks, K, per_N]
    tensor = tensor.reshape(batch, K, num_banks, per_N).permute(0, 2, 1, 3).contiguous()
    shards = tensor.reshape(-1, K, per_N)

    # to tiles: [shards, K_tiles, th, per_N_tiles, tw] -> [shards, K_tiles, per_N_tiles, th, tw]
    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    # shuffled position i <- source (i % K_tiles) * per_N_tiles + (i // K_tiles)
    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
    shuffled_tiles = tiles[:, source_idx, :, :]

    shuffled = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled = shuffled.permute(0, 1, 3, 2, 4).contiguous().reshape(-1, K, per_N)
    shuffled = shuffled.reshape(batch, num_banks, K, per_N).permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch, K, n_padded)
    if needs_padding:
        shuffled = shuffled[:, :, :N]
    return shuffled.reshape(*orig_shape)


def stacked_experts_to_b1_dram_layout(
    weight_ekn: torch.Tensor,
    num_banks: int,
    tile_w: int = TILE_W,
) -> torch.Tensor:
    """GPT-OSS stacked expert weights ``[E, K, N]`` -> b1 streaming DRAM layout.

    Since GPT-OSS already stacks experts, we only:
      1. per-expert column-major tile shuffle (so K tiles stream contiguously),
      2. stack experts along K into ``[E*K, N]`` (expert e at rows e*K .. (e+1)*K),
    which is exactly what the kernel reads via ``base + expert_idx * (K_tiles)``.

    The returned tensor is ready for a single WIDTH_SHARDED DRAM upload
    (shard shape ``[E*K, N_padded // num_banks]``). No transpose: GPT-OSS
    ``gate_up_proj``/``down_proj`` are already ``[E, K, N]`` with K the
    contraction dim.

    Args:
        weight_ekn: ``[E, K, N]`` (e.g. gate_up [32,2880,5760] or down [32,2880,2880]).
        num_banks: DRAM banks (12 on Wormhole, 8 on Blackhole).
        tile_w: tile width (32).
    Returns:
        ``[E*K, N_padded]`` shuffled, ready for WIDTH_SHARDED DRAM.
    """
    assert weight_ekn.dim() == 3, f"expected [E, K, N], got {tuple(weight_ekn.shape)}"
    E, K, N = weight_ekn.shape
    assert K % tile_w == 0, f"K({K}) must be tile-aligned"

    n_padded = _pad_to_dram_banks(N, tile_w * num_banks)
    # Shuffle every expert's [K, N] independently, then stack along K.
    # shuffle_tensor_tiles handles the N-padding internally and returns [E, K, N];
    # we pad N here so the stacked tensor has a uniform, bank-aligned width.
    if n_padded != N:
        weight_ekn = torch.nn.functional.pad(weight_ekn, (0, n_padded - N))
    shuffled = shuffle_tensor_tiles(weight_ekn, tile_w, num_banks)  # [E, K, n_padded]
    return shuffled.reshape(E * K, n_padded).contiguous()


def gptoss_clamped_glu(
    gate_up: torch.Tensor,
    alpha: float = GPTOSS_GLU_ALPHA,
    limit: float = GPTOSS_GLU_LIMIT,
) -> torch.Tensor:
    """GPT-OSS clamped GLU on a ``[..., 2*intermediate]`` gate_up tensor.

    gate = even cols, up = odd cols (interleaved). Reference (must match
    transformers ``GptOssExperts._apply_gate``); the device path runs this as a
    separate step between the gate_up and down streaming matmuls.
    """
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


def pad_gptoss_experts_for_tp_stream(model, num_model_devices: int = 8):
    """Zero-pad gpt-oss expert MLP weights so the TP (intermediate-sliced) shards
    tile-align, then return ``model``. Mutates the experts' params in place; call
    ONCE before sharding.

    A TTNN tile is 32 wide, so an N-way weight-slice needs the sliced dim divisible by
    ``32 * N`` (= 256 for 8 devices). gpt-oss ``intermediate=2880`` (90 tiles) is not,
    so we pad ``down_proj`` intermediate (and ``gate_up_proj``'s 2*intermediate, which
    stays interleaved since we append at the end) up to a multiple of ``32 *
    num_model_devices``. Zero padding contributes nothing: padded gate/up are 0 ->
    ``GLU(0) = (0+1)*0*sigmoid(0) = 0`` -> padded ``down`` rows are 0, so no masking.

    The pad runs on the HOST: the benchmark moves the model to the device before
    calling the shard-spec hook, so padding the live device tensor would lower to a
    device ``pad`` op (re-run every const-eval). Padding on CPU and copying back
    makes the padded weight a plain device parameter (persists across the
    prefill/decode executables), so no pad op is traced.
    """
    multiple = TILE_W * num_model_devices

    def _pad_host(t: torch.Tensor, pad) -> torch.nn.Parameter:
        out = torch.nn.functional.pad(t.detach().to("cpu"), pad).to(t.device)
        return torch.nn.Parameter(out, requires_grad=False)

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return model
    for layer in layers:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        two_inter = int(experts.gate_up_proj.shape[-1])
        inter = two_inter // 2
        inter_pad = ((inter + multiple - 1) // multiple) * multiple
        if inter_pad == inter:
            continue  # already tile-aligned for this device count
        d2 = 2 * inter_pad - two_inter  # pad on 2*intermediate (even -> gate/up preserved)
        di = inter_pad - inter  # pad on intermediate
        experts.gate_up_proj = _pad_host(experts.gate_up_proj.data, (0, d2))
        gub = getattr(experts, "gate_up_proj_bias", None)
        if gub is not None:
            experts.gate_up_proj_bias = _pad_host(gub.data, (0, d2))
        # down_proj [E, intermediate, hidden]: pad the intermediate (dim -2).
        experts.down_proj = _pad_host(experts.down_proj.data, (0, 0, 0, di))
    return model
