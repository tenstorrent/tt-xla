# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill/decode loop test for DeepSeek-V4 Attention.

Drives a single Attention instance through one prefill call followed by N
sequential decode calls and asserts that the in-place KV / compressor /
indexer cache buffers on the TT device stay coherent with a CPU reference at
every step.

Bypasses run_graph_test because that runner compiles each call in isolation
and cannot maintain or compare in-place cache state across calls.
"""

import copy

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd

from .test_deepseek_v4_tp import (
    _attn_shard_spec,
    init_weights,
    make_attention,
    make_mesh,
    real_args,
    small_args,
)

REQUIRED_PCC = 0.99
ALLCLOSE_RTOL = 1e-2
ALLCLOSE_ATOL = 1e-2


def small_args_for_loop(**overrides):
    # Bump n_heads / index_n_heads to 8 so the (4, 8) mesh can shard on the
    # model axis exactly the same way as real_args does.
    base = dict(n_heads=8, index_n_heads=8)
    base.update(overrides)
    return small_args(**base)


# (args_factory, layer_id, decode_steps)
#   - layer_id picks a compress_ratio: small_args_for_loop -> {0,4,8}; real_args -> {0,4,128}
#   - prompt_len is fixed at args.window_size, so the very first decode step
#     wraps the attention's circular kv_cache.
#   - decode_steps = max(window_size, compress_ratio) so we hit at least one
#     compress event AND wrap the window.
PREFILL_DECODE_CASES = [
    pytest.param(small_args_for_loop, 0, 8, id="small-ratio0"),
    pytest.param(small_args_for_loop, 1, 8, id="small-ratio4"),
    pytest.param(small_args_for_loop, 2, 8, id="small-ratio8"),
    pytest.param(real_args, 0, 8, id="real-ratio0"),
    pytest.param(real_args, 2, 8, id="real-ratio4"),
    pytest.param(real_args, 3, 128, id="real-ratio128"),
]


def cache_buffers(attn) -> dict:
    """All in-place cache buffers held by an Attention module."""
    bufs = {"kv_cache": attn.kv_cache}
    if attn.compress_ratio:
        c = attn.compressor
        bufs["compressor.kv_cache"] = c.kv_cache
        bufs["compressor.kv_state"] = c.kv_state
        bufs["compressor.score_state"] = c.score_state
        if attn.indexer is not None:
            ic = attn.indexer.compressor
            bufs["indexer.compressor.kv_cache"] = ic.kv_cache
            bufs["indexer.compressor.kv_state"] = ic.kv_state
            bufs["indexer.compressor.score_state"] = ic.score_state
    return bufs


def _pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation. Mirrors tests/infra/evaluators/torch_comparison_evaluator.py:
    float64 conversion, allclose short-circuit to 1.0 (PCC is ill-conditioned when tensors
    are nearly identical), Pearson on flattened tensors.

    Handles tensors that contain -inf sentinel values (e.g. score_state, which uses -inf for
    not-yet-computed slots): first checks that the non-finite pattern agrees between x and y
    (a mismatch means one side has a real value where the other has a sentinel, which is a hard
    failure), then computes Pearson on the finite positions only."""
    x = x.detach().to(torch.float64).flatten()
    y = y.detach().to(torch.float64).flatten()
    # Structural agreement on non-finite positions: catches cases where TT writes a real value
    # where CPU expects -inf (or vice versa), which PCC alone cannot detect.
    if not (x.isfinite() == y.isfinite()).all():
        return 0.0
    finite = x.isfinite()
    if not finite.any():
        return 1.0  # both entirely non-finite and structurally identical
    x, y = x[finite], y[finite]
    if torch.allclose(x, y, rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL):
        return 1.0
    if x.numel() <= 1:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return float((vx @ vy) / denom)


def assert_pcc(
    tt_tensor: torch.Tensor,
    cpu_tensor: torch.Tensor,
    label: str,
    required: float = REQUIRED_PCC,
) -> None:
    tt_cpu = tt_tensor.detach().to("cpu")
    pcc = _pcc(tt_cpu, cpu_tensor)
    assert (
        pcc >= required
    ), f"PCC failed for {label}: got {pcc:.6f} (required >= {required})"


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize("args_factory,layer_id,decode_steps", PREFILL_DECODE_CASES)
def test_prefill_decode_cache_coherence(args_factory, layer_id, decode_steps):
    """Prefill + decode loop, asserting TT/CPU coherence on output and every cache buffer."""
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    args = args_factory()
    bsz = 4
    args.max_batch_size = bsz
    prompt_len = args.window_size

    # CPU reference and a deep-copy that will be moved to the TT device.
    cpu_attn = make_attention(args, layer_id)
    init_weights(cpu_attn)
    tt_attn = copy.deepcopy(cpu_attn).to(torch_xla.device())

    # Mark weight sharding once. Sharding persists across forward calls; only
    # the per-call input tensor needs to be re-marked because it's a fresh
    # tensor each step.
    mesh = make_mesh()
    weight_specs = _attn_shard_spec(tt_attn, args=[None], kwargs={})
    weight_specs.pop(None, None)  # drop the placeholder input entry
    for tensor, spec in weight_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    def run_step(start_pos: int, seq_len: int) -> None:
        x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)
        cpu_out = cpu_attn(x, start_pos)

        x_tt = x.to(torch_xla.device())
        xs.mark_sharding(x_tt, mesh, ("batch", None, None))
        tt_out = tt_attn(x_tt, start_pos)

        kind = "prefill" if start_pos == 0 else f"decode start_pos={start_pos}"
        assert_pcc(tt_out, cpu_out, label=f"{kind} output")
        cpu_bufs = cache_buffers(cpu_attn)
        tt_bufs = cache_buffers(tt_attn)
        for name, cpu_buf in cpu_bufs.items():
            assert_pcc(tt_bufs[name], cpu_buf, label=f"{kind} {name}")

    # Prefill (start_pos=0, seqlen=window_size so the first decode wraps).
    run_step(start_pos=0, seq_len=prompt_len)

    # Decode loop.
    for step in range(decode_steps):
        run_step(start_pos=prompt_len + step, seq_len=1)
