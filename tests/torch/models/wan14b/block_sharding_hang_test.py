# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for the sharded VAE decoder hang.

Loads only ``vae.decoder.mid_block.resnets[0]`` (the first
``WanResidualBlock`` the full decoder hits, where the runtime hang
reproduces) and runs one forward pass through it under
``torch.compile``. Same Megatron column→row sharding pattern as the full
decoder test.

Compared to ``test_vae_decoder.py::test_vae_decoder_480p_sharded`` this:

  - drops streaming decode (no per-frame loop)
  - drops post_quant_conv / decoder.conv_in / norm_out / conv_out
  - drops all the other 13 ``WanResidualBlock``s
  - drops upsamplers / mid_block attention

What's preserved: ``torch.compile`` with the ``tt`` backend, the
Megatron 4-tensor shard (conv1 column-parallel, norm2 gamma matched,
conv2 row-parallel), and a 480p-sized 5D input
``(1, 384, 1, 60, 104)`` that matches the shape mid_block.resnets[0]
sees in the real decoder.

Expected if the same bug reproduces: hang at the second ``ttnn.all_gather``
(``all_gather_dim=4`` W, input ``1×384×1×60×32`` → output
``1×384×1×60×128``) after ~2 CCL pairs and the leading conv ops.

Run unsharded baseline:
    pytest tests/torch/models/wan14b/block_sharding_test.py::test_block_unsharded -v -s

Run sharded (the repro):
    pytest tests/torch/models/wan14b/block_sharding_test.py::test_block_sharded -v -s
"""

import time

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import _megatron_pair_specs, _pick_axis, compute_pcc, load_vae, wan22_mesh

# Shape mid_block.resnets[0] sees in the full decoder for 480p, first chunk:
#   B=1, C=384 (mid_block dim = base_dim * dim_mult[-1] = 96 * 4),
#   T=1 (per-frame streaming),
#   H=60, W=104 (480p latent before any upsample).
INPUT_SHAPE = (1, 384, 1, 60, 104)


class SingleBlockWrapper(nn.Module):
    """Forward a single ``WanResidualBlock`` with no ``feat_cache``.

    With ``feat_cache=None`` the block's two ``WanCausalConv3d``s use their
    default causal F.pad path (no cross-call state), which keeps the trace
    self-contained.
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def test_block_unsharded() -> None:
    _run(sharded=False)


def test_block_sharded() -> None:
    _run(sharded=True)


def _run(sharded: bool) -> None:
    xr.set_device_type("TT")
    # Match the minimal compiler options the WAN 2.1 reference uses for VAE
    # (`optimization_level=1` only). The full decoder test has extra flags
    # (`enable_trace`, dram-saving, export) which are unrelated to sharding
    # and just noise in the minimal repro.
    compiler_config = CompilerConfig(optimization_level=1)
    torch.manual_seed(42)

    vae = load_vae()
    block = vae.decoder.mid_block.resnets[0]
    wrapper = SingleBlockWrapper(block).eval().bfloat16()

    x = torch.randn(*INPUT_SHAPE, dtype=torch.bfloat16)

    mesh = wan22_mesh() if sharded else None
    use_sharding = sharded and len(mesh.device_ids) > 1
    if use_sharding:
        enable_spmd()

    device = xm.xla_device()
    torch_xla.set_custom_compile_options(compiler_config.to_torch_compile_options())

    wrapper_on_device = wrapper.to(device)
    x_on_device = x.to(device)

    if use_sharding:
        axis = _pick_axis(mesh)
        # ``_megatron_pair_specs`` returns exactly 4 tensor → spec entries:
        # conv1.weight (col), conv1.bias (col), norm2.gamma (col), conv2.weight (row).
        for tensor, spec in _megatron_pair_specs(wrapper_on_device.block, axis).items():
            xs.mark_sharding(tensor, mesh, spec)

    print(
        f"[block_sharding_test] sharded={sharded} "
        f"input_shape={INPUT_SHAPE} starting torch.compile...",
        flush=True,
    )
    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        cold_start = time.perf_counter_ns()
        out = compiled(x_on_device)
        out_cpu = out.to("cpu")
        cold_end = time.perf_counter_ns()

    # CPU reference for PCC.
    wrapper_cpu = wrapper_on_device.to("cpu")
    with torch.no_grad():
        ref = wrapper_cpu(x)
    pcc = compute_pcc(out_cpu, ref)

    cold_ms = (cold_end - cold_start) / 1e6
    print("=" * 70)
    print(f"| PERF: block_sharding {'sharded' if sharded else 'single'}")
    print("-" * 70)
    print(f"| cold (compile + run) e2e: {cold_ms:.4f} ms")
    print(f"| out shape: {tuple(out_cpu.shape)}")
    print(f"| PCC vs CPU: {pcc}")
    print("=" * 70)
