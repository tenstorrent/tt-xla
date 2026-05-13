# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 A14B — WanDiT (Transformer) component test.

Loads the high-noise expert (``transformer/``) and runs one forward pass,
comparing CPU vs TT output. Model config: 40 layers, 40 heads x 128 dim =
5120, ffn_dim 13824, in/out 16 channels (T2V-A14B; I2V variant uses
in_channels=36).

A14B uses ``expand_timesteps=False`` — timestep is a scalar per-batch
tensor, not per-token like the 5B TI2V model.

IN:  hidden_states (1, 16, latent_frames, latent_h, latent_w)
     timestep (1,)
     encoder_hidden_states (1, 512, 4096)
OUT: velocity (1, 16, latent_frames, latent_h, latent_w)
"""

import time
import copy

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd

from tests.infra.testers.compiler_config import CompilerConfig

from .monkey_patch import (
    _disable_tt_torch_function_override,
    _patch_adaln_modulation_bf16,
    _patch_apply_lora_scale,
    _patch_apply_rotary_emb_stack_form,
    _patch_fuse_qkv_projections,
    _patch_patchify_ndhwc_aware,
)
from tt_torch.torch_overrides import torch_function_override_disabled
from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    WanDiTWrapper,
    compute_pcc,
    load_dit,
    shard_dit_specs,
    wan22_mesh,
)

# Set to 0 to run the full model, otherwise set to the number of blocks to run.
MAX_BLOCKS = 1
N_RUNS = 5
PERF_MODE = True

# ---------------------------------------------------------------------------
# Monkey patches
# ---------------------------------------------------------------------------

_patch_apply_lora_scale()
if PERF_MODE:
    # Validated DiT perf patches. Measured on test_wan_dit_480p_sharded
    # (BH 4-chip, MAX_BLOCKS=1):
    #   baseline:                          1445 ms / PCC 0.99943
    #   + adaln modulation in bf16:        ~1230 ms / PCC ~0.99947 (~-15 %)
    # Conv3d patch_embedding perf is now handled by tt-mlir's (1,2,2)
    # Conv3dConfig heuristic — see TTIRToTTNN.cpp.
    _patch_adaln_modulation_bf16()
    # New reshape/permute reduction patches (May 2026), combined effect
    # 1255 ms → 1031 ms (-17.9 %) at MAX_BLOCKS=1, PCC 0.99946:
    #   + patchify NDHWC-aware (cancels post-conv3d permute pair, ~30ms)
    #   + rotary_emb half-rotation form (eliminates aten__index chain,
    #     ~190ms incremental)
    _patch_patchify_ndhwc_aware()
    _patch_apply_rotary_emb_stack_form()
#_disable_tt_torch_function_override()



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wan_dit_480p():
    _run(resolution="480p", sharded=False)


def test_wan_dit_720p():
    _run(resolution="720p", sharded=False)


def test_wan_dit_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_wan_dit_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(optimization_level=1, experimental_enable_dram_space_saving_optimization=True, enable_trace=True)
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]

    wrapper = WanDiTWrapper(load_dit(max_blocks=MAX_BLOCKS)).eval().bfloat16()

    hidden_states = torch.randn(1, LATENT_CHANNELS, t, h, w, dtype=torch.bfloat16)
    timestep = torch.full((1,), 500.0, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 512, 4096, dtype=torch.bfloat16)

    mesh = wan22_mesh() if sharded else None
    use_sharding = sharded and len(mesh.device_ids) > 1
    if use_sharding:
        enable_spmd()

    device = xm.xla_device()
    torch_xla.set_custom_compile_options(compiler_config.to_torch_compile_options())

    wrapper_cpu = copy.deepcopy(wrapper).to("cpu")
    wrapper_on_device = wrapper.to(device)
    inputs_on_device = [
        hidden_states.to(device),
        timestep.to(device),
        encoder_hidden_states.to(device),
    ]

    if use_sharding:
        for tensor, spec in shard_dit_specs(wrapper_on_device.dit).items():
            xs.mark_sharding(tensor, mesh, spec)

    with torch_function_override_disabled():
        compiled = torch.compile(wrapper_on_device, backend="tt")

        with torch.no_grad():
            warmup_start = time.perf_counter_ns()
            out = compiled(*inputs_on_device)
            cpu_out = out.to("cpu")
            warmup_end = time.perf_counter_ns()


            warm_times = []
            for _ in range(N_RUNS):
                warm_start = time.perf_counter_ns()
                tt_out = compiled(*inputs_on_device)
                tt_out = tt_out.to("cpu")
                warm_end = time.perf_counter_ns()
                warm_times.append(warm_end - warm_start)

    with torch.no_grad():
        cpu_out = wrapper_cpu(hidden_states, timestep, encoder_hidden_states)

    pcc = compute_pcc(tt_out, cpu_out)

    warmup_ms = (warmup_end - warmup_start) / 1e6
    warm_times_ms = [t / 1e6 for t in warm_times]

    print("====================================================================")
    print(f"| PERF: wan_dit {resolution} {'sharded' if sharded else 'single'}")
    print("--------------------------------------------------------------------")
    print(f"| cold (compile + run) e2e: {warmup_ms:.4f} ms")
    print(f"| warm times: {', '.join(f'{t:.4f} ms' for t in warm_times_ms)}")
    print(f"| PCC: {pcc}")
    print("====================================================================")
