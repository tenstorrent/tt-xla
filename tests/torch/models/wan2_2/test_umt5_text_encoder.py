# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — UMT5-XXL Text Encoder component test.

IN:  input_ids (1, 512) int64, attention_mask (1, 512) int64
OUT: last_hidden_state (1, 512, 4096) float
"""

import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import (
    RESOLUTIONS,
    UMT5Wrapper,
    compute_pcc,
    load_umt5,
    shard_umt5_specs,
    wan22_mesh,
)


def test_umt5_480p():  # OOM on single device
    _run(resolution="480p", sharded=False)


def test_umt5_720p():  # OOM on single device
    _run(resolution="720p", sharded=False)


def test_umt5_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_umt5_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)
    _ = RESOLUTIONS[resolution]  # resolution is a no-op for UMT5 shapes

    wrapper = UMT5Wrapper(load_umt5()).eval().bfloat16()

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 512), dtype=torch.long)
    attention_mask = torch.ones(1, 512, dtype=torch.long)

    mesh = wan22_mesh() if sharded else None
    use_sharding = sharded and len(mesh.device_ids) > 1
    if use_sharding:
        enable_spmd()

    device = xm.xla_device()
    torch_xla.set_custom_compile_options(compiler_config.to_torch_compile_options())

    wrapper_on_device = wrapper.to(device)
    if hasattr(wrapper_on_device, "tie_weights"):
        wrapper_on_device.tie_weights()
    inputs_on_device = [input_ids.to(device), attention_mask.to(device)]

    if use_sharding:
        for tensor, spec in shard_umt5_specs(wrapper_on_device.encoder).items():
            xs.mark_sharding(tensor, mesh, spec)

    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        warmup_start = time.perf_counter_ns()
        _ = compiled(*inputs_on_device)
        torch_xla.sync(wait=True)
        warmup_end = time.perf_counter_ns()

        warm_start = time.perf_counter_ns()
        tt_out = compiled(*inputs_on_device)
        torch_xla.sync(wait=True)
        warm_end = time.perf_counter_ns()

    tt_out_cpu = tt_out.to("cpu")

    wrapper_cpu = wrapper_on_device.to("cpu")
    if hasattr(wrapper_cpu, "tie_weights"):
        wrapper_cpu.tie_weights()
    with torch.no_grad():
        cpu_out = wrapper_cpu(input_ids, attention_mask)

    pcc = compute_pcc(tt_out_cpu, cpu_out)

    warmup_ms = (warmup_end - warmup_start) / 1e6
    warm_ms = (warm_end - warm_start) / 1e6

    print("====================================================================")
    print(f"| PERF: umt5 {resolution} {'sharded' if sharded else 'single'}")
    print("--------------------------------------------------------------------")
    print(f"| warmup (compile + run) e2e: {warmup_ms:.4f} ms")
    print(f"| warm                   e2e: {warm_ms:.4f} ms")
    print(f"| PCC: {pcc}")
    print("====================================================================")
