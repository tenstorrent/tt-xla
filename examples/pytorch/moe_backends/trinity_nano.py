# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MoE experts backend demo: Arcee's Trinity-Nano-Preview on Tenstorrent hardware.

Registers the `tt_moe` experts backend from `tt_torch.moe_backend`, runs a
CPU reference forward with the stock `eager` backend, then runs the same
prompt on the card under `torch.compile(backend="tt")` with `tt_moe` and
reports the PCC between the two logits tensors.

When more than one TT device is visible, the example installs a 1D SPMD
mesh and shards each `AfmoeExperts` module's `gate_up_proj` / `down_proj`
across the expert axis. `tt_experts_forward` then picks up the global mesh
and emits `all_to_all_dispatch` / `all_to_all_combine` around the
`sparse_matmul` chain, giving a real end-to-end expert-parallel execution
without any model-specific module surgery.

Usage:
    # Single-device (data-parallel expert compute):
    python examples/pytorch/moe_backends/trinity_nano.py

    # Multi-device EP: nothing to change — the script detects device count
    # and installs the mesh automatically.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import tt_torch  # registers the "tt" torch.compile backend and torch.ops.tt.*
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch.attention_backend import (
    TT_ATTENTION_BACKEND_NAME,
    register_tt_attention_backend,
)
from tt_torch.moe_backend import (
    REDUCTION_SIZE,
    TT_MOE_BACKEND_NAME,
    register_tt_moe_backend,
)

MODEL_ID = "arcee-ai/Trinity-Nano-Preview"
PROMPT = "Explain in one sentence what a mixture-of-experts model is."
SEQ_LEN = 64  # multiple of tt::sparse_matmul's REDUCTION_SIZE
EXPERT_AXIS = "experts"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient — the in-house accuracy metric."""
    a = a.detach().to(dtype=torch.float32, device="cpu").flatten()
    b = b.detach().to(dtype=torch.float32, device="cpu").flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _build_ep_mesh(num_devices: int) -> Mesh:
    """1D mesh whose only axis is the expert-parallel axis (cluster_axis=0)."""
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, (num_devices,), (EXPERT_AXIS,))


def _shard_experts(model: torch.nn.Module, mesh: Mesh) -> int:
    """Shard every AfmoeExperts module's `gate_up_proj` / `down_proj` on the
    expert dim. Returns the number of MoE layers sharded.
    """
    count = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        # gate_up_proj: [E, 2*I, H] — shard E on the expert axis.
        xs.mark_sharding(experts.gate_up_proj, mesh, (EXPERT_AXIS, None, None))
        # down_proj:    [E, H,   I] — same treatment.
        xs.mark_sharding(experts.down_proj, mesh, (EXPERT_AXIS, None, None))
        count += 1
    return count


def main() -> None:
    assert (
        SEQ_LEN % REDUCTION_SIZE == 0
    ), f"SEQ_LEN must be a multiple of {REDUCTION_SIZE}"

    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()

    if num_devices > 1:
        # SPMD must be enabled before any XLA tensor is created so the runtime
        # treats parameters as globally-shapeded. CONVERT_SHLO_TO_SHARDY=1 is
        # required for `xs.mark_sharding` annotations to reach the compiler.
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        mesh = _build_ep_mesh(num_devices)
        xs.set_global_mesh(mesh)
        print(f"EP enabled: {num_devices} devices, axis={EXPERT_AXIS!r}")
    else:
        mesh = None
        print("Single-device run (no EP)")

    # MoE models are large; cast matmul weights to bfp_bf8 so the whole model
    # fits in device DRAM.
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    register_tt_moe_backend(cluster_axis=0)
    register_tt_attention_backend()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

    # --- CPU reference (stock "eager" backend). ---
    print("Running CPU reference...", flush=True)
    cpu_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, experts_implementation="eager"
    ).eval()
    with torch.no_grad():
        logits_cpu = cpu_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits.clone()
    del cpu_model

    # --- Same model on card with the tt_moe experts + tt_sdpa attention. ---
    print(
        f"Running on TT card with tt_moe + tt_sdpa backends ({num_devices} device"
        f"{'s' if num_devices != 1 else ''})...",
        flush=True,
    )
    card_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        experts_implementation=TT_MOE_BACKEND_NAME,
        attn_implementation=TT_ATTENTION_BACKEND_NAME,
    ).eval()
    device = torch_xla.device()
    card_model = card_model.to(device)

    if mesh is not None:
        moe_layers = _shard_experts(card_model, mesh)
        print(f"Sharded experts on {moe_layers} MoE layers")

    compiled = torch.compile(card_model, backend="tt")
    with torch.no_grad():
        logits_card = compiled(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).logits.to("cpu")

    print(
        f"PCC cpu(eager) vs card({TT_MOE_BACKEND_NAME}) = {pcc(logits_cpu, logits_card):.6f}"
    )


if __name__ == "__main__":
    main()
