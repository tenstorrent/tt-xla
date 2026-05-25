# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MoE experts backend demo: AllenAI's OLMoE-1B-7B-0125-Instruct on Tenstorrent.

Runs a CPU reference forward with the stock `eager` backend, then runs the same
prompt on the card under `torch.compile(backend="tt")` with the auto-registered
`tt_moe` experts backend and reports the PCC between the two logits tensors.

When multiple TT devices are visible, the example installs a 2D `(1, N)` SPMD mesh
named `("batch", "model")`. Both tensor parallelism (attention projections)
and expert parallelism (routed experts only) run over the mesh's `model`
axis. `tt_experts_forward` picks up that global mesh and emits
`all_to_all_dispatch` / `all_to_all_combine` around the `sparse_matmul`
chain, while the rest of the decoder is annotated with explicit sharding so
the compiler sees a coherent TP+EP layout end to end.

Usage:
    # Requires at least two TT devices:
    python examples/pytorch/hf_interface_integration/olmoe.py

    # The script detects device count and installs the mesh automatically.
"""

import os
from typing import Any, cast

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
from tt_torch.moe_backend import REDUCTION_SIZE, TT_MOE_BACKEND_NAME

MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
PROMPT = "What is a mixture-of-experts model? Answer in one sentence."
SEQ_LEN = 64  # multiple of tt::sparse_matmul's REDUCTION_SIZE
BATCH_AXIS = "batch"
MODEL_AXIS = "model"
MODEL_AXIS_INDEX = 1


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient — the in-house accuracy metric."""
    a = a.detach().to(dtype=torch.float32, device="cpu").flatten()
    b = b.detach().to(dtype=torch.float32, device="cpu").flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _build_tp_ep_mesh(num_devices: int) -> Mesh:
    """Build a `(1, N)` mesh for TP+EP over the model axis."""
    if num_devices < 2:
        raise ValueError(
            f"This demo expects at least 2 TT devices for TP+EP, got {num_devices}."
        )
    device_ids = np.arange(num_devices)
    return Mesh(device_ids, (1, num_devices), (BATCH_AXIS, MODEL_AXIS))


def _mark_replicated(tensor: Any, mesh: Mesh) -> None:
    xs.mark_sharding(tensor, mesh, (None,) * tensor.dim())


def _mark_optional(tensor: Any, mesh: Mesh, spec: tuple) -> None:
    if tensor is not None:
        xs.mark_sharding(tensor, mesh, spec)


def _mark_routed_experts(experts: Any, mesh: Mesh) -> None:
    expert_spec = (MODEL_AXIS, None, None)
    if hasattr(experts, "gate_up_proj"):
        xs.mark_sharding(experts.gate_up_proj, mesh, expert_spec)
    else:
        xs.mark_sharding(experts.gate_proj, mesh, expert_spec)
        xs.mark_sharding(experts.up_proj, mesh, expert_spec)
    xs.mark_sharding(experts.down_proj, mesh, expert_spec)


def _shard_model(model: Any, mesh: Mesh) -> int:
    """Apply TP shardings to attention and EP shardings to routed experts."""
    _mark_replicated(model.model.embed_tokens.weight, mesh)
    _mark_replicated(model.model.norm.weight, mesh)
    _mark_replicated(model.lm_head.weight, mesh)

    moe_layers = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        xs.mark_sharding(attn.q_proj.weight, mesh, (MODEL_AXIS, None))
        _mark_optional(attn.q_proj.bias, mesh, (MODEL_AXIS,))
        xs.mark_sharding(attn.k_proj.weight, mesh, (MODEL_AXIS, None))
        _mark_optional(attn.k_proj.bias, mesh, (MODEL_AXIS,))
        xs.mark_sharding(attn.v_proj.weight, mesh, (MODEL_AXIS, None))
        _mark_optional(attn.v_proj.bias, mesh, (MODEL_AXIS,))
        xs.mark_sharding(attn.o_proj.weight, mesh, (None, MODEL_AXIS))
        _mark_optional(attn.o_proj.bias, mesh, (None,))
        _mark_replicated(attn.q_norm.weight, mesh)
        _mark_replicated(attn.k_norm.weight, mesh)

        _mark_replicated(layer.input_layernorm.weight, mesh)
        _mark_replicated(layer.post_attention_layernorm.weight, mesh)

        mlp = layer.mlp
        _mark_replicated(mlp.gate.weight, mesh)

        # EP shards only the routed-expert bank along the expert dimension.
        _mark_routed_experts(mlp.experts, mesh)
        moe_layers += 1

    return moe_layers


def main() -> None:
    assert (
        SEQ_LEN % REDUCTION_SIZE == 0
    ), f"SEQ_LEN must be a multiple of {REDUCTION_SIZE}"

    xr.set_device_type("TT")
    num_devices = xr.global_runtime_device_count()

    if num_devices > 1:
        # SPMD must be enabled before any XLA tensor is created so the runtime
        # treats parameters as globally-shaped. CONVERT_SHLO_TO_SHARDY=1 is
        # required for `xs.mark_sharding` annotations to reach the compiler.
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        mesh = _build_tp_ep_mesh(num_devices)
        xs.set_global_mesh(mesh)
        print(
            f"TP+EP enabled: mesh_shape={mesh.mesh_shape}, axes={mesh.axis_names}, "
            f"cluster_axis={MODEL_AXIS_INDEX}"
        )
    else:
        raise RuntimeError("tt_moe requires at least two TT devices for EP.")

    # MoE models are large; cast matmul weights to bfp_bf8 so the whole model
    # fits in device DRAM.
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
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
    card_model = cast(Any, card_model).to(device)

    if mesh is not None:
        moe_layers = _shard_model(card_model, mesh)
        print(f"Applied TP+EP shardings across {moe_layers} MoE layers")

    compiled = torch.compile(card_model, backend="tt")
    tt_input_ids = input_ids.to(device)
    tt_attention_mask = attention_mask.to(device)
    if mesh is not None:
        xs.mark_sharding(tt_input_ids, mesh, (None, None))
        xs.mark_sharding(tt_attention_mask, mesh, (None, None))
    with torch.no_grad():
        logits_card = compiled(
            input_ids=tt_input_ids,
            attention_mask=tt_attention_mask,
            use_cache=False,
        ).logits.to("cpu")

    print(
        f"PCC cpu(eager) vs card({TT_MOE_BACKEND_NAME}) = {pcc(logits_cpu, logits_card):.6f}"
    )


if __name__ == "__main__":
    main()
