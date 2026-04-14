# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Isolated GPT-OSS 20B MoE MLP block on Galaxy (32-chip), aligned with ``test_gpt_oss_20b_tp_galaxy_batch_size_64``.

Captures post-attention RMSNorm outputs (MoE inputs) using the same tokenizer prompt and
``construct_inputs`` path as ``benchmarks.llm_benchmark``.  Prefill uses full context length;
decode uses the first post-prefill token (argmax), matching the benchmark decode step shape
``[batch, 1, hidden]``.

CPU golden runs the same ``layer.mlp`` weights in bf16.  TT runs a compiled MoE-only module with
Galaxy mesh, MoE weight shard specs, and batch-sharded activations ``("batch", None, None)``.

**Optional: MoE inputs from TTNN op-trace dumps** (same layout as ``test_topk_to_scatter_pipeline``):

* ``GPT_OSS_MOE_INPUT_DUMP_DIR`` — directory with ``ttnn_linear_<seq>_in0_dev0.npy`` from
  ``TT_RUNTIME_OP_TENSOR_TRACE_TOPK_DUMP_DIR`` captures (e.g.
  ``modules/gpt_oss_input_sharding_dbg/topk_dump`` under ``tests/benchmark``).
* ``GPT_OSS_MOE_ROUTER_LINEAR_OP_SEQ`` — op_seq for the router ``ttnn.linear`` before ``ttnn.topk``
  (default ``142`` for dumps that match ``ttnn_topk_144``).
* ``GPT_OSS_MOE_DUMP_PREFILL_LAYOUT`` — optional ``batch,seq`` (e.g. ``16,17`` for ``N=272`` tokens);
  default is ``1,N`` (one batch dimension, sequence ``N``).
* ``GPT_OSS_MOE_DECODE_HIDDEN_NPY`` — optional path to a ``.npy`` decode hidden tensor
  ``[batch, hidden]`` or ``[batch, 1, hidden]``.  If unset while ``GPT_OSS_MOE_INPUT_DUMP_DIR``
  is set, the decode PCC test is skipped (prefill-only dumps typically have no separate decode slice).
"""

from __future__ import annotations

import copy
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.llm_benchmark import (
    DEFAULT_INPUT_PROMPT,
    MODULE_EXPORT_PATH,
    check_transformers_version,
    construct_inputs,
    get_mesh,
    setup_model_and_tokenizer,
)
from utils import build_xla_export_name, compute_pcc_flat_pair, create_model_loader

xr.set_device_type("TT")

_BENCH_ROOT = Path(__file__).resolve().parent

# Match ``test_gpt_oss_20b_tp_galaxy_batch_size_64`` / ``test_llm`` defaults.
DEFAULT_BATCH_SIZE = 64
DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE = "bfp_bf8"
DEFAULT_REQUIRED_PCC = 0.95

# Decoder layer whose MoE block is exercised (post-attention norm → mlp).
MOE_LAYER_IDX = 0


def _gpt_oss_galaxy_mesh_config_fn(model_loader, num_devices: int):
    if num_devices != 32:
        raise ValueError(
            "GPT-OSS wormhole_galaxy MoE block test expects 32 devices (4x8 mesh)."
        )
    return (4, 8), ("batch", "model")


def _galaxy_moe_shard_specs(mlp: torch.nn.Module) -> Dict[torch.Tensor, tuple]:
    """MoE parameter shard specs for Galaxy — same as ``_gpt_oss_galaxy_shard_spec_fn`` MoE lines."""

    batch_axis = None
    specs: Dict[torch.Tensor, tuple] = {}
    specs[mlp.router.weight] = (None, batch_axis)
    specs[mlp.experts.gate_up_proj] = ("model", None, None)
    specs[mlp.experts.gate_up_proj_bias] = ("model", None)
    specs[mlp.experts.down_proj] = ("model", None, None)
    specs[mlp.experts.down_proj_bias] = ("model", None)
    return specs


def _post_attn_norm(module: torch.nn.Module) -> torch.nn.Module:
    norm = getattr(module, "post_attention_layernorm", None)
    if norm is None:
        raise AssertionError(
            "Expected ``post_attention_layernorm`` on decoder layer (GPT-OSS / Llama-style)."
        )
    return norm


def _mlp_forward_hidden(mlp: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    out = mlp(hidden_states)
    return out[0] if isinstance(out, tuple) else out


class _MoeSubnet(torch.nn.Module):
    """Torch-compilable wrapper (stable output tensor)."""

    def __init__(self, mlp: torch.nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _mlp_forward_hidden(self.mlp, hidden_states)


def _min_pcc_across_batch_rows(golden: torch.Tensor, actual: torch.Tensor) -> float:
    if golden.shape != actual.shape:
        raise AssertionError(
            f"Shape mismatch for PCC: golden {tuple(golden.shape)} vs actual {tuple(actual.shape)}"
        )
    if golden.dim() < 2:
        return compute_pcc_flat_pair(golden, actual)
    pccs: list[float] = []
    for b in range(golden.shape[0]):
        pccs.append(compute_pcc_flat_pair(golden[b], actual[b]))
    return min(pccs)


@dataclass(frozen=True)
class _MoeCpuReference:
    h_prefill: torch.Tensor
    gold_prefill: torch.Tensor
    h_decode: Optional[torch.Tensor]
    gold_decode: Optional[torch.Tensor]
    mlp_template: torch.nn.Module
    batch_size: int
    prefill_seq_len: int
    num_layers: Optional[int]


def _resolve_under_benchmark(path_str: str) -> Path:
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (_BENCH_ROOT / p).resolve()


def _reshape_dump_tokens(flat: torch.Tensor, layout: str) -> torch.Tensor:
    """``flat`` is ``[N, H]``; ``layout`` is ``batch,seq`` with ``batch*seq==N``."""

    n, _h = flat.shape
    parts = [x.strip() for x in layout.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected batch,seq layout, got {layout!r}")
    b, s = int(parts[0]), int(parts[1])
    if b * s != n:
        raise ValueError(f"layout {b}x{s} does not match N={n} token rows")
    return flat.reshape(b, s, -1)


def _load_router_hidden_npy(dump_dir: Path, op_seq: int) -> torch.Tensor:
    """Router ``ttnn.linear`` activation input ``[N, H]`` (``in0``), dev0 replica."""

    npy = dump_dir / f"ttnn_linear_{op_seq}_in0_dev0.npy"
    if not npy.is_file():
        pytest.fail(
            f"Missing dump file {npy} (op_seq={op_seq}). "
            "Set GPT_OSS_MOE_ROUTER_LINEAR_OP_SEQ or fix GPT_OSS_MOE_INPUT_DUMP_DIR."
        )
    arr = np.load(npy)
    return torch.from_numpy(arr).to(torch.bfloat16)


def _load_prefill_hidden_from_dump() -> Optional[torch.Tensor]:
    raw = os.environ.get("GPT_OSS_MOE_INPUT_DUMP_DIR", "").strip()
    if not raw:
        return None
    dump_dir = _resolve_under_benchmark(raw)
    if not dump_dir.is_dir():
        pytest.fail(f"GPT_OSS_MOE_INPUT_DUMP_DIR is not a directory: {dump_dir}")
    seq = int(os.environ.get("GPT_OSS_MOE_ROUTER_LINEAR_OP_SEQ", "142"))
    flat = _load_router_hidden_npy(dump_dir, seq)
    layout = os.environ.get("GPT_OSS_MOE_DUMP_PREFILL_LAYOUT", "").strip()
    if layout:
        return _reshape_dump_tokens(flat, layout)
    n, _h = flat.shape
    return flat.reshape(1, n, _h)


def _load_decode_hidden_from_dump() -> Optional[torch.Tensor]:
    raw = os.environ.get("GPT_OSS_MOE_DECODE_HIDDEN_NPY", "").strip()
    if not raw:
        return None
    path = _resolve_under_benchmark(raw)
    if not path.is_file():
        pytest.fail(f"GPT_OSS_MOE_DECODE_HIDDEN_NPY not found: {path}")
    arr = np.load(path)
    t = torch.from_numpy(arr).to(torch.bfloat16)
    if t.dim() == 2:
        return t.unsqueeze(1)
    if t.dim() == 3:
        return t
    pytest.fail(f"Decode hidden must be 2D or 3D, got shape {tuple(t.shape)}")


def _build_cpu_reference(
    *,
    batch_size: int,
    max_cache_len: int,
    num_layers: Optional[int],
) -> _MoeCpuReference:
    from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

    check_transformers_version()
    model_loader = create_model_loader(ModelLoader, num_layers=num_layers, variant=ModelVariant.GPT_OSS_20B)
    if num_layers is not None and model_loader is None:
        pytest.fail("num_layers override requested but ModelLoader does not support it.")

    model, tokenizer = setup_model_and_tokenizer(model_loader, ModelVariant.GPT_OSS_20B)
    layer = model.model.layers[MOE_LAYER_IDX]
    norm = _post_attn_norm(layer)

    h_prefill_dump = _load_prefill_hidden_from_dump()
    h_decode_dump = _load_decode_hidden_from_dump()

    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        input_prompt=DEFAULT_INPUT_PROMPT,
    )

    captured: list[torch.Tensor] = []

    def _hook(_m, _i, output):
        captured.append(output.detach().clone())

    if h_prefill_dump is not None:
        h_prefill = h_prefill_dump.clone()
        with torch.no_grad():
            gold_prefill = _mlp_forward_hidden(layer.mlp, h_prefill.clone())
        prefill_seq_len = int(input_args["input_ids"].shape[1])
    else:
        handle = norm.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                prefill_out = model(
                    input_ids=input_args["input_ids"],
                    past_key_values=input_args["past_key_values"],
                    cache_position=input_args["cache_position"],
                    use_cache=True,
                )
        finally:
            handle.remove()

        if len(captured) != 1:
            pytest.fail(f"Expected one post-attn norm capture, got {len(captured)}")
        h_prefill = captured[0]
        with torch.no_grad():
            gold_prefill = _mlp_forward_hidden(layer.mlp, h_prefill.clone())

        prefill_seq_len = int(input_args["input_ids"].shape[1])
        first_token_ids = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_args["input_ids"] = first_token_ids
        input_args["cache_position"] = torch.tensor([prefill_seq_len])

    h_decode: Optional[torch.Tensor] = None
    gold_decode: Optional[torch.Tensor] = None

    if h_decode_dump is not None:
        h_decode = h_decode_dump.clone()
        with torch.no_grad():
            gold_decode = _mlp_forward_hidden(layer.mlp, h_decode.clone())
    elif h_prefill_dump is None:
        captured.clear()
        handle2 = norm.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(
                    input_ids=input_args["input_ids"],
                    past_key_values=input_args["past_key_values"],
                    cache_position=input_args["cache_position"],
                    use_cache=True,
                )
        finally:
            handle2.remove()

        if len(captured) != 1:
            pytest.fail(f"Expected one decode post-attn norm capture, got {len(captured)}")
        h_decode = captured[0]
        with torch.no_grad():
            gold_decode = _mlp_forward_hidden(layer.mlp, h_decode.clone())

    # CPU golden uses plain bf16 weights (same as benchmark CPU prefill / PCC reference).
    # Per-tensor ``apply_weight_dtype_overrides`` targets full-model param paths; the TT path
    # uses ``experimental_weight_dtype`` in compile options like ``test_llm``.
    mlp_template = copy.deepcopy(layer.mlp).cpu().eval()

    del model
    gc.collect()

    return _MoeCpuReference(
        h_prefill=h_prefill.cpu(),
        gold_prefill=gold_prefill.cpu(),
        h_decode=h_decode.cpu() if h_decode is not None else None,
        gold_decode=gold_decode.cpu() if gold_decode is not None else None,
        mlp_template=mlp_template,
        batch_size=batch_size,
        prefill_seq_len=prefill_seq_len,
        num_layers=num_layers,
    )


@pytest.fixture(scope="module")
def gpt_oss_20b_galaxy_moe_cpu_reference(request) -> _MoeCpuReference:
    """One CPU load: realistic MoE inputs/goldens for prefill and first decode step."""

    num_layers = request.config.getoption("--num-layers")
    batch_size_opt = request.config.getoption("--batch-size")
    batch_size = batch_size_opt if batch_size_opt is not None else DEFAULT_BATCH_SIZE
    return _build_cpu_reference(
        batch_size=batch_size,
        max_cache_len=DEFAULT_INPUT_SEQUENCE_LENGTH,
        num_layers=num_layers,
    )


def _require_galaxy_devices():
    if xr.global_runtime_device_count() != 32:
        pytest.skip(
            "Requires 32 TT devices (wormhole_galaxy), same as "
            "test_gpt_oss_20b_tp_galaxy_batch_size_64."
        )


def _run_tt_moe_pcc(
    ref: _MoeCpuReference,
    hidden_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    *,
    case: str,
    required_pcc: float,
) -> None:
    _require_galaxy_devices()
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

    model_loader = create_model_loader(
        ModelLoader, num_layers=ref.num_layers, variant=ModelVariant.GPT_OSS_20B
    )
    mesh = get_mesh(model_loader, _gpt_oss_galaxy_mesh_config_fn)
    device = torch_xla.device()

    mlp = copy.deepcopy(ref.mlp_template).to(device=device, dtype=torch.bfloat16)
    mlp.eval()
    wrapper = _MoeSubnet(mlp).eval()

    for tensor, spec in _galaxy_moe_shard_specs(mlp).items():
        xs.mark_sharding(tensor, mesh, spec)

    hidden = hidden_cpu.to(device=device, dtype=torch.bfloat16)
    xs.mark_sharding(hidden, mesh, ("batch", None, None))

    export_model_name = build_xla_export_name(
        model_name=f"gpt_oss_moe_block_{case}",
        num_layers=ref.num_layers,
        batch_size=ref.batch_size,
        input_sequence_length=hidden.shape[1],
    )
    options = {
        "optimization_level": DEFAULT_OPTIMIZATION_LEVEL,
        "enable_trace": False,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": False,
        "experimental_weight_dtype": DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE,
        "experimental_enable_permute_matmul_fusion": False,
    }
    torch_xla.set_custom_compile_options(options)

    torch._dynamo.reset()
    compiled = torch.compile(wrapper, backend="tt")

    with torch.no_grad():
        out = compiled(hidden)
    xm.mark_step()
    out_cpu = out.detach().cpu()

    pcc_min = _min_pcc_across_batch_rows(golden_cpu, out_cpu)
    print(
        f"[gpt_oss_moe_block_galaxy/{case}] shape={tuple(golden_cpu.shape)} "
        f"pcc_min_across_batch={pcc_min:.6f} required={required_pcc:.6f}"
    )
    assert pcc_min >= required_pcc, (
        f"MoE block PCC failed ({case}): min batch PCC={pcc_min:.6f}, required={required_pcc:.6f}"
    )


@pytest.mark.model_test
@pytest.mark.galaxy
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.notimeout
def test_gpt_oss_20b_moe_block_galaxy_prefill_pcc(gpt_oss_20b_galaxy_moe_cpu_reference):
    """MoE MLP on prefill-shaped activations ``[batch, seq, hidden]`` vs CPU golden."""

    ref = gpt_oss_20b_galaxy_moe_cpu_reference
    _run_tt_moe_pcc(
        ref,
        ref.h_prefill,
        ref.gold_prefill,
        case="prefill",
        required_pcc=DEFAULT_REQUIRED_PCC,
    )


@pytest.mark.model_test
@pytest.mark.galaxy
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.notimeout
def test_gpt_oss_20b_moe_block_galaxy_decode_pcc(gpt_oss_20b_galaxy_moe_cpu_reference):
    """MoE MLP on first-decode activations ``[batch, 1, hidden]`` vs CPU golden."""

    ref = gpt_oss_20b_galaxy_moe_cpu_reference
    if ref.h_decode is None or ref.gold_decode is None:
        pytest.skip(
            "No decode activations: with GPT_OSS_MOE_INPUT_DUMP_DIR set, provide "
            "GPT_OSS_MOE_DECODE_HIDDEN_NPY (.npy [batch, hidden] or [batch, 1, hidden]). "
            "Otherwise unset GPT_OSS_MOE_INPUT_DUMP_DIR to use CPU hook capture for both steps."
        )
    _run_tt_moe_pcc(
        ref,
        ref.h_decode,
        ref.gold_decode,
        case="decode",
        required_pcc=DEFAULT_REQUIRED_PCC,
    )
