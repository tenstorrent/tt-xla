#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal GPT-OSS TT-XLA run mirroring the LLM perf benchmark SPMD path.

Reproduces the difference between:
  * **decode-only** batch sharding: ``input_sharding_fn`` runs only for decode
    steps (step index > 0), matching ``benchmark_llm_torch_xla`` +
    ``make_prepare_step_inputs_fn``. Each step marks ``input_ids`` on the batch
    axis. On multichip, KV + ``input_ids`` are marked once after ``transfer_to_device``
    (``mark_multichip_gpt_oss_activation_shardings``, commit 26a5555). ``--kv-cache-input-sharding``
    only adds a ``_kvbm`` suffix to export names.
  * **prefill + decode** batch sharding: ``input_sharding_fn`` runs on every
    step including prefill (step 0), which is when logits can go all-zero.

From repo root (wrapper)::

    python scripts/debug_gpt_oss_input_sharding.py --mode compare

Or from the benchmark tree::

    cd tests/benchmark
    python scripts/debug_gpt_oss_input_sharding.py --mode decode_only
    python scripts/debug_gpt_oss_input_sharding.py --mode prefill_and_decode

Default shape matches the Galaxy-style slice: 2 layers, batch **64**,
``--isl 128``, ``--decode-steps 2`` (prefill + one decode). Use
``--generated-tokens N`` for prefill + **N** new tokens (sets ``decode_steps=N+1``).
Use ``--full-layers`` for the full checkpoint (or ``--num-layers 6`` for a common traced depth).

By default this matches the **perf** benchmark path only (``return_logits=False``,
``collect_logits=False``), like ``benchmark_llm_torch_xla`` after warmup—no logits
PCC/accuracy pass. Pass ``--collect-logits`` to move logits to CPU each step and
print min/max/all_zero stats (slower; for debugging bad logits).

Exports go under a dedicated directory (default
``tests/benchmark/modules/gpt_oss_input_sharding_dbg``): pipeline MLIR for **all**
stages (vhlo, shlo, ttir, ttnn, …) in ``<export_path>/irs/``, plus flatbuffers
like ``fb_*.ttnn`` next to ``irs/``. Override with ``--export-path``.

IR filenames are ``<stage>_<mode>_g<N>_<timestamp>.mlir``: Python sets a short
``export_model_name`` (``dec`` = shard activations decode-only, ``all`` = shard
every step). By default only ``input_ids`` get batch sharding; export names are
plain ``dec``/``all``. Pass ``--kv-cache-input-sharding`` to add ``_kvbm`` to export
basenames (KV is always marked on multichip; matches galaxy benchmark). PJRT appends ``_g0``, ``_g1``, …
per graph; the compiler adds the per-file timestamp and stage prefix (``ttnn``,
``ttir``, …).

To rename **old** long-named exports under ``modules/gpt_oss_input_sharding_dbg``,
run ``python scripts/rename_gpt_oss_sharding_export_files.py`` (from
``tests/benchmark``; use ``--dry-run`` first if you like).

Per-op TTNN logging (host tensor checks) lives in the **tt-mlir runtime** only.
Either pass ``--op-tensor-trace`` (sets ``TT_RUNTIME_OP_TENSOR_TRACE=1`` and,
unless ``TTMLIR_RUNTIME_LOGGER_FILE`` was set before launch,
``…/runtime_logs/<export_model_name>_ttnn_op_trace_<stamp>.log`` (wall time + hex ns;
e.g. ``dec`` or ``dec_kvbm``; new file every run) plus
``TTMLIR_RUNTIME_LOGGER_TYPES=RuntimeTTNN``), or set those env vars yourself.
With ``--mode compare`` and ``--op-tensor-trace``, the script runs **decode_only** and
**prefill_and_decode** in **separate subprocesses** so each gets its own trace file
(tt-mlir opens the log path only once per process). Requires a tt-mlir runtime built
with op tracing. **Very slow.**

When op tracing is enabled, ``TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL`` defaults to
``1`` (unless you already exported it). That adds per-op **MESH** lines for
``mesh_shard`` (full→shard) and ``mesh_partition``: formatted input/output previews,
which slice of the global input each device output corresponds to, **fnv1a64**
checksums on raw host payloads, float sums, and checks that unique-slice sums match
the global input and that the sum over all device outputs matches the replicated
layout (see trace log ``MESH | CHECK`` lines). Set
``TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL=0`` to keep only the short ``IN/OUT`` trace.

Mesh **tensor row previews** (many lines per device) stay **off** unless you set
``TT_RUNTIME_OP_TENSOR_TRACE_MESH_PREVIEWS=1``. ``TT_RUNTIME_OP_TENSOR_TRACE_VERBOSE``
does not enable those previews.

Requires TT devices, ``TTMLIR_TOOLCHAIN_DIR``, venv with torch_xla + tt plugin,
and the same env you use for ``pytest tests/benchmark/test_llms.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

# tests/benchmark: benchmarks/, llm_utils/, utils/
# repo root: third_party/tt_forge_models/
_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent.parent

# IR + flatbuffer artifacts (relative → resolved under ``tests/benchmark/``).
DEFAULT_EXPORT_PATH = "modules/gpt_oss_input_sharding_dbg"

# First ``--op-tensor-trace`` pass: if user already set TTMLIR_RUNTIME_LOGGER_FILE,
# never rewrite it (compare mode still gets per-mode files when FILE was unset).
_mlir_op_trace_logger_file_user_supplied: bool | None = None


def _resolve_export_path(export_path: str) -> str:
    """Absolute path for PJRT: relative paths are anchored at ``tests/benchmark``."""
    p = Path(export_path)
    if p.is_absolute():
        return str(p.resolve())
    return str((_BENCH_ROOT / p).resolve())


def _ttnn_op_trace_log_filename(export_model_name: str, *, run_stamp: str | None = None) -> str:
    """Unique log basename per run so traces are never overwritten."""
    stamp = run_stamp if run_stamp is not None else (
        time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() & 0xFFFFFFFF:08x}"
    )
    return f"{export_model_name}_ttnn_op_trace_{stamp}.log"


def _configure_mlir_op_tensor_trace_env(
    *,
    export_path_resolved: str,
    export_model_name: str,
    enabled: bool,
) -> None:
    """Enable tt-mlir ``op_tensor_trace`` via env (Logger reads env at first use)."""
    if not enabled:
        return
    os.environ["TT_RUNTIME_OP_TENSOR_TRACE"] = "1"
    if "TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL" not in os.environ:
        os.environ["TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL"] = "1"
    if not os.environ.get("TTMLIR_RUNTIME_LOGGER_TYPES"):
        os.environ["TTMLIR_RUNTIME_LOGGER_TYPES"] = "RuntimeTTNN"
        print("[op-tensor-trace] TTMLIR_RUNTIME_LOGGER_TYPES=RuntimeTTNN")

    global _mlir_op_trace_logger_file_user_supplied
    if _mlir_op_trace_logger_file_user_supplied is None:
        _mlir_op_trace_logger_file_user_supplied = bool(
            os.environ.get("TTMLIR_RUNTIME_LOGGER_FILE")
        )
    if _mlir_op_trace_logger_file_user_supplied:
        print(
            "[op-tensor-trace] TT_RUNTIME_OP_TENSOR_TRACE=1; "
            "TTMLIR_RUNTIME_LOGGER_FILE unchanged (set before script)"
        )
        return

    log_dir = Path(export_path_resolved) / "runtime_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / _ttnn_op_trace_log_filename(export_model_name)
    os.environ["TTMLIR_RUNTIME_LOGGER_FILE"] = str(out.resolve())
    print(f"[op-tensor-trace] TTMLIR_RUNTIME_LOGGER_FILE={out!r}")
    lev = (os.environ.get("TTMLIR_RUNTIME_LOGGER_LEVEL") or "").strip().upper()
    if lev in ("ERROR", "FATAL", "WARNING", "WARN"):
        print(
            "[op-tensor-trace] warning: TTMLIR_RUNTIME_LOGGER_LEVEL may hide INFO; "
            "prefer unset or INFO"
        )


def _argv_for_child_replacing_mode(new_mode: str) -> list[str]:
    """``sys.argv`` for a subprocess: same flags as this run, but ``--mode`` is ``new_mode``."""
    script = Path(__file__).resolve()
    argv = list(sys.argv[1:])
    out: list[str] = []
    i = 0
    mode_set = False
    while i < len(argv):
        tok = argv[i]
        if tok == "--mode":
            mode_set = True
            out.extend(["--mode", new_mode])
            i += 2
            continue
        if tok.startswith("--mode="):
            mode_set = True
            out.append(f"--mode={new_mode}")
            i += 1
            continue
        out.append(tok)
        i += 1
    if not mode_set:
        out.extend(["--mode", new_mode])
    return [sys.executable, str(script), *out]


def _run_compare_op_tensor_trace_subprocesses(
    export_path: str, *, shard_kv_cache: bool
) -> None:
    """Run decode vs all in fresh processes so tt-mlir Logger uses one file each."""
    ep = _resolve_export_path(export_path)
    log_root = Path(ep) / "runtime_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    compare_stamp = (
        time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() & 0xFFFFFFFF:08x}"
    )
    pairs = [
        ("decode_only (compare leg 1)", "decode_only", True),
        ("prefill_and_decode (compare leg 2)", "prefill_and_decode", False),
    ]
    for label, mode_arg, decode_only in pairs:
        name = _export_model_name_for_mode(
            decode_only=decode_only, shard_kv_cache=shard_kv_cache
        )
        log_path = (
            log_root / _ttnn_op_trace_log_filename(name, run_stamp=compare_stamp)
        ).resolve()
        print(
            f"\n{'='*60}\n SUBPROCESS: {label}\n"
            f" TTMLIR_RUNTIME_LOGGER_FILE={log_path!s}\n{'='*60}"
        )
        env = os.environ.copy()
        env["TT_RUNTIME_OP_TENSOR_TRACE"] = "1"
        env.setdefault("TT_RUNTIME_OP_TENSOR_TRACE_MESH_DETAIL", "1")
        env["TTMLIR_RUNTIME_LOGGER_FILE"] = str(log_path)
        if not env.get("TTMLIR_RUNTIME_LOGGER_TYPES"):
            env["TTMLIR_RUNTIME_LOGGER_TYPES"] = "RuntimeTTNN"
        cmd = _argv_for_child_replacing_mode(mode_arg)
        r = subprocess.run(cmd, cwd=str(_BENCH_ROOT), env=env)
        if r.returncode != 0:
            raise SystemExit(r.returncode)


def _prepend_sys_path(path: Path) -> None:
    """Put ``path`` at sys.path[0], removing earlier duplicates.

    If ``PYTHONPATH`` includes ``tests/`` but not ``tests/benchmark``, a plain
    ``import utils`` resolves to ``tests/utils.py`` unless the benchmark tree
    is always first.
    """
    s = str(path.resolve())
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


# Repo root first, then benchmark — second prepend wins front, so benchmark is [0].
_prepend_sys_path(_REPO_ROOT)
_prepend_sys_path(_BENCH_ROOT)

from benchmarks.llm_benchmark import (  # noqa: E402
    construct_inputs,
    get_mesh,
    mark_multichip_gpt_oss_activation_shardings,
    setup_model_and_tokenizer,
    transfer_to_device,
)
from llm_utils.decode_utils import LLMSamplingWrapper, generate_and_benchmark  # noqa: E402
from tt_torch.sharding import (  # noqa: E402
    sharding_constraint_hook,
    sharding_constraint_tensor,
)
from tt_torch.sparse_mlp import enable_sparse_mlp, get_moe_shard_specs  # noqa: E402
from tt_torch.weight_dtype import apply_weight_dtype_overrides  # noqa: E402
from utils import compute_pcc_flat_pair, create_model_loader  # noqa: E402

xr.set_device_type("TT")


def _export_model_name_for_mode(*, decode_only: bool, shard_kv_cache: bool) -> str:
    """Short PJRT ``export_model_name``; runtime appends ``_gN``, files add timestamp.

    * ``dec`` / ``all``: default basename (KV + input_ids marked on multichip after transfer).
    * ``dec_kvbm`` / ``all_kvbm``: same sharding; ``_kvbm`` suffix when ``shard_kv_cache``.
    """
    base = "dec" if decode_only else "all"
    if shard_kv_cache:
        return f"{base}_kvbm"
    return base


def _make_batch_parallel_input_sharding_fn() -> Callable[[Mesh, dict], None]:
    """Per-step ``input_ids`` batch sharding (KV is marked once after transfer; see 26a5555)."""

    def _fn(mesh: Mesh, input_args: dict) -> None:
        xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))

    return _fn


def _gpt_oss_20b_mesh_config_fn(model_loader, num_devices: int):
    return (1, num_devices), ("batch", "model")


def _gpt_oss_20b_shard_spec_fn(model_loader, model):
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        shard_specs[layer.self_attn.sinks] = (None,)
        shard_specs[layer.mlp.router.weight] = (None, None)
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", None)
    return shard_specs


def _gpt_oss_galaxy_mesh_config_fn(model_loader, num_devices: int):
    if num_devices != 32:
        raise ValueError(
            "Galaxy mesh expects 32 devices; got %s. Use --mesh llmbox for 8 devices."
            % num_devices
        )
    return (4, 8), ("batch", "model")


def _gpt_oss_galaxy_shard_spec_fn(model_loader, model):
    batch_axis = None
    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
    shard_specs[model.model.norm.weight] = (batch_axis,)
    shard_specs[model.lm_head.weight] = (None, None)
    for layer in model.model.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
        shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
        shard_specs[layer.self_attn.sinks] = ("model",)
        shard_specs[layer.mlp.router.weight] = (None, batch_axis)
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", None)
        shard_specs[layer.input_layernorm.weight] = (batch_axis,)
        shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)
    return shard_specs


def _pick_mesh_and_shard_fns(mesh_name: str, num_devices: int):
    if mesh_name == "galaxy":
        return _gpt_oss_galaxy_mesh_config_fn, _gpt_oss_galaxy_shard_spec_fn
    if mesh_name == "llmbox":
        if num_devices != 8:
            raise ValueError(
                "llmbox layout expects 8 devices; got %s. Use --mesh galaxy for 32."
                % num_devices
            )
        return _gpt_oss_20b_mesh_config_fn, _gpt_oss_20b_shard_spec_fn
    raise ValueError(mesh_name)


def _compute_cpu_reference(
    model: torch.nn.Module,
    tokenizer,
    batch_size: int,
    max_cache_len: int,
) -> torch.Tensor:
    """Run a single prefill forward on CPU and return step-0 logits."""
    input_args = construct_inputs(tokenizer, model.config, batch_size, max_cache_len)
    cpu_wrapper = LLMSamplingWrapper(model, _read_logits, return_logits=True)
    cpu_wrapper.eval()
    cpu_output_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )
    return cpu_output_logits[0]


def _compare_pcc(
    tt_logits: torch.Tensor,
    cpu_logits: torch.Tensor,
    required_pcc: float,
) -> dict:
    """Per-batch PCC of step-0 logits; returns summary dict."""
    if tt_logits.shape != cpu_logits.shape:
        raise AssertionError(
            f"[PCC] Logits shape mismatch: TT {tuple(tt_logits.shape)} "
            f"vs CPU {tuple(cpu_logits.shape)}"
        )
    batch_dim = tt_logits.shape[0]
    pcc_by_batch: list[float] = []
    for bi in range(batch_dim):
        try:
            pcc_bi = compute_pcc_flat_pair(cpu_logits[bi], tt_logits[bi])
        except AssertionError as exc:
            if "denominator is zero" not in str(exc):
                raise
            print(
                f"[PCC] batch {bi}/{batch_dim}: skipping PCC "
                "(degenerate logits, e.g. constant or all-zero device output)"
            )
            continue
        pcc_by_batch.append(pcc_bi)
        print(
            f"[PCC] batch {bi}/{batch_dim} "
            f"shape={tuple(tt_logits[bi].shape)} PCC={pcc_bi:.6f}"
        )
    if not pcc_by_batch:
        print(
            "[PCC] skipped: no valid PCC (all batch rows had degenerate logits); "
            "continuing without PCC threshold check"
        )
        return {
            "pcc_min": float("nan"),
            "pcc_mean": float("nan"),
            "required_pcc": required_pcc,
            "passed": False,
        }
    pcc_min = min(pcc_by_batch)
    pcc_mean = float(sum(pcc_by_batch) / len(pcc_by_batch))
    passed = pcc_min >= required_pcc
    print(
        f"[PCC] summary: min={pcc_min:.6f} mean={pcc_mean:.6f} "
        f"required={required_pcc:.6f} {'PASS' if passed else 'FAIL'}"
    )
    return {
        "pcc_min": pcc_min,
        "pcc_mean": pcc_mean,
        "required_pcc": required_pcc,
        "passed": passed,
    }


def _read_logits(output):
    return output.logits


def _make_prepare_step_inputs_fn(
    mesh: Mesh,
    input_sharding_fn: Callable[[Mesh, dict], None],
    *,
    decode_only: bool,
):
    """Match ``benchmark_llm_torch_xla``: if decode_only, skip sharding on step 0."""

    if not decode_only:

        def _prepare(input_args: dict) -> None:
            input_sharding_fn(mesh, input_args)

        return _prepare

    step = {"i": 0}

    def _prepare(input_args: dict) -> None:
        if step["i"] > 0:
            input_sharding_fn(mesh, input_args)
        step["i"] += 1

    return _prepare


def _logits_step_stats(logits: torch.Tensor, step: int) -> dict:
    x = logits.detach().float().cpu()
    return {
        "step": step,
        "shape": tuple(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "all_zero": bool((x == 0).all().item()),
        "any_nan": bool(torch.isnan(x).any().item()),
    }


def setup_spmd_gpt_oss(
    *,
    model: torch.nn.Module,
    model_loader,
    mesh_name: str,
    inject_custom_moe: bool = False,
) -> tuple[torch.nn.Module, torch.device, Mesh]:
    """Move model to device, mark weight shardings, lm_head hook, weight-dtype overrides.

    The model must already be loaded on CPU (see ``setup_model_and_tokenizer``).
    """
    num_devices = xr.global_runtime_device_count()
    mesh_cfg_fn, shard_spec_fn = _pick_mesh_and_shard_fns(mesh_name, num_devices)

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    device = torch_xla.device()
    model = model.to(device, dtype=torch.bfloat16)

    mesh = get_mesh(model_loader, mesh_cfg_fn)
    if inject_custom_moe:
        mesh_info = mesh_cfg_fn(model_loader, num_devices)
        mesh_names = mesh_info[1] if isinstance(mesh_info, tuple) else ("batch", "model")
        shard_specs = get_moe_shard_specs(
            model, lambda m: shard_spec_fn(model_loader, m), mesh_names
        )
    else:
        shard_specs = shard_spec_fn(model_loader, model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    if hasattr(model, "lm_head") and model.lm_head is not None:
        hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
        model.lm_head.register_forward_hook(hook)

    weight_dtype_config = model_loader.get_weight_dtype_config_path()
    if weight_dtype_config:
        apply_weight_dtype_overrides(model, weight_dtype_config)

    return model, device, mesh


def run_sharding_mode(
    *,
    decode_only: bool,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    mesh: Mesh,
    input_sharding_fn: Callable[[Mesh, dict], None],
    batch_size: int,
    max_cache_len: int,
    decode_steps: int,
    optimization_level: int,
    experimental_weight_dtype: str,
    experimental_enable_permute_matmul_fusion: bool,
    export_model_name: str,
    collect_logits: bool,
    export_path: str,
    op_tensor_trace: bool = False,
    cpu_logits: Optional[torch.Tensor] = None,
    required_pcc: float = 0.99,
) -> list[dict]:
    """Fresh StaticCache + compile + timed generate (perf or logits debug)."""
    torch._dynamo.reset()

    export_path_resolved = _resolve_export_path(export_path)
    _configure_mlir_op_tensor_trace_env(
        export_path_resolved=export_path_resolved,
        export_model_name=export_model_name,
        enabled=op_tensor_trace,
    )

    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
    )

    # Make each batch row unique so batch-axis sharding produces visibly
    # different per-device OPSUM values (default prompt is identical for all rows).
    ids = input_args["input_ids"]
    vocab_size = model.config.vocab_size
    offset = torch.arange(batch_size, device=ids.device).unsqueeze(1) % vocab_size
    input_args["input_ids"] = (ids + offset) % vocab_size

    input_args = transfer_to_device(input_args, device)

    if xr.global_runtime_device_count() > 1:
        mark_multichip_gpt_oss_activation_shardings(
            mesh, input_args, mark_input_ids=decode_only
        )

    print(
        f"[export] export_path={export_path_resolved!r} "
        f"(irs: <stage>_{export_model_name}_gN_<timestamp>.mlir; fb*.ttnn in DIR)"
    )
    print(f"[export] export_model_name={export_model_name!r} (PJRT appends _g0, _g1, …)")
    options = {
        "optimization_level": optimization_level,
        "enable_trace": False,
        "export_path": export_path_resolved,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": False,
        "experimental_weight_dtype": experimental_weight_dtype,
        "experimental_enable_permute_matmul_fusion": experimental_enable_permute_matmul_fusion,
    }
    torch_xla.set_custom_compile_options(options)

    output_sharding_constraint = None
    if xr.global_runtime_device_count() > 1:
        output_sharding_constraint = lambda t, _m=mesh: sharding_constraint_tensor(
            t, _m, ("batch", None)
        )

    wrapper = LLMSamplingWrapper(
        model,
        _read_logits,
        return_logits=collect_logits,
        output_sharding_constraint=output_sharding_constraint,
    )
    wrapper.eval()

    compiled = torch.compile(wrapper, backend="tt")

    prepare_fn = _make_prepare_step_inputs_fn(
        mesh, input_sharding_fn, decode_only=decode_only
    )

    output_logits, times_ns = generate_and_benchmark(
        compiled,
        input_args,
        device,
        decode_steps,
        verbose=True,
        collect_logits=collect_logits,
        prepare_step_inputs_fn=prepare_fn,
        tokenizer=tokenizer if collect_logits else None,
    )

    pcc_result = None
    if cpu_logits is not None and collect_logits and output_logits:
        pcc_result = _compare_pcc(output_logits[0], cpu_logits, required_pcc)

    stats = []
    if collect_logits:
        for i, logits in enumerate(output_logits):
            stats.append(_logits_step_stats(logits, i))
            stats[-1]["time_ms"] = times_ns[i] / 1e6
    else:
        for i, t_ns in enumerate(times_ns):
            stats.append({"step": i, "time_ms": t_ns / 1e6})
    if pcc_result is not None:
        stats.append(pcc_result)

    xm.mark_step()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("decode_only", "prefill_and_decode", "compare"),
        default="compare",
        help="compare runs decode_only then prefill_and_decode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="default 64 (Galaxy benchmark batch)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=128,
        metavar="N",
        help="prompt length / max_cache_len (default 128, Galaxy benchmark -isl)",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=2,
        help="total forward steps (default 2 = prefill + 1 decode)",
    )
    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "prefill then N autoregressive decode steps (N new tokens); "
            "sets decode_steps=N+1. Overrides --decode-steps when set."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        metavar="N",
        help="hidden layers to load (default 2). Ignored if --full-layers",
    )
    parser.add_argument(
        "--full-layers",
        action="store_true",
        help="load full model (all layers); overrides --num-layers",
    )
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--experimental-weight-dtype",
        type=str,
        default="bfp_bf8",
        help='e.g. "bfp_bf8" or "" to disable',
    )
    parser.add_argument(
        "--mesh",
        choices=("galaxy", "llmbox"),
        default="galaxy",
        help="galaxy=4x8 (32 devices), llmbox=1x8 (8 devices)",
    )
    parser.add_argument(
        "--variant",
        choices=("20b", "120b"),
        default="20b",
    )
    parser.add_argument(
        "--collect-logits",
        action="store_true",
        help="move logits to CPU each step and print stats (accuracy-style; slower)",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=DEFAULT_EXPORT_PATH,
        metavar="DIR",
        help=(
            "PJRT export root (MLIR in DIR/irs/ for all stages; fb*.ttnn in DIR). "
            f"Default {DEFAULT_EXPORT_PATH!r} under tests/benchmark if relative."
        ),
    )
    parser.add_argument(
        "--op-tensor-trace",
        action="store_true",
        help=(
            "tt-mlir per-op tensor trace (env). With --mode compare, runs each leg in "
            "a subprocess so each mode gets runtime_logs/<export_model_name>_ttnn_op_trace_<stamp>.log."
        ),
    )
    parser.add_argument(
        "--kv-cache-input-sharding",
        action="store_true",
        help=(
            "Append _kvbm to export basenames. On multichip, KV + input_ids are always "
            "marked after transfer (same as llm benchmark); this flag does not disable "
            "KV marking."
        ),
    )
    parser.add_argument(
        "--pcc",
        action="store_true",
        help=(
            "Compute PCC against CPU reference logits (prefill step). "
            "Implies --collect-logits. Runs a CPU forward before device setup."
        ),
    )
    parser.add_argument(
        "--required-pcc",
        type=float,
        default=0.99,
        metavar="F",
        help="minimum PCC threshold (default 0.99)",
    )
    parser.add_argument(
        "--inject-custom-moe",
        action="store_true",
        help="Replace HF MoE MLP with A2aSparseMLP (avoids iota sharding bug)",
    )
    args = parser.parse_args()
    if args.pcc:
        args.collect_logits = True
    kv_cache_input_sharding = args.kv_cache_input_sharding

    decode_steps = args.decode_steps
    if args.generated_tokens is not None:
        if args.generated_tokens < 0:
            parser.error("--generated-tokens must be >= 0")
        decode_steps = args.generated_tokens + 1

    if args.mode == "compare" and args.op_tensor_trace:
        _run_compare_op_tensor_trace_subprocesses(
            args.export_path, shard_kv_cache=kv_cache_input_sharding
        )
        return

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = (
        ModelVariant.GPT_OSS_20B
        if args.variant == "20b"
        else ModelVariant.GPT_OSS_120B
    )
    num_layers = None if args.full_layers else args.num_layers
    model_loader = create_model_loader(ModelLoader, num_layers=num_layers, variant=variant)
    if num_layers is not None and model_loader is None:
        raise SystemExit("ModelLoader does not support num_layers override.")
    modes = []
    if args.mode == "compare":
        modes = [("decode_only", True), ("prefill_and_decode", False)]
    elif args.mode == "decode_only":
        modes = [("decode_only", True)]
    else:
        modes = [("prefill_and_decode", False)]

    fusion = False
    wdtype = args.experimental_weight_dtype

    model, tokenizer = setup_model_and_tokenizer(model_loader, variant)

    if args.inject_custom_moe:
        num_devices = xr.global_runtime_device_count()
        mesh_cfg_fn, _ = _pick_mesh_and_shard_fns(args.mesh, num_devices)
        mesh_info = mesh_cfg_fn(model_loader, num_devices)
        mesh_shape = mesh_info[0] if isinstance(mesh_info, tuple) else mesh_info
        print(f"[SparseMLP] Injecting custom MoE (mesh={mesh_shape})")
        enable_sparse_mlp(model, mesh=mesh_shape)

    cpu_logits = None
    if args.pcc:
        print("[PCC] Computing CPU reference logits (prefill, 1 step) …")
        cpu_logits = _compute_cpu_reference(
            model, tokenizer, args.batch_size, args.isl
        )
        print(f"[PCC] CPU reference shape: {tuple(cpu_logits.shape)}")

    model, device, mesh = setup_spmd_gpt_oss(
        model=model,
        model_loader=model_loader,
        mesh_name=args.mesh,
        inject_custom_moe=args.inject_custom_moe,
    )

    input_sharding_fn = _make_batch_parallel_input_sharding_fn()

    for label, decode_only in modes:
        export_model_name = _export_model_name_for_mode(
            decode_only=decode_only, shard_kv_cache=kv_cache_input_sharding
        )
        print(f"\n{'='*60}\n MODE: {label} (decode_only={decode_only})\n{'='*60}")
        stats = run_sharding_mode(
            decode_only=decode_only,
            model=model,
            tokenizer=tokenizer,
            device=device,
            mesh=mesh,
            input_sharding_fn=input_sharding_fn,
            batch_size=args.batch_size,
            max_cache_len=args.isl,
            decode_steps=decode_steps,
            optimization_level=args.optimization_level,
            experimental_weight_dtype=wdtype,
            experimental_enable_permute_matmul_fusion=fusion,
            export_model_name=export_model_name,
            collect_logits=args.collect_logits,
            export_path=args.export_path,
            op_tensor_trace=args.op_tensor_trace,
            cpu_logits=cpu_logits,
            required_pcc=args.required_pcc,
        )
        for row in stats:
            print(row)


if __name__ == "__main__":
    main()
