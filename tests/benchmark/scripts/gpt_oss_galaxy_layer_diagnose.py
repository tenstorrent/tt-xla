#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
r"""GPT-OSS 20B Galaxy (32-chip) layer diagnosis: full 1-layer TT baseline vs CPU, then bisect.

================================================================================
DEVICE SAFETY (read before every run)
================================================================================

* Run **one** TT job at a time. Do not start Tracy, pytest, or this script while another
  compile or decode is still using the mesh.
* If you **cancel** or **kill** a run mid-flight (especially after a hang or ``TT_FATAL``),
  assume the device needs recovery: **do not** immediately start a new job.
* **Reset** must be performed **once** and allowed to **finish** before the next run.
  Typical recovery is ``tt-smi -r`` (or your lab’s equivalent). Wait until reset completes.
* If reset fails, stop; do not retry profiling or this script until hardware is healthy.

This matches the failure-recovery guidance in ``.cursor/skills/layer-profiling/SKILL.md``:
do not reset on the user’s behalf from automation; in the lab, you run reset yourself
and wait until it is done before relaunching.

================================================================================
How to run layer profiling yourself (from the same skill)
================================================================================

* Work **inside** the TT-XLA Docker image; ``cd`` to the repo and ``source venv/activate``
  (always from the repo root so ``venv/activate`` resolves correctly).
* **GPT-OSS Tracy example** (6 layers, not 1L baseline):

  ``tracy -p -r --sync-host-device -o <ARTIFACT_DIR>/gpt_oss_20b_tp/<RUN_ID>/raw \``
  ``  -m pytest -sv tests/benchmark/test_llms.py -k test_gpt_oss_20b_tp \``
  ``  --num-layers 6 --max-output-tokens 3``

* For CSV-only work, start from ``ops_perf_results_*.csv`` and follow the skill’s decode_2
  window + ``tt-perf-report`` handoff (``.cursor/skills/tt-perf-report/SKILL.md``).

================================================================================
What this script does
================================================================================

**baseline** — Load **one** GPT-OSS 20B layer (``--num-layers 1``), batch **64**, cache **128**,
same mesh/shard/activation policy as ``test_gpt_oss_20b_tp_galaxy_batch_size_64``:

* ``_gpt_oss_galaxy_mesh_config_fn``, ``_gpt_oss_galaxy_shard_spec_fn``,
  ``_batch_parallel_input_sharding_fn``, ``mark_multichip_gpt_oss_activation_shardings``,
  ``lm_head`` sharding-constraint hook (benchmark path).
* CPU prefill + CPU reference **first decode** logits (``build_cpu_decode_start_after_prefill``
  with capture + KV restore, same as ``llm_benchmark`` PCC path).
* Full model on TT, **one decode** forward, ``torch.compile(..., backend="tt")``.
* Prints PCC (last token, min over batch), max |logit|, and whether TT logits are all zero.

If the known issue is present, PCC should be **very low** and/or TT outputs **all zero**.

* Optional: ``GPT_OSS_LAYER_DIAGNOSE_DEBUG=1`` prints layer-0 KV min/max/mean abs before device
  transfer and again after SPMD marks (with an XLA sync) to see if the cache is zeroed or
  degenerate before decode.

**bisect** — Runs ``bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py`` (**decode** token hybrid).

**bisect_prefill** — Runs ``bisect_gpt_oss_layer_cpu_attn_tt_moe_prefill.py`` (**prefill**
sequence hybrid: CPU attention + TT MoE at layer L over full prompt). Use after
``prefill_check`` to localize TT vs CPU mismatch to MoE vs tail during long-seq forward.

Both bisects are **separate** processes; run only after the device is idle. Use
``--tt-scope mlp`` or ``post_norm_mlp``.

**kv_tt_prefill** — Same mesh/sharding as **baseline**, but fills ``StaticCache`` with a **TT
prefill** forward (full prompt on device), then one **TT decode** forward. Compares decode
logits to the CPU golden and reports layer-0 K/V PCC vs CPU-filled cache. Use this when you
suspect **KV transfer** from CPU (``transfer_to_device`` + sharding marks) rather than the
decode graph: if **baseline** is bad but **kv_tt_prefill** decode PCC is good, the bug likely
lies in how transferred KV is used; if both are bad, attention/MoE or sharding may dominate.

**prefill_check** — One **prefill** forward only (full prompt, no decode step): compare **last
position logits** CPU vs TT on the same tokenized prompt. Isolates embeddings + full prefill
graph (all layers once over the sequence) from **decode** and from **KV transfer** semantics.
Run **after** ``baseline`` if you need to know whether mismatch starts in prefill; good
prefill PCC but bad ``baseline`` decode → focus on decode step, ``StaticCache`` transfer, or
single-token path.

**isolate** — Prints a **subsystem map** for one GPT-OSS decoder layer and an **ordered
decision tree**: which command pins which block (norm, attention, MoE, KV, lm_head) when
results agree or disagree. Run once on paper before hardware.

Run **baseline**, **prefill_check**, and **kv_tt_prefill** as **separate** invocations (one TT
job at a time). Suggested order: ``baseline`` (CPU prefill + TT decode) → ``prefill_check``
(prefill-only PCC) → ``kv_tt_prefill`` (TT prefill KV + decode) → ``bisect``.

Run from ``tests/benchmark``::

    python scripts/gpt_oss_galaxy_layer_diagnose.py baseline
    # wait for device idle; reset if you had to cancel anything
    python scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py --layer 0 --tt-scope mlp
    python scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py --layer 0 --tt-scope post_norm_mlp

Or::

    python scripts/gpt_oss_galaxy_layer_diagnose.py bisect_prefill -- --layer 0 --tt-scope mlp
    python scripts/gpt_oss_galaxy_layer_diagnose.py bisect -- --layer 0 --tt-scope mlp
    python scripts/gpt_oss_galaxy_layer_diagnose.py prefill_check
    python scripts/gpt_oss_galaxy_layer_diagnose.py kv_tt_prefill
    python scripts/gpt_oss_galaxy_layer_diagnose.py isolate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent.parent
sys.path.insert(0, str(_BENCH_ROOT))
sys.path.insert(0, str(_REPO_ROOT))


# One GPT-OSS decoder layer (HuggingFace-style), in evaluation order:
#   input_layernorm → self_attn (Q/K/V, RoPE, mask, KV cache) → residual
#   post_attention_layernorm → mlp (MoE: router, top-k, experts) → residual
# Even/odd layers differ by attention pattern (full context vs sliding window); use
# bisect --layer N to match the layer you care about.
_ISOLATION_PLAYBOOK = """
================================================================================
Isolate which GPT-OSS block differs from CPU (Galaxy 32-chip, decode-focused)
================================================================================

Subsystem map (single decoder layer L)
----------------------------------------
  [A] input_layernorm
  [B] self_attn  — projections, RoPE, softmax/mask, KV read/write
  [C] residual (post-attention)
  [D] post_attention_layernorm
  [E] mlp        — MoE router, top-k, expert FFNs, combine
  [F] residual (post-MoE)

Above the stack: embeddings + final norm + lm_head (logits). The benchmark fills KV via
CPU prefill then runs one TT decode step unless you use kv_tt_prefill.

Ordered checks (run one TT job at a time; device idle between commands)
-----------------------------------------------------------------------
  Step 1 — CPU prefill + TT decode (golden = CPU first-decode logits)
    python scripts/gpt_oss_galaxy_layer_diagnose.py baseline
    # optional: GPT_OSS_LAYER_DIAGNOSE_DEBUG=1 → llm_utils.gpt_oss_kv_debug stats
  Pins: full decode path: transferred KV + single TT token forward + lm_head.

  Step 2 — Prefill only: last-position logits CPU vs TT (same prompt, no decode step)
    python scripts/gpt_oss_galaxy_layer_diagnose.py prefill_check
  Pins: embeddings + entire stack over the prefill sequence (no decode / no CPU KV handoff).
    • Good Step 2 PCC, bad Step 1  → bug likely in **decode** graph, **cache_position**,
      or **StaticCache** after prefill (align with Step 3–4).
    • Bad Step 2  → prefill on TT already wrong (sharding, compile, attention over long seq).

  Step 3 — TT writes KV (prefill on TT), then TT decode vs same CPU golden as baseline
    python scripts/gpt_oss_galaxy_layer_diagnose.py kv_tt_prefill
  Compare to Step 1:
    • Step 1 bad, Step 3 good  → CPU→TT KV **transfer** / sharding around transferred cache.
    • Step 1 bad, Step 3 bad  → not fixed by on-device prefill; deeper TT vs CPU mismatch.

  Step 4 — MoE block only (same mesh/MoE shards; activations from real CPU layer boundary)
    cd tests/benchmark
    pytest -sv test_gpt_oss_moe_block_galaxy.py -k decode_pcc
  Pins: [E] only (prefill and decode shapes as in that test). If this fails, router/top-k/
  experts/layout are suspect before mixing in attention.

  Step 5 — Hybrid full model: CPU runs [A][B][C]; TT runs [E] with CPU [D] output as input
    python scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py --layer L --tt-scope mlp
    # optional: GPT_OSS_LAYER_DIAGNOSE_DEBUG=1 prints CPU KV after prefill (bisect keeps KV on CPU)
  Pins: integration of [E] on TT under real decode, with [D] on CPU.

  Step 6 — Hybrid: CPU runs [A][B][C]; TT runs [D][E][F] tail in one compiled graph
    python scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py --layer L --tt-scope post_norm_mlp
  Pins: post-attention tail on TT (norm + MoE + residual add) without splitting [D] from [E].

  Prefill-phase hybrid (full prompt seq; same as decode bisect but S>1 at layer L)
    python scripts/bisect_gpt_oss_layer_cpu_attn_tt_moe_prefill.py --layer L --tt-scope mlp
    # or: python scripts/gpt_oss_galaxy_layer_diagnose.py bisect_prefill -- --layer 0 --tt-scope mlp
  Pins: whether TT MoE (or post_norm+mlp) matches CPU **during prefill** vs only at decode.
    Good decode bisect + bad prefill bisect → shape/compile path differs seq>1 vs seq==1.

Reading combinations (logits PCC vs same CPU golden decode)
-----------------------------------------------------------
  • Step 4 OK, Step 5 FAIL  → likely compile/sharding/device boundary around MoE in the full
    model, or decode-step context, not raw MoE math on fixed activations.

  • Step 5 OK, Step 1 FAIL  → use Step 2 (prefill TT vs CPU) vs Step 3 (KV path) to split
    prefill, decode, and transfer.

  • Step 5 FAIL, Step 6 OK  → suspect crossing the CPU→TT boundary between [D] and [E] when
    norm stays on CPU (mlp scope) vs keeping [D]+[E] together on TT (post_norm_mlp scope).

  • Step 5 FAIL, Step 6 FAIL  → [E] (MoE) or shared tail compile path; align with Step 4.

  • Step 1 OK  → no isolation needed on this configuration.

Prefill_check vs bisect_prefill (pinpoint long-seq MoE vs rest of TT prefill)
-----------------------------------------------------------------------------
  • prefill_check FAIL, bisect_prefill OK at layer L (mlp or post_norm_mlp)
      → Layer L TT MoE (or D+E+F tail) matches CPU on the full prompt; the bug is not that
        block in isolation. Suspect other layers, embeddings, final norm, lm_head, or how
        the full compiled prefill differs from hybrid (e.g. only one layer on TT).
  • prefill_check FAIL, bisect_prefill FAIL at L
      → Mismatch appears when TT runs MoE (or post_norm+mlp) at L over seq>1; compare Step 4
        (MoE-only pytest) and exported IR for [batch, S, hidden].
  • prefill_check OK, baseline FAIL
      → Prefill logits agree; failure is decode path, cache_position, or StaticCache after CPU
        prefill (see Step 3 kv_tt_prefill).

Not automated here (would need Galaxy shard specs for Q/K/V/O)
----------------------------------------------------------------
  TT self_attn only while rest on CPU would isolate [B] directly; bisect script documents
  that gap. Op-level replay: tests/benchmark/scripts/test_topk_gpt_oss.py and related dumps.

Trace / MLIR alignment
----------------------
  Map Tracy or TTNN op windows to [A]–[F] using .cursor/skills/gpt-oss-layer-parsing/SKILL.md.
================================================================================
""".strip()


def _print_isolation_playbook() -> None:
    print(_ISOLATION_PLAYBOOK)


def _print_runbook() -> None:
    """Print the device-safety + layer-profiling sections from this module docstring."""

    import ast

    src = Path(__file__).read_text(encoding="utf-8")
    mod = ast.parse(src)
    doc = ast.get_docstring(mod)
    if doc:
        print(doc)
    else:
        print("(no module docstring)")


def _run_baseline(*, batch_size: int, isl: int, optimization_level: int) -> None:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from loguru import logger
    from test_llms import (
        _batch_parallel_input_sharding_fn,
        _gpt_oss_galaxy_mesh_config_fn,
        _gpt_oss_galaxy_shard_spec_fn,
    )
    from tt_torch.sharding import sharding_constraint_hook
    from tt_torch.weight_dtype import apply_weight_dtype_overrides

    from benchmarks.llm_benchmark import (  # noqa: E402
        DEFAULT_INPUT_PROMPT,
        MODULE_EXPORT_PATH,
        build_cpu_decode_start_after_prefill,
        check_transformers_version,
        get_mesh,
        setup_model_and_tokenizer,
        transfer_to_device,
    )
    from llm_utils.gpt_oss_kv_debug import debug_kv_report  # noqa: E402
    from utils import build_xla_export_name, compute_pcc_flat_pair, create_model_loader  # noqa: E402

    xr.set_device_type("TT")
    check_transformers_version()

    n = xr.global_runtime_device_count()
    if n != 32:
        raise SystemExit(f"Galaxy baseline expects 32 TT devices, got {n}.")

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

    model_loader = create_model_loader(ModelLoader, num_layers=1, variant=ModelVariant.GPT_OSS_20B)
    if model_loader is None:
        raise SystemExit("ModelLoader does not support num_layers.")

    model, tokenizer = setup_model_and_tokenizer(model_loader, ModelVariant.GPT_OSS_20B)

    def read_logits(o):
        return o.logits

    input_args, golden_logits, prefill_len = build_cpu_decode_start_after_prefill(
        model=model,
        tokenizer=tokenizer,
        model_config=model.config,
        batch_size=batch_size,
        max_cache_len=isl,
        custom_input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
        read_logits_fn=read_logits,
        accuracy_testing=False,
        token_accuracy=None,
        capture_cpu_decode_logits_for_pcc=True,
    )
    print(
        f"[baseline] prefill_len={prefill_len} decode input_ids={tuple(input_args['input_ids'].shape)} "
        f"golden_logits={tuple(golden_logits.shape)}"
    )

    debug_kv_report(
        input_args["past_key_values"], "after_cpu_prefill_before_transfer", flush_xla=False
    )

    device = torch_xla.device()
    input_args = transfer_to_device(input_args, device)
    model = model.to(device=device, dtype=torch.bfloat16)

    mesh = get_mesh(model_loader, _gpt_oss_galaxy_mesh_config_fn)
    shard_specs = _gpt_oss_galaxy_shard_spec_fn(model_loader, model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    from benchmarks.llm_benchmark import mark_multichip_gpt_oss_activation_shardings  # noqa: E402

    mark_multichip_gpt_oss_activation_shardings(mesh, input_args)
    _batch_parallel_input_sharding_fn(mesh, input_args)

    debug_kv_report(
        input_args["past_key_values"],
        "after_transfer_and_spmd_marks_pre_compile",
        flush_xla=True,
    )

    if hasattr(model, "lm_head") and model.lm_head is not None:
        hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
        model.lm_head.register_forward_hook(hook)

    wd = model_loader.get_weight_dtype_config_path()
    if wd:
        applied = apply_weight_dtype_overrides(model, wd)
        logger.info(f"[baseline] weight dtype overrides: {len(applied)} from {wd}")

    export_name = build_xla_export_name(
        model_name="gpt_oss_1layer_galaxy_baseline_decode",
        num_layers=1,
        batch_size=batch_size,
        input_sequence_length=isl,
    )
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": optimization_level,
            "enable_trace": False,
            "export_path": MODULE_EXPORT_PATH,
            "export_model_name": export_name,
            "ttnn_perf_metrics_enabled": False,
            "experimental_weight_dtype": "bfp_bf8",
            "experimental_enable_permute_matmul_fusion": False,
        }
    )

    torch._dynamo.reset()
    compiled = torch.compile(model, backend="tt")

    with torch.no_grad():
        out = compiled(
            input_ids=input_args["input_ids"],
            past_key_values=input_args["past_key_values"],
            cache_position=input_args["cache_position"],
            use_cache=True,
        )
    xm.mark_step()
    tt_logits = out.logits.detach().cpu()

    g = golden_logits.float()
    t = tt_logits.float()
    pccs = [
        compute_pcc_flat_pair(g[b, -1], t[b, -1]) for b in range(g.shape[0])
    ]
    pcc_min = min(pccs)
    tmax = float(t.abs().max().item())
    all_zero = bool((t == 0).all().item())

    print(
        f"[baseline] TT logits: max_abs={tmax:.6g} all_zero={all_zero} "
        f"pcc_min_last_token={pcc_min:.6f} (over batch rows)"
    )
    print(
        "[baseline] Interpretation: if all_zero or pcc_min near 0, full 1L Galaxy decode "
        "on TT is broken; run bisect (CPU attn + TT MoE scopes) next, one job at a time."
    )


def _run_prefill_check(*, batch_size: int, isl: int, optimization_level: int) -> None:
    """One prefill forward on CPU vs TT; compare logits at last prompt position only."""

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from loguru import logger
    from test_llms import (
        _batch_parallel_input_sharding_fn,
        _gpt_oss_galaxy_mesh_config_fn,
        _gpt_oss_galaxy_shard_spec_fn,
    )
    from tt_torch.sharding import sharding_constraint_hook
    from tt_torch.weight_dtype import apply_weight_dtype_overrides

    from benchmarks.llm_benchmark import (  # noqa: E402
        DEFAULT_INPUT_PROMPT,
        MODULE_EXPORT_PATH,
        check_transformers_version,
        construct_inputs,
        get_mesh,
        mark_multichip_gpt_oss_activation_shardings,
        setup_model_and_tokenizer,
        transfer_to_device,
    )
    from llm_utils.gpt_oss_kv_debug import debug_kv_report  # noqa: E402
    from utils import build_xla_export_name, compute_pcc_flat_pair, create_model_loader  # noqa: E402

    xr.set_device_type("TT")
    check_transformers_version()

    n = xr.global_runtime_device_count()
    if n != 32:
        raise SystemExit(f"prefill_check expects 32 TT devices, got {n}.")

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

    model_loader = create_model_loader(ModelLoader, num_layers=1, variant=ModelVariant.GPT_OSS_20B)
    if model_loader is None:
        raise SystemExit("ModelLoader does not support num_layers.")

    model, tokenizer = setup_model_and_tokenizer(model_loader, ModelVariant.GPT_OSS_20B)

    def read_logits(o):
        return o.logits

    ia_cpu = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        isl,
        input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
    )
    prefill_len = int(ia_cpu["input_ids"].shape[1])
    model.eval()
    with torch.no_grad():
        cpu_out = model(
            input_ids=ia_cpu["input_ids"],
            past_key_values=ia_cpu["past_key_values"],
            cache_position=ia_cpu["cache_position"],
            use_cache=True,
        )
    cpu_last = read_logits(cpu_out)[:, -1, :].detach().float()

    device = torch_xla.device()
    model = model.to(device=device, dtype=torch.bfloat16)

    mesh = get_mesh(model_loader, _gpt_oss_galaxy_mesh_config_fn)
    shard_specs = _gpt_oss_galaxy_shard_spec_fn(model_loader, model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    ia_tt = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        isl,
        input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
    )
    ia_tt = transfer_to_device(ia_tt, device)
    mark_multichip_gpt_oss_activation_shardings(mesh, ia_tt)
    _batch_parallel_input_sharding_fn(mesh, ia_tt)

    if hasattr(model, "lm_head") and model.lm_head is not None:
        hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
        model.lm_head.register_forward_hook(hook)

    wd = model_loader.get_weight_dtype_config_path()
    if wd:
        applied = apply_weight_dtype_overrides(model, wd)
        logger.info(f"[prefill_check] weight dtype overrides: {len(applied)} from {wd}")

    export_name = build_xla_export_name(
        model_name="gpt_oss_1layer_galaxy_prefill_check",
        num_layers=1,
        batch_size=batch_size,
        input_sequence_length=isl,
    )
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": optimization_level,
            "enable_trace": False,
            "export_path": MODULE_EXPORT_PATH,
            "export_model_name": export_name,
            "ttnn_perf_metrics_enabled": False,
            "experimental_weight_dtype": "bfp_bf8",
            "experimental_enable_permute_matmul_fusion": False,
        }
    )

    torch._dynamo.reset()
    compiled = torch.compile(model, backend="tt")

    prefill_ids = ia_cpu["input_ids"].to(device=device)
    xs.mark_sharding(prefill_ids, mesh, ("batch", None))
    cache_prefill = torch.arange(0, prefill_len, device=device, dtype=torch.long)

    with torch.no_grad():
        tt_out = compiled(
            input_ids=prefill_ids,
            past_key_values=ia_tt["past_key_values"],
            cache_position=cache_prefill,
            use_cache=True,
        )
    xm.mark_step()
    tt_last = read_logits(tt_out)[:, -1, :].detach().float().cpu()

    debug_kv_report(
        ia_tt["past_key_values"],
        "prefill_check_after_tt_prefill_only",
        flush_xla=True,
    )

    pccs = [compute_pcc_flat_pair(cpu_last[b], tt_last[b]) for b in range(cpu_last.shape[0])]
    pcc_min = min(pccs)
    tmax = float(tt_last.abs().max().item())
    all_zero = bool((tt_last == 0).all().item())

    print(
        f"[prefill_check] prefill_len={prefill_len} last-position logits "
        f"CPU vs TT: shape={tuple(tt_last.shape)} tt_max_abs={tmax:.6g} tt_all_zero={all_zero} "
        f"pcc_min_over_batch={pcc_min:.6f}"
    )
    print(
        "[prefill_check] Interpretation: high PCC here but bad `baseline` decode → focus on "
        "decode step / KV transfer / cache_position. Low PCC here → TT prefill stack differs "
        "from CPU before any decode. Next: `kv_tt_prefill` then `bisect` scopes."
    )


def _run_kv_tt_prefill(*, batch_size: int, isl: int, optimization_level: int) -> None:
    """TT prefill fills StaticCache on device, then TT decode; compare KV and logits to CPU reference."""

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from loguru import logger
    from test_llms import (
        _batch_parallel_input_sharding_fn,
        _gpt_oss_galaxy_mesh_config_fn,
        _gpt_oss_galaxy_shard_spec_fn,
    )
    from tt_torch.sharding import sharding_constraint_hook
    from tt_torch.weight_dtype import apply_weight_dtype_overrides

    from benchmarks.llm_benchmark import (  # noqa: E402
        DEFAULT_INPUT_PROMPT,
        MODULE_EXPORT_PATH,
        _static_cache_kv_snapshot,
        build_cpu_decode_start_after_prefill,
        check_transformers_version,
        construct_inputs,
        get_mesh,
        mark_multichip_gpt_oss_activation_shardings,
        setup_model_and_tokenizer,
        transfer_to_device,
    )
    from llm_utils.gpt_oss_kv_debug import debug_kv_report  # noqa: E402
    from utils import build_xla_export_name, compute_pcc_flat_pair, create_model_loader  # noqa: E402

    xr.set_device_type("TT")
    check_transformers_version()

    n = xr.global_runtime_device_count()
    if n != 32:
        raise SystemExit(f"kv_tt_prefill expects 32 TT devices, got {n}.")

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

    model_loader = create_model_loader(ModelLoader, num_layers=1, variant=ModelVariant.GPT_OSS_20B)
    if model_loader is None:
        raise SystemExit("ModelLoader does not support num_layers.")

    model, tokenizer = setup_model_and_tokenizer(model_loader, ModelVariant.GPT_OSS_20B)

    def read_logits(o):
        return o.logits

    input_args_cpu, golden_logits, prefill_len = build_cpu_decode_start_after_prefill(
        model=model,
        tokenizer=tokenizer,
        model_config=model.config,
        batch_size=batch_size,
        max_cache_len=isl,
        custom_input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
        read_logits_fn=read_logits,
        accuracy_testing=False,
        token_accuracy=None,
        capture_cpu_decode_logits_for_pcc=True,
    )
    kv_ref = _static_cache_kv_snapshot(input_args_cpu["past_key_values"])

    ia_align = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        isl,
        input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
    )
    prefill_ids_cpu = ia_align["input_ids"]
    if int(prefill_ids_cpu.shape[1]) != int(prefill_len):
        raise SystemExit(
            f"Tokenized prefill length mismatch: construct_inputs got {prefill_ids_cpu.shape[1]} "
            f"vs build_cpu_decode_start_after_prefill prefill_len={prefill_len}"
        )

    device = torch_xla.device()
    model = model.to(device=device, dtype=torch.bfloat16)

    mesh = get_mesh(model_loader, _gpt_oss_galaxy_mesh_config_fn)
    shard_specs = _gpt_oss_galaxy_shard_spec_fn(model_loader, model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    input_args_tt = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        isl,
        input_prompt=DEFAULT_INPUT_PROMPT,
        input_prompt_tokens=None,
    )
    input_args_tt = transfer_to_device(input_args_tt, device)
    mark_multichip_gpt_oss_activation_shardings(mesh, input_args_tt)
    _batch_parallel_input_sharding_fn(mesh, input_args_tt)

    if hasattr(model, "lm_head") and model.lm_head is not None:
        hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
        model.lm_head.register_forward_hook(hook)

    wd = model_loader.get_weight_dtype_config_path()
    if wd:
        applied = apply_weight_dtype_overrides(model, wd)
        logger.info(f"[kv_tt_prefill] weight dtype overrides: {len(applied)} from {wd}")

    export_name = build_xla_export_name(
        model_name="gpt_oss_1layer_galaxy_kv_tt_prefill",
        num_layers=1,
        batch_size=batch_size,
        input_sequence_length=isl,
    )
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": optimization_level,
            "enable_trace": False,
            "export_path": MODULE_EXPORT_PATH,
            "export_model_name": export_name,
            "ttnn_perf_metrics_enabled": False,
            "experimental_weight_dtype": "bfp_bf8",
            "experimental_enable_permute_matmul_fusion": False,
        }
    )

    torch._dynamo.reset()
    compiled = torch.compile(model, backend="tt")

    prefill_ids = prefill_ids_cpu.to(device=device)
    xs.mark_sharding(prefill_ids, mesh, ("batch", None))
    cache_prefill = torch.arange(0, prefill_len, device=device, dtype=torch.long)

    with torch.no_grad():
        compiled(
            input_ids=prefill_ids,
            past_key_values=input_args_tt["past_key_values"],
            cache_position=cache_prefill,
            use_cache=True,
        )
    xm.mark_step()

    debug_kv_report(
        input_args_tt["past_key_values"],
        "kv_tt_prefill_after_tt_prefill_before_decode",
        flush_xla=True,
    )

    k0_ref, v0_ref = kv_ref[0]
    layer0 = input_args_tt["past_key_values"].layers[0]
    k0_tt = layer0.keys.detach().cpu()
    v0_tt = layer0.values.detach().cpu()
    pcc_k0 = compute_pcc_flat_pair(k0_ref.flatten().float(), k0_tt.flatten().float())
    pcc_v0 = compute_pcc_flat_pair(v0_ref.flatten().float(), v0_tt.flatten().float())
    max_abs_k0 = float((k0_ref - k0_tt).abs().max().item())
    max_abs_v0 = float((v0_ref - v0_tt).abs().max().item())

    first_tok = input_args_cpu["input_ids"].to(device=device)
    xs.mark_sharding(first_tok, mesh, ("batch", None))
    cache_decode = torch.tensor([prefill_len], device=device, dtype=torch.long)

    with torch.no_grad():
        out_dec = compiled(
            input_ids=first_tok,
            past_key_values=input_args_tt["past_key_values"],
            cache_position=cache_decode,
            use_cache=True,
        )
    xm.mark_step()
    tt_logits = out_dec.logits.detach().cpu()

    g = golden_logits.float()
    t = tt_logits.float()
    pccs = [compute_pcc_flat_pair(g[b, -1], t[b, -1]) for b in range(g.shape[0])]
    pcc_min = min(pccs)
    tmax = float(t.abs().max().item())
    all_zero = bool((t == 0).all().item())

    print(
        f"[kv_tt_prefill] prefill_len={prefill_len} layer0 KV vs CPU-filled cache: "
        f"pcc_keys={pcc_k0:.6f} pcc_values={pcc_v0:.6f} max_abs_k={max_abs_k0:.6g} max_abs_v={max_abs_v0:.6g}"
    )
    print(
        f"[kv_tt_prefill] decode logits (TT prefill KV): max_abs={tmax:.6g} all_zero={all_zero} "
        f"pcc_min_last_token={pcc_min:.6f}"
    )
    print(
        "[kv_tt_prefill] Interpretation: compare to `baseline` (CPU prefill + transfer KV). "
        "Good TT prefill KV PCC but bad baseline decode → suspect KV transfer/sharding. "
        "Bad TT prefill KV PCC → on-device attention/cache write path. "
        "Good decode PCC here but bad baseline → strongly implicates transferred KV."
    )


def _run_bisect_forwarded(passthrough: list[str]) -> None:
    target = _BENCH_ROOT / "scripts" / "bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py"
    if not target.is_file():
        raise SystemExit(f"Missing {target}")
    cmd = [sys.executable, str(target), *passthrough]
    print(f"[bisect] exec: {' '.join(cmd)} (cwd={_BENCH_ROOT})")
    raise SystemExit(subprocess.call(cmd, cwd=str(_BENCH_ROOT)))


def _run_bisect_prefill_forwarded(passthrough: list[str]) -> None:
    target = _BENCH_ROOT / "scripts" / "bisect_gpt_oss_layer_cpu_attn_tt_moe_prefill.py"
    if not target.is_file():
        raise SystemExit(f"Missing {target}")
    cmd = [sys.executable, str(target), *passthrough]
    print(f"[bisect_prefill] exec: {' '.join(cmd)} (cwd={_BENCH_ROOT})")
    raise SystemExit(subprocess.call(cmd, cwd=str(_BENCH_ROOT)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-OSS Galaxy 1L baseline (full TT vs CPU) or bisect subprocess.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Use  `python ... runbook`  for device-safety + profiling; "
            "`python ... isolate`  for GPT-OSS subsystem isolation playbook."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rb = sub.add_parser("runbook", help="Print device safety + layer profiling how-to.")
    sub.add_parser(
        "isolate",
        help="Print subsystem map + ordered checks to localize CPU vs TT mismatch.",
    )
    p_bl = sub.add_parser("baseline", help="Full 1-layer model TT decode vs CPU (Galaxy sharding).")
    p_bl.add_argument("--batch-size", type=int, default=64)
    p_bl.add_argument("--isl", type=int, default=128)
    p_bl.add_argument("--optimization-level", type=int, default=1)

    p_kv = sub.add_parser(
        "kv_tt_prefill",
        help="TT prefill + TT decode vs CPU golden; layer-0 KV PCC vs CPU-filled cache.",
    )
    p_kv.add_argument("--batch-size", type=int, default=64)
    p_kv.add_argument("--isl", type=int, default=128)
    p_kv.add_argument("--optimization-level", type=int, default=1)

    p_pf = sub.add_parser(
        "prefill_check",
        help="CPU vs TT prefill only: last-position logits PCC (no decode step).",
    )
    p_pf.add_argument("--batch-size", type=int, default=64)
    p_pf.add_argument("--isl", type=int, default=128)
    p_pf.add_argument("--optimization-level", type=int, default=1)

    p_bpf = sub.add_parser(
        "bisect_prefill",
        help="Forward argv to bisect_gpt_oss_layer_cpu_attn_tt_moe_prefill.py (prefill hybrid).",
    )
    p_bpf.add_argument(
        "bisect_prefill_args",
        nargs=argparse.REMAINDER,
        help="Pass e.g. -- --layer 0 --tt-scope mlp",
    )

    p_bi = sub.add_parser(
        "bisect",
        help="Forward argv to bisect_gpt_oss_layer_cpu_attn_tt_moe_decode.py (decode hybrid).",
    )
    p_bi.add_argument(
        "bisect_args",
        nargs=argparse.REMAINDER,
        help="Pass e.g. -- --layer 0 --tt-scope mlp",
    )

    args = parser.parse_args()
    if args.cmd == "runbook":
        _print_runbook()
        return
    if args.cmd == "isolate":
        _print_isolation_playbook()
        return
    if args.cmd == "baseline":
        _run_baseline(
            batch_size=args.batch_size,
            isl=args.isl,
            optimization_level=args.optimization_level,
        )
        return
    if args.cmd == "kv_tt_prefill":
        _run_kv_tt_prefill(
            batch_size=args.batch_size,
            isl=args.isl,
            optimization_level=args.optimization_level,
        )
        return
    if args.cmd == "prefill_check":
        _run_prefill_check(
            batch_size=args.batch_size,
            isl=args.isl,
            optimization_level=args.optimization_level,
        )
        return
    if args.cmd == "bisect_prefill":
        extra = args.bisect_prefill_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        _run_bisect_prefill_forwarded(extra)
        return
    if args.cmd == "bisect":
        extra = args.bisect_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        _run_bisect_forwarded(extra)
        return


if __name__ == "__main__":
    main()
