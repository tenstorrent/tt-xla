# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-vs-host PCC correctness driver for LLMs (NOT a performance benchmark).

`run_llm_pcc_e2e` runs the same model end-to-end on the host (CPU, bf16
reference) and on the Tenstorrent device (torch-xla "tt" backend, with the same
mesh/sharding the perf path uses), then compares the **prefill** and
**first-decode** output logits via PCC and asserts they meet a threshold.

Unlike `benchmark_llm_torch_xla`, this does NO warmup pass, NO timed generation,
NO tokens/sec or TTFT, and emits no benchmark result — it exists purely to
validate numerical correctness (device output ≈ host output). It reuses the
benchmark's model-setup/sharding/decode helpers verbatim so the device execution
path is identical to the perf path.
"""

import os

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.llm_benchmark import (
    MODULE_EXPORT_PATH,
    _shard_kv_cache,
    construct_inputs,
    get_mesh,
    setup_model_and_tokenizer,
    transfer_to_device,
)
from llm_utils import generate_and_benchmark
from llm_utils.decode_utils import LLMSamplingWrapper
from loguru import logger
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from utils import build_xla_export_name, compute_pcc, compute_rel_l2


def _default_read_logits_fn(output):
    return output.logits


def run_llm_pcc_e2e(
    model_loader,
    model_variant,
    display_name,
    *,
    optimization_level,
    batch_size,
    input_sequence_length,
    mesh_config_fn,
    shard_spec_fn,
    required_pcc,
    experimental_weight_dtype="",
    experimental_kv_cache_dtype=None,
    experimental_enable_permute_matmul_fusion=False,
    read_logits_fn=_default_read_logits_fn,
    input_output_sharding_spec=None,
    kv_cache_sharding_spec=None,
    experts_implementation=None,
    weight_dtype_overrides=None,
    fp32_dest_acc_en=None,
    use_mla_cache=False,
    decode_only=False,
) -> dict:
    """Run the model on host and device and assert prefill/decode logit PCC.

    Returns a dict with the measured ``prefill_pcc`` / ``decode_pcc`` (and their
    rel-L2). Raises AssertionError if either PCC is below ``required_pcc``.
    """
    xr.set_device_type("TT")

    # Enable SPMD sharding for the multi-chip (mesh) path, exactly like the
    # perf benchmark does.
    if mesh_config_fn is not None and shard_spec_fn is not None:
        is_multichip = xr.global_runtime_device_count() > 1
        if is_multichip:
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            xr.use_spmd()
    else:
        is_multichip = False

    max_cache_len = input_sequence_length
    device = torch_xla.device()

    model, tokenizer = setup_model_and_tokenizer(
        model_loader,
        model_variant,
        experts_implementation=experts_implementation,
    )

    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        use_mla_cache=use_mla_cache,
    )

    # ------------------------------------------------------------------
    # HOST (CPU) reference: prefill logits + first-decode logits.
    # tt_* experts/attention backends auto-fall-back to HF builtins for CPU
    # tensors, so no backend swap is needed here.
    # ------------------------------------------------------------------
    cpu_wrapper = LLMSamplingWrapper(model, read_logits_fn, return_logits=True)
    cpu_wrapper.eval()

    # Iter 0: prefill. Always run — besides providing the prefill PCC reference,
    # it populates the KV cache (decode_only_cache) that the decode step / the
    # decode-only device path reads from. Advances input_args to the post-prefill
    # decode state (input_ids = first predicted token).
    cpu_prefill_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )

    # Snapshot first-decode inputs before the CPU decode step advances them.
    first_decode_input_ids = input_args["input_ids"].clone()
    decode_only_cache_position = input_args["cache_position"].clone()
    decode_only_cache = input_args["past_key_values"]

    # Iter 1: first decode. Provides the PCC reference for the device decode.
    cpu_decode_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )

    # Layout: [prefill_logits, first_decode_logits]. [-1] is always the
    # first-decode step (used as the decode PCC reference in both modes).
    cpu_output_logits = cpu_prefill_logits + cpu_decode_logits

    # ------------------------------------------------------------------
    # DEVICE setup: transfer + shard the model identically to the perf path.
    # ------------------------------------------------------------------
    model = model.to(device, dtype=torch.bfloat16)

    mesh = None
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, model)
        mesh = get_mesh(model_loader, mesh_config_fn)
        # Register the mesh globally so mesh-aware experts backends (tt_moe,
        # tt_moe_fused) can read it via torch_xla get_global_mesh().
        xs.set_global_mesh(mesh)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # All-gather lm_head logits so the host-side comparison sees full logits.
        if hasattr(model, "lm_head") and model.lm_head is not None:
            hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
            model.lm_head.register_forward_hook(hook)

    export_model_name = build_xla_export_name(
        model_name=display_name + "_pcc",
        num_layers=getattr(model_loader, "num_layers", None),
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
    )
    options = {
        "optimization_level": optimization_level,
        # Correctness test: trace disabled, no perf metrics collection.
        "enable_trace": False,
        "export_path": MODULE_EXPORT_PATH,
        **({"dry_run": True} if os.environ.get("TTXLA_DRY_RUN") else {}),
        **({"export_tensors": True} if os.environ.get("TTXLA_EXPORT_TENSORS") else {}),
        "export_model_name": export_model_name,
        "experimental_weight_dtype": experimental_weight_dtype,
        "experimental_enable_permute_matmul_fusion": experimental_enable_permute_matmul_fusion,
    }
    if fp32_dest_acc_en is not None:
        options["fp32_dest_acc_en"] = fp32_dest_acc_en
    if experimental_kv_cache_dtype is not None:
        options["experimental-kv-cache-dtype"] = experimental_kv_cache_dtype
    torch_xla.set_custom_compile_options(options)

    if weight_dtype_overrides:
        apply_weight_dtype_overrides(model, weight_dtype_overrides)
    else:
        weight_dtype_config = model_loader.get_weight_dtype_config_path()
        if weight_dtype_config:
            apply_weight_dtype_overrides(model, weight_dtype_config)

    # ------------------------------------------------------------------
    # DEVICE run: prefill logits + first-decode logits (no warmup, no timing).
    # ------------------------------------------------------------------
    logits_wrapper = LLMSamplingWrapper(
        model,
        read_logits_fn,
        return_logits=True,
        mesh=mesh,
        output_sharding_spec=input_output_sharding_spec,
    )
    logits_wrapper.eval()
    compiled_logits = torch.compile(logits_wrapper, backend="tt")

    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        past_key_values=decode_only_cache if decode_only else None,
        use_mla_cache=use_mla_cache,
    )
    if decode_only:
        input_args["input_ids"] = first_decode_input_ids.clone()
        input_args["cache_position"] = decode_only_cache_position.clone()

    input_args = transfer_to_device(input_args, device)
    if is_multichip:
        _shard_kv_cache(input_args["past_key_values"], mesh, kv_cache_sharding_spec)
    if input_output_sharding_spec:
        xs.mark_sharding(input_args["input_ids"], mesh, input_output_sharding_spec)

    device_prefill_logits = []
    if not decode_only:
        logger.info("PCC test: running device prefill...")
        device_prefill_logits, _ = generate_and_benchmark(
            compiled_logits,
            input_args,
            device,
            1,
            verbose=False,
            collect_logits=True,
        )

        # Keep the first-decode input aligned with the CPU reference: if the
        # device prefill argmax diverged from CPU's, feed CPU's token so the
        # decode PCC compares like-for-like (a poor prefill PCC otherwise
        # compounds into the decode PCC — see tt-xla #4614).
        device_prefill_output_ids = input_args["input_ids"].to("cpu")
        if not torch.equal(device_prefill_output_ids, first_decode_input_ids.cpu()):
            logger.warning(
                "Device prefill produced different tokens than CPU prefill; "
                "using CPU prefill output as the decode PCC reference input."
            )
            input_args["input_ids"] = first_decode_input_ids.to(device)
            if input_output_sharding_spec:
                xs.mark_sharding(
                    input_args["input_ids"], mesh, input_output_sharding_spec
                )

    logger.info("PCC test: running device first-decode...")
    device_decode_logits, _ = generate_and_benchmark(
        compiled_logits,
        input_args,
        device,
        1,
        verbose=False,
        collect_logits=True,
    )

    output_logits = (
        device_decode_logits
        if decode_only
        else device_prefill_logits + device_decode_logits
    )

    # ------------------------------------------------------------------
    # Compare prefill + decode logit PCC (device vs host) and assert.
    # ------------------------------------------------------------------
    results = {}
    if not decode_only:
        prefill_pcc = compute_pcc(output_logits[0][0], cpu_output_logits[0][0])
        prefill_rel_l2 = compute_rel_l2(cpu_output_logits[0][0], output_logits[0][0])
        results["prefill_pcc"] = prefill_pcc
        results["prefill_rel_l2"] = prefill_rel_l2
        logger.info(
            f"Prefill PCC = {prefill_pcc:.6f} (required {required_pcc}), "
            f"rel_l2 = {prefill_rel_l2:.6e}"
        )

    # cpu_output_logits[-1] / output_logits[-1] is the first-decode step in both
    # the prefill+decode ([1]) and decode-only ([0]) layouts.
    decode_pcc = compute_pcc(output_logits[-1][0], cpu_output_logits[-1][0])
    decode_rel_l2 = compute_rel_l2(cpu_output_logits[-1][0], output_logits[-1][0])
    results["decode_pcc"] = decode_pcc
    results["decode_rel_l2"] = decode_rel_l2
    logger.info(
        f"Decode PCC = {decode_pcc:.6f} (required {required_pcc}), "
        f"rel_l2 = {decode_rel_l2:.6e}"
    )

    if not decode_only:
        assert (
            results["prefill_pcc"] >= required_pcc
        ), f"Prefill PCC failed: {results['prefill_pcc']:.6f} < {required_pcc}"
    assert (
        decode_pcc >= required_pcc
    ), f"Decode PCC failed: {decode_pcc:.6f} < {required_pcc}"

    logger.info(
        f"[{display_name}] device-vs-host PCC PASSED "
        f"(prefill={results.get('prefill_pcc', 'n/a')}, decode={decode_pcc:.6f})"
    )
    return results
