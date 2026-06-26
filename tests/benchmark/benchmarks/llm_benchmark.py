# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import socket
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import tracy
from fusion_check import check_fusions
from harness import MODULE_EXPORT_PATH, init_tt_runtime
from harness import build_compile_options as _build_compile_options
from infra import MLACache, MLAStaticLayer
from llm_utils import (
    generate_and_benchmark,
    init_accuracy_testing,
    init_indexer_cache,
    init_mla_cache,
    init_static_cache,
)
from llm_utils.decode_utils import LLMSamplingWrapper
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from utils import (
    build_xla_export_name,
    compute_pcc,
    compute_rel_l2,
    create_benchmark_result,
    get_benchmark_metadata,
    get_xla_device_arch,
    print_benchmark_results,
)

init_tt_runtime()

MIN_STEPS = 16
DEFAULT_EXPERTS_IMPLEMENTATION = "batched_mm"

# Default input prompt
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)


@dataclass
class PccMode:
    """PCC-only iteration mode parsed from the TT_PCC_MODE env var.

    TT_PCC_MODE = "prefill" | "decode" | "both" skips warmup and the timed perf
    loop and runs a single PCC iteration. Both prefill and decode PCC are always
    printed; only the selected mode's PCC(s) are asserted. For "decode"/"both"
    the decode PCC is isolated by reseeding the device KV cache from the
    CPU-golden post-prefill cache so it does not inherit the device prefill's
    numerical error.
    """

    pcc_only: bool
    assert_prefill: bool
    assert_decode: bool
    isolated: bool

    @classmethod
    def from_env(cls) -> "PccMode":
        pcc_mode = os.environ.get("TT_PCC_MODE", "").strip().lower()
        pcc_only = pcc_mode in ("prefill", "decode", "both")
        return cls(
            pcc_only=pcc_only,
            assert_prefill=(not pcc_only) or (pcc_mode in ("both", "prefill")),
            assert_decode=(not pcc_only) or (pcc_mode in ("both", "decode")),
            isolated=pcc_mode in ("decode", "both"),
        )


@dataclass
class CpuReference:
    """CPU golden outputs and post-prefill decode seeds used as PCC baseline."""

    output_logits: list
    first_decode_input_ids: torch.Tensor
    decode_cache_position: torch.Tensor
    decode_cache: object
    decode_cumulative_lengths: list


@dataclass
class PerfSummary:
    """Throughput metrics derived from per-iteration timings."""

    ttft_ms: float
    decode_total_time: float
    decode_total_tokens: int
    tokens_per_second: float


@dataclass
class CompileConfig:
    """tt-mlir / torch-xla compilation knobs."""

    optimization_level: int
    trace_enabled: bool
    experimental_weight_dtype: str
    experimental_enable_permute_matmul_fusion: bool
    fp32_dest_acc_en: Optional[bool] = None
    experimental_kv_cache_dtype: Optional[str] = None
    enable_create_d2m_subgraphs: bool = False


@dataclass
class ShardingConfig:
    """Multi-chip mesh and sharding specs (all None for single-chip)."""

    mesh_config_fn: Optional[Callable] = None
    shard_spec_fn: Optional[Callable] = None
    input_output_sharding_spec: Optional[tuple] = None
    kv_cache_sharding_spec: Optional[tuple] = None


@dataclass
class AccuracyConfig:
    """Token-accuracy testing settings."""

    enabled: bool = False
    model_name_for_accuracy: Optional[str] = None
    hf_model_name_for_accuracy: Optional[str] = None


def setup_model_and_tokenizer(
    model_loader, model_variant, experts_implementation: Optional[str] = None
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_loader: Loader of the HuggingFace model.
        model_variant: Specific variant of the model.
        expert_implementation: Expert implementation type

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_loader.get_model_info(variant=model_variant).name}...")

    model_kwargs = {}
    if experts_implementation is not None:
        model_kwargs["experts_implementation"] = experts_implementation
    model = model_loader.load_model(dtype_override=torch.bfloat16, **model_kwargs)
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    # Use static dense experts forward to avoid graph breaks from data-dependent
    # loops in the original experts and _grouped_mm CPU crashes.
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = (
            experts_implementation or DEFAULT_EXPERTS_IMPLEMENTATION
        )
    model = model.eval()
    tokenizer = model_loader.tokenizer

    return model, tokenizer


def construct_inputs(
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
    past_key_values: Optional[Union[StaticCache, MLACache]] = None,
    input_prompt: str = None,
    input_prompt_tokens: Optional[torch.Tensor] = None,
    use_mla_cache: bool = False,
) -> dict:
    """
    Construct inputs including static cache.

    Args:
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length
        input_prompt: Input text prompt (optional, defaults to DEFAULT_INPUT_PROMPT)
        input_prompt_tokens: Pre-tokenized input prompt (optional, overrides input_prompt)
        use_mla_cache: Whether to use MLA cache

    Returns:
        Dictionary containing input_ids, past_key_values, cache_position, and use_cache
    """
    if input_prompt_tokens is not None:
        if input_prompt_tokens.ndim != 1:
            raise ValueError(
                f"input_prompt_tokens must be 1D token IDs, got shape {tuple(input_prompt_tokens.shape)}"
            )
        if input_prompt_tokens.shape[0] > max_cache_len:
            input_prompt_tokens = input_prompt_tokens[:max_cache_len]

        input_ids = input_prompt_tokens.unsqueeze(0).expand(batch_size, -1).contiguous()
        inputs = {"input_ids": input_ids}
    else:
        if input_prompt is None:
            input_prompt = DEFAULT_INPUT_PROMPT
        input_prompt = [input_prompt] * batch_size

        # TODO: Only works on same length inputs for now
        prompt_len = len(input_prompt[0])
        assert all(
            len(prompt) == prompt_len for prompt in input_prompt
        ), "All input prompts must have the same length"

        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_cache_len,
            truncation=True,
        )

    if past_key_values is None:
        # Static cache should be initialized on CPU and separately transferred to device
        # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
        if use_mla_cache:
            static_cache = init_mla_cache(
                config=model_config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                device="cpu",
                dtype=torch.bfloat16,
            )
        else:
            static_cache = init_static_cache(
                config=model_config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                device="cpu",
                dtype=torch.bfloat16,
            )
    else:
        static_cache = past_key_values
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    cache_position: torch.Tensor = torch.arange(0, input_ids.shape[1])

    input_args = {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    return input_args


def get_mesh(model_loader, mesh_config_fn):
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = mesh_config_fn(model_loader, num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, mesh_name)


def transfer_to_device(input_args: dict, device: torch.device) -> dict:
    """
    Transfer inputs to device.

    Args:
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        input_args on device
    """
    for layer in input_args["past_key_values"].layers:
        if isinstance(layer, MLAStaticLayer):
            layer.compressed_kv = layer.compressed_kv.to(device)
            layer.k_pe = layer.k_pe.to(device)
            layer.keys = layer.compressed_kv
            layer.values = layer.k_pe
            if not torch.compiler.is_compiling():
                torch._dynamo.mark_static_address(layer.compressed_kv)
                torch._dynamo.mark_static_address(layer.k_pe)
        else:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
            # CL must be on device: StaticLayer.update() does
            # torch.arange(kv_length, device=self.device) + self.cumulative_length,
            # which fails if CL is on CPU and self.device is XLA.
            # Zero before moving so the fresh cache starts at position 0.
            layer.cumulative_length.zero_()
            layer.cumulative_length = layer.cumulative_length.to(device)
            layer.device = device
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    return input_args


def _restore_cumulative_length(past_key_values, cumulative_lengths):
    """Restore each layer's cumulative_length from a post-prefill snapshot.

    transfer_to_device() zeroes cumulative_length (it assumes a fresh cache that
    prefill will fill). When a device decode is seeded from a pre-filled CPU
    cache (decode-only and isolated-decode PCC) the length must instead be the
    post-prefill value, or the decode reads an empty KV window at opt_level > 0.
    """
    for layer, cumulative_length in zip(past_key_values.layers, cumulative_lengths):
        if cumulative_length is not None and hasattr(layer, "cumulative_length"):
            layer.cumulative_length.copy_(
                cumulative_length.to(layer.cumulative_length.device)
            )


def _shard_kv_cache(past_key_values, mesh, kv_cache_sharding_spec):
    kv_spec = kv_cache_sharding_spec
    for layer in past_key_values.layers:
        if isinstance(layer, MLAStaticLayer):
            if kv_spec is None:
                kv_spec = ("batch", None, None, None)
            xs.mark_sharding(layer.compressed_kv, mesh, kv_spec)
            xs.mark_sharding(layer.k_pe, mesh, kv_spec)
        else:
            if kv_spec is None:
                kv_spec = (None, "model", None, None)
            xs.mark_sharding(layer.keys, mesh, kv_spec)
            xs.mark_sharding(layer.values, mesh, kv_spec)


class GenerationSession:
    """Owns the input + KV-cache lifecycle for one model on one device.

    Centralizes input construction, device transfer, sharding, and the
    post-prefill cache bookkeeping (cumulative_length reset/restore) that the
    warmup, perf, and PCC phases would otherwise each re-implement. The session
    holds the static context (tokenizer, mesh, sharding specs, prompt) so the
    phases only express what differs between them.
    """

    def __init__(
        self,
        *,
        tokenizer,
        model_config,
        batch_size: int,
        max_cache_len: int,
        device: torch.device,
        mesh=None,
        use_mla_cache: bool = False,
        kv_cache_sharding_spec=None,
        input_output_sharding_spec=None,
        custom_input_prompt=None,
        input_prompt_tokens=None,
    ):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = device
        self.mesh = mesh
        self.use_mla_cache = use_mla_cache
        self.kv_cache_sharding_spec = kv_cache_sharding_spec
        self.input_output_sharding_spec = input_output_sharding_spec
        self.custom_input_prompt = custom_input_prompt
        self.input_prompt_tokens = input_prompt_tokens

    @property
    def is_multichip(self) -> bool:
        return self.mesh is not None

    def build_inputs(self, *, past_key_values=None) -> dict:
        """Construct fresh CPU input_args (optionally reusing an existing cache)."""
        return construct_inputs(
            self.tokenizer,
            self.model_config,
            self.batch_size,
            self.max_cache_len,
            past_key_values=past_key_values,
            input_prompt=self.custom_input_prompt,
            input_prompt_tokens=self.input_prompt_tokens,
            use_mla_cache=self.use_mla_cache,
        )

    @staticmethod
    def seed_decode(input_args: dict, *, input_ids, cache_position) -> dict:
        """Reset input_args to a single-token post-prefill decode state."""
        input_args["input_ids"] = input_ids.clone()
        input_args["cache_position"] = cache_position.clone()
        return input_args

    @staticmethod
    def reset_cumulative_length(cache) -> None:
        """Zero each layer's cumulative_length on CPU (fresh-cache invariant)."""
        for layer in cache.layers:
            if hasattr(layer, "cumulative_length"):
                if layer.cumulative_length.device.type != "cpu":
                    layer.cumulative_length = layer.cumulative_length.cpu()
                layer.cumulative_length.zero_()

    def place_on_device(
        self,
        input_args: dict,
        *,
        shard_kv: bool,
        shard_input_ids: bool,
        cumulative_lengths=None,
    ) -> dict:
        """Transfer inputs to the device and apply the sharding/seed ritual.

        Consolidates the transfer + KV-cache shard + input_ids shard +
        cumulative-length restore sequence shared by the warmup, perf, and PCC
        phases. shard_kv / shard_input_ids stay explicit because the phases
        differ in whether a cache/input is freshly placed (and thus needs
        re-sharding) versus reused from a prior placement.
        """
        input_args = transfer_to_device(input_args, self.device)
        if shard_kv:
            _shard_kv_cache(
                input_args["past_key_values"], self.mesh, self.kv_cache_sharding_spec
            )
        if shard_input_ids:
            xs.mark_sharding(
                input_args["input_ids"], self.mesh, self.input_output_sharding_spec
            )
        if cumulative_lengths is not None:
            _restore_cumulative_length(
                input_args["past_key_values"], cumulative_lengths
            )
        return input_args


def _validate_args(
    *,
    data_format,
    model_loader,
    loop_count,
    input_sequence_length,
    decode_only,
    accuracy_testing,
    task,
    enable_create_d2m_subgraphs,
    optimization_level,
):
    """Validate benchmark arguments, raising ValueError on unsupported configs."""
    # Enforce bfloat16 only
    if data_format != "bfloat16":
        raise ValueError(
            f"Only bfloat16 data format is supported for llm benchmark. Got: {data_format}. "
            "Please use -df bfloat16"
        )

    if not model_loader:
        raise ValueError("Model loader must be specified for benchmark. ")

    if loop_count != 1:
        raise ValueError(
            f"Loop count must be 1 for llm benchmark (not yet supported). Got: {loop_count}. "
            "Please use -lp 1"
        )

    if not input_sequence_length or input_sequence_length <= 0:
        raise ValueError(
            f"Input sequence length must be a positive integer for llm benchmark. Got: {input_sequence_length}. "
            "Please use -isl <length> (e.g., -isl 128)"
        )

    if decode_only and accuracy_testing:
        raise ValueError("--decode-only cannot be combined with --accuracy-testing")

    if task != "text-generation":
        raise ValueError(
            f"Only 'text-generation' task is supported for llm benchmark. Got: {task}. "
            "Please use -t text-generation"
        )

    if enable_create_d2m_subgraphs and optimization_level < 1:
        raise ValueError(
            f"optimization_level must be >= 1 when enable_create_d2m_subgraphs "
            f"is enabled, got optimization_level={optimization_level}"
        )


def compute_cpu_reference(model, read_logits_fn, input_args: dict) -> CpuReference:
    """Run CPU prefill + first decode to produce the PCC golden reference.

    Mutates input_args in place (advances it through prefill then decode).
    tt_* experts/attention backends auto-fall-back to HF builtins for CPU
    tensors, so no backend swap is needed here.
    """
    cpu_wrapper = LLMSamplingWrapper(model, read_logits_fn, return_logits=True)
    cpu_wrapper.eval()

    # Iter 0: prefill. After this, input_args holds the post-prefill decode
    # state (input_ids=next_token_0, cache_position=[prompt_len]).
    cpu_prefill_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )

    # Snapshot first-decode inputs before CPU iter 1 advances them.
    first_decode_input_ids = input_args["input_ids"].clone()
    decode_cache_position = input_args["cache_position"].clone()
    decode_cache = input_args["past_key_values"]
    # Post-prefill cumulative_length per layer, used to restore the
    # reseeded isolated-decode cache (transfer_to_device zeroes it).
    decode_cumulative_lengths = [
        (
            layer.cumulative_length.detach().clone()
            if hasattr(layer, "cumulative_length")
            and layer.cumulative_length is not None
            else None
        )
        for layer in decode_cache.layers
    ]

    # Iter 1: first decode. Provides the PCC reference for device first decode.
    cpu_decode_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )

    return CpuReference(
        output_logits=cpu_prefill_logits + cpu_decode_logits,
        first_decode_input_ids=first_decode_input_ids,
        decode_cache_position=decode_cache_position,
        decode_cache=decode_cache,
        decode_cumulative_lengths=decode_cumulative_lengths,
    )


def build_compile_options(
    compile_config: "CompileConfig",
    *,
    export_model_name,
    ttnn_perf_metrics_output_file,
) -> dict:
    """Assemble the torch-xla custom compile options dict from a CompileConfig.

    Thin adapter over :func:`harness.build_compile_options` that unpacks the
    LLM-specific :class:`CompileConfig`.
    """
    return _build_compile_options(
        optimization_level=compile_config.optimization_level,
        enable_trace=compile_config.trace_enabled,
        export_model_name=export_model_name,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        experimental_weight_dtype=compile_config.experimental_weight_dtype,
        experimental_enable_permute_matmul_fusion=(
            compile_config.experimental_enable_permute_matmul_fusion
        ),
        fp32_dest_acc_en=compile_config.fp32_dest_acc_en,
        experimental_kv_cache_dtype=compile_config.experimental_kv_cache_dtype,
        enable_create_d2m_subgraphs=compile_config.enable_create_d2m_subgraphs,
    )


def apply_weight_dtypes(model, model_loader, weight_dtype_overrides):
    """Apply per-tensor weight dtype overrides.

    An explicit dict takes priority; otherwise fall back to the model's
    weight_dtype_configs JSON (auto-discovery).
    """
    if weight_dtype_overrides:
        applied = apply_weight_dtype_overrides(model, weight_dtype_overrides)
        logger.info(f"Applied {len(applied)} weight dtype overrides from explicit dict")
        return

    weight_dtype_config = model_loader.get_weight_dtype_config_path()
    if weight_dtype_config:
        applied = apply_weight_dtype_overrides(model, weight_dtype_config)
        logger.info(
            f"Applied {len(applied)} weight dtype overrides from {weight_dtype_config}"
        )


def summarize_perf(iteration_times: list, decode_only: bool) -> PerfSummary:
    """Derive TTFT and decode throughput from per-iteration timings (ns)."""
    ttft_ns = iteration_times[0] if (not decode_only and iteration_times) else 0.0
    ttft_ms = ttft_ns / 1e6

    decode_iteration_times = iteration_times[1:]
    decode_total_time_ns = sum(decode_iteration_times)
    decode_total_time = decode_total_time_ns / 1e9

    decode_total_tokens = len(decode_iteration_times)
    tokens_per_second = (
        (decode_total_tokens / decode_total_time) if decode_total_time > 0 else 0.0
    )
    return PerfSummary(
        ttft_ms=ttft_ms,
        decode_total_time=decode_total_time,
        decode_total_tokens=decode_total_tokens,
        tokens_per_second=tokens_per_second,
    )


def evaluate_accuracy(output_logits: list, token_accuracy) -> tuple[float, list]:
    """Compute per-user TOP1/TOP5 token accuracy across the batch.

    Returns the primary evaluation score (TOP1 p5) and the list of custom
    measurement dicts to record.
    """
    # Derive predicted tokens per user.
    batch_size_for_accuracy = output_logits[0].shape[0]
    per_user_predictions = []
    for user_idx in range(batch_size_for_accuracy):
        user_tokens = [
            logits[:, -1, :].argmax(dim=-1)[user_idx].item() for logits in output_logits
        ]
        per_user_predictions.append(user_tokens)

    all_top1 = []
    all_top5 = []
    for user_tokens in per_user_predictions:
        user_top1, user_top5 = token_accuracy.compute_accuracy(user_tokens)
        all_top1.append(user_top1)
        all_top5.append(user_top5)

    all_top1 = np.array(all_top1)
    all_top5 = np.array(all_top5)

    # Use 5th-percentile (p5) as primary metric: "95% of users are at or above this"
    # This catches broken users that averaging would hide.
    top1_p5 = float(np.percentile(all_top1, 5))
    top5_p5 = float(np.percentile(all_top5, 5))

    num_users = len(all_top1)
    print(
        f"Token accuracy over {num_users} users:\n"
        f"  TOP1 — min={all_top1.min()*100:.2f}%  p5={top1_p5*100:.2f}%  "
        f"median={np.median(all_top1)*100:.2f}%  mean={all_top1.mean()*100:.2f}%  "
        f"max={all_top1.max()*100:.2f}%\n"
        f"  TOP5 — min={all_top5.min()*100:.2f}%  p5={top5_p5*100:.2f}%  "
        f"median={np.median(all_top5)*100:.2f}%  mean={all_top5.mean()*100:.2f}%  "
        f"max={all_top5.max()*100:.2f}%"
    )

    top1_mean = float(all_top1.mean())
    top5_mean = float(all_top5.mean())
    measurements = [
        {"measurement_name": "top1_accuracy_p5", "value": top1_p5 * 100},
        {"measurement_name": "top5_accuracy_p5", "value": top5_p5 * 100},
        {"measurement_name": "top1_accuracy_mean", "value": top1_mean * 100},
        {"measurement_name": "top5_accuracy_mean", "value": top5_mean * 100},
    ]
    # Use TOP1 p5 as primary score.
    return top1_p5, measurements


def evaluate_pcc(
    output_logits: list,
    cpu_output_logits: list,
    required_pcc,
    *,
    decode_only: bool,
    assert_prefill: bool,
    assert_decode: bool,
):
    """Compute, print, and (optionally) assert prefill/decode PCC vs CPU golden."""
    if decode_only:
        decode_pcc_value = compute_pcc(output_logits[0][0], cpu_output_logits[1][0])
        decode_rel_l2_value = compute_rel_l2(
            cpu_output_logits[1][0], output_logits[0][0]
        )
        print(
            "First decode PCC={:.6f}, rel_l2={:.6e}, Required={}".format(
                decode_pcc_value, decode_rel_l2_value, required_pcc
            )
        )
        if assert_decode:
            assert (
                decode_pcc_value >= required_pcc
            ), f"First decode PCC failed. PCC={decode_pcc_value:.6f}, Required={required_pcc}"
        return

    # Check PCC for prefill
    pcc_value = compute_pcc(output_logits[0][0], cpu_output_logits[0][0])
    rel_l2_value = compute_rel_l2(cpu_output_logits[0][0], output_logits[0][0])
    print(
        "Prefill PCC={:.6f}, rel_l2={:.6e}, Required={}".format(
            pcc_value, rel_l2_value, required_pcc
        )
    )
    # Check PCC for first decode token (when available).
    decode_pcc_value = None
    if len(output_logits) > 1 and len(cpu_output_logits) > 1:
        decode_pcc_value = compute_pcc(output_logits[1][0], cpu_output_logits[1][0])
        decode_rel_l2_value = compute_rel_l2(
            cpu_output_logits[1][0], output_logits[1][0]
        )
        print(
            "First decode PCC={:.6f}, rel_l2={:.6e}, Required={}".format(
                decode_pcc_value, decode_rel_l2_value, required_pcc
            )
        )
    if assert_prefill:
        assert (
            pcc_value >= required_pcc
        ), f"Prefill PCC failed. PCC={pcc_value:.6f}, Required={required_pcc}"
    if assert_decode:
        assert (
            decode_pcc_value is not None and decode_pcc_value >= required_pcc
        ), f"First decode PCC failed. PCC={decode_pcc_value}, Required={required_pcc}"


@dataclass
class DeviceRunContext:
    """Immutable context shared by the on-device perf and logits phases."""

    session: GenerationSession
    model: torch.nn.Module
    read_logits_fn: Callable
    decode_only: bool
    pcc: PccMode
    accuracy_testing: bool
    golden: Optional[CpuReference]
    ground_truth_tokens: Optional[torch.Tensor]
    tokenizer: object


def run_perf_phase(ctx: DeviceRunContext, *, initial_inputs: dict, max_output_tokens):
    """Warm up and run the timed decode loop; return per-iteration times (ns).

    Returns an empty list in PCC-only mode (the timed loop is skipped), but the
    perf inputs are still placed on device to mirror the full-run setup.
    """
    session = ctx.session
    iospec = session.input_output_sharding_spec

    # No logits returned to maximize performance and avoid device DRAM OOM.
    perf_wrapper = LLMSamplingWrapper(
        ctx.model,
        ctx.read_logits_fn,
        return_logits=False,
        mesh=session.mesh,
        output_sharding_spec=iospec,
    )
    perf_wrapper.eval()
    compiled_perf_model = torch.compile(perf_wrapper, backend="tt")

    # Warmup run (skip in decode-only mode and in pcc-only mode)
    current_inputs = initial_inputs
    if not ctx.decode_only and not ctx.pcc.pcc_only:
        warmup_inputs = session.build_inputs()
        warmup_inputs = session.place_on_device(
            warmup_inputs,
            shard_kv=session.is_multichip,
            shard_input_ids=session.is_multichip and bool(iospec),
        )
        print("Warming up...")
        warmup_tokens = min(MIN_STEPS, max_output_tokens)
        generate_and_benchmark(
            compiled_perf_model,
            warmup_inputs,
            session.device,
            warmup_tokens,
            verbose=False,
            collect_logits=False,
        )
        print("Warmup complete")
        tracy.signpost("warmup_complete")
        current_inputs = warmup_inputs

    # Reconstruct inputs for the perf benchmark run.
    existing_cache = (
        current_inputs["past_key_values"]
        if not ctx.decode_only
        else ctx.golden.decode_cache
    )
    session.reset_cumulative_length(existing_cache)

    perf_inputs = session.build_inputs(past_key_values=existing_cache)
    if ctx.decode_only:
        session.seed_decode(
            perf_inputs,
            input_ids=ctx.golden.first_decode_input_ids,
            cache_position=ctx.golden.decode_cache_position,
        )
    perf_inputs = session.place_on_device(
        perf_inputs,
        shard_kv=session.is_multichip and ctx.decode_only,
        shard_input_ids=bool(iospec),
        cumulative_lengths=(
            ctx.golden.decode_cumulative_lengths if ctx.decode_only else None
        ),
    )

    # Run perf benchmark (skipped in pcc-only mode for fast iteration)
    iteration_times = []
    if not ctx.pcc.pcc_only:
        print("\nStarting performance benchmark...")
        _, iteration_times = generate_and_benchmark(
            compiled_perf_model,
            perf_inputs,
            session.device,
            max_output_tokens,
            verbose=True,
            tokenizer=ctx.tokenizer,
            ground_truth_tokens=ctx.ground_truth_tokens,
            collect_logits=False,
        )
        print("\nPerformance benchmark complete")
    return iteration_times


def run_logits_phase(ctx: DeviceRunContext, *, max_output_tokens):
    """Run prefill + decode collecting logits; return per-step logits (on CPU)."""
    session = ctx.session
    iospec = session.input_output_sharding_spec

    # Return logits to calculate PCC/TOPK
    logits_wrapper = LLMSamplingWrapper(
        ctx.model,
        ctx.read_logits_fn,
        return_logits=True,
        mesh=session.mesh,
        output_sharding_spec=iospec,
    )
    logits_wrapper.eval()
    compiled_logits = torch.compile(logits_wrapper, backend="tt")

    logits_steps = (
        (1 if ctx.decode_only else 2) if ctx.pcc.pcc_only else max_output_tokens
    )

    # Reconstruct inputs for PCC/TOPK run
    input_args = session.build_inputs(
        past_key_values=ctx.golden.decode_cache if ctx.decode_only else None
    )
    if ctx.decode_only:
        session.seed_decode(
            input_args,
            input_ids=ctx.golden.first_decode_input_ids,
            cache_position=ctx.golden.decode_cache_position,
        )
    input_args = session.place_on_device(
        input_args,
        shard_kv=session.is_multichip,
        shard_input_ids=bool(iospec),
        cumulative_lengths=(
            ctx.golden.decode_cumulative_lengths if ctx.decode_only else None
        ),
    )

    print("\nStarting PCC/TOPK benchmark...")

    device_prefill_logits = []
    if not ctx.decode_only:
        device_prefill_logits, _ = generate_and_benchmark(
            compiled_logits,
            input_args,
            session.device,
            1,
            verbose=False,
            ground_truth_tokens=ctx.ground_truth_tokens,
            collect_logits=True,
        )

        device_prefill_output_ids = input_args["input_ids"].to("cpu")

        if ctx.pcc.isolated:
            # Rebuild decode inputs from the CPU-golden post-prefill KV cache.
            input_args = session.build_inputs(past_key_values=ctx.golden.decode_cache)
            session.seed_decode(
                input_args,
                input_ids=ctx.golden.first_decode_input_ids,
                cache_position=ctx.golden.decode_cache_position,
            )
            input_args = session.place_on_device(
                input_args,
                shard_kv=session.is_multichip,
                shard_input_ids=bool(iospec),
                cumulative_lengths=ctx.golden.decode_cumulative_lengths,
            )
        # Override device's first-decode input with CPU's prefill output when they
        # diverge, so the decode PCC reference is comparing apples to apples
        # (otherwise a poor prefill PCC compounds into the decode PCC — see #4614).
        elif not ctx.accuracy_testing and not torch.equal(
            device_prefill_output_ids, ctx.golden.first_decode_input_ids.cpu()
        ):
            logger.warning(
                "Device prefill produced different tokens than CPU prefill; "
                "using CPU prefill output as decode PCC reference."
            )
            input_args["input_ids"] = ctx.golden.first_decode_input_ids.to(
                session.device
            )
            if iospec:
                xs.mark_sharding(input_args["input_ids"], session.mesh, iospec)

    # The prefill call above already consumed gt[0] as teacher-forced input for
    # the first decode step, so the decode call must start its ground-truth
    # window at gt[1] to stay aligned. It also consumed one of the logits_steps
    # total iterations, so decode runs logits_steps-1 steps (not logits_steps).
    decode_ground_truth = (
        ctx.ground_truth_tokens[1:]
        if ctx.ground_truth_tokens is not None and not ctx.decode_only
        else ctx.ground_truth_tokens
    )
    decode_steps = logits_steps if ctx.decode_only else logits_steps - 1
    device_decode_logits, _ = generate_and_benchmark(
        compiled_logits,
        input_args,
        session.device,
        decode_steps,
        verbose=False,
        ground_truth_tokens=decode_ground_truth,
        collect_logits=True,
    )

    output_logits = device_prefill_logits + device_decode_logits
    print("\nPCC/TOPK benchmark complete")
    return output_logits


def benchmark_llm_torch_xla(
    model_loader,
    model_variant,
    display_name,
    batch_size,
    loop_count,
    task,
    data_format,
    input_sequence_length,
    ttnn_perf_metrics_output_file,
    read_logits_fn,
    required_pcc,
    compile_config: CompileConfig,
    sharding_config: Optional[ShardingConfig] = None,
    accuracy_config: Optional[AccuracyConfig] = None,
    pcc_mode: Optional[PccMode] = None,
    max_output_tokens=None,
    decode_only: bool = False,
    weight_dtype_overrides: dict = None,
    use_mla_cache: bool = False,
    expected_ops: list = None,
    check_fusions_enabled: bool = False,
    use_indexer_cache: bool = False,
    experts_implementation: Optional[str] = None,
):
    """
    Benchmark an LLM (Large Language Model) using PyTorch and torch-xla.

    This function loads an LLM, compiles it with torch-xla for the Tenstorrent backend,
    and measures its text generation performance. It performs warmup runs, collects token
    generation metrics, and validates output correctness via PCC (Pearson Correlation Coefficient)
    or token accuracy (TOP1/TOP5) when accuracy_testing is enabled.
    The benchmark measures tokens per second on the device backends.

    Args:
        model_loader: Model loader instance for loading the LLM
        model_variant: Specific variant/version of the model to benchmark
        display_name: Human-readable model name (used for export/report naming)
        batch_size: Batch size for text generation
        loop_count: Number of inference iterations
        task: Task type
        data_format: Data precision format
        input_sequence_length: Length of input sequence for generation context
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        read_logits_fn: Callback function to extract logits from model output
        required_pcc: Required PCC threshold for validation
        compile_config: tt-mlir / torch-xla compilation knobs (CompileConfig)
        sharding_config: Multi-chip mesh and sharding specs (ShardingConfig);
            None means single-chip
        accuracy_config: Token-accuracy testing settings (AccuracyConfig);
            when enabled, validates TOP1/TOP5 against reference data instead of PCC
        pcc_mode: PCC-only iteration mode; defaults to the TT_PCC_MODE env var
        max_output_tokens: Max tokens to generate (defaults to fill the cache)
        decode_only: Run prefill on CPU and only benchmark decode on device
        weight_dtype_overrides: Optional per-module weight dtype overrides
        use_mla_cache / use_indexer_cache: Cache variants for MLA / indexer models
        expected_ops / check_fusions_enabled: Op-fusion verification settings
        experts_implementation: Expert implementation type
    Returns:
        Benchmark result containing token generation performance metrics and model information
    """
    sharding_config = sharding_config or ShardingConfig()
    accuracy_config = accuracy_config or AccuracyConfig()
    accuracy_testing = accuracy_config.enabled

    # PCC-only iteration mode. Defaults to the TT_PCC_MODE env var (a dev
    # iteration knob) but can be passed explicitly. Prefill still runs first
    # even for isolated decode: the standalone decode graph crashes when
    # compiled at optimization_level > 0.
    pcc = pcc_mode if pcc_mode is not None else PccMode.from_env()

    _validate_args(
        data_format=data_format,
        model_loader=model_loader,
        loop_count=loop_count,
        input_sequence_length=input_sequence_length,
        decode_only=decode_only,
        accuracy_testing=accuracy_testing,
        task=task,
        enable_create_d2m_subgraphs=compile_config.enable_create_d2m_subgraphs,
        optimization_level=compile_config.optimization_level,
    )

    xr.set_device_type("TT")

    # Set up for multi-chip if applicable
    if (
        sharding_config.mesh_config_fn is not None
        and sharding_config.shard_spec_fn is not None
    ):
        is_multichip = xr.global_runtime_device_count() > 1
        if is_multichip:
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            xr.use_spmd()
    else:
        is_multichip = False

    # Set up config variables
    # WARNING: max_cache_len determines context window size for accuracy testing.
    # Reference outputs must be generated with total_length = max_cache_len for accurate comparison.
    # Changing this value requires regenerating ALL reference outputs (*.refpt files).
    max_cache_len: int = input_sequence_length

    # Connect the device
    device: torch.device = torch_xla.device()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_loader,
        model_variant,
        experts_implementation=experts_implementation,
    )
    full_model_name = model_loader.get_model_info(variant=model_variant).name

    # Initialize accuracy testing if enabled
    token_accuracy = None
    custom_input_prompt = None
    if accuracy_testing:
        token_accuracy, custom_input_prompt = init_accuracy_testing(
            model_name_for_accuracy=accuracy_config.model_name_for_accuracy,
            max_cache_len=max_cache_len,
            tokenizer=tokenizer,
            hf_model_name=accuracy_config.hf_model_name_for_accuracy,
        )

    # Session owns input construction + device placement + cache bookkeeping.
    # mesh is attached later, once multi-chip sharding has been set up.
    session = GenerationSession(
        tokenizer=tokenizer,
        model_config=model.config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        mesh=None,
        use_mla_cache=use_mla_cache,
        kv_cache_sharding_spec=sharding_config.kv_cache_sharding_spec,
        input_output_sharding_spec=sharding_config.input_output_sharding_spec,
        custom_input_prompt=custom_input_prompt,
        input_prompt_tokens=(token_accuracy.input_prompt if accuracy_testing else None),
    )

    # Construct inputs, including static cache
    input_args = session.build_inputs()

    # Initialize indexer cache if enabled (needs to be done before model.to(device))
    # Models using this cache are expected to handle stale values since it cannot be
    # reset once the model is transferred to device.
    if use_indexer_cache:
        init_indexer_cache(
            model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )

    # Limit maximum generation count to fit within preallocated static cache
    if max_output_tokens is None:
        max_output_tokens = max_cache_len - input_args["input_ids"].shape[1]

    # Run CPU prefill+decode (used as PCC baseline, and as decode-only prefill).
    # The returned CpuReference carries the post-prefill seeds (input ids, cache
    # position, KV cache, cumulative lengths) consumed by the device phases.
    cpu_reference = None
    cpu_output_logits = None
    if not accuracy_testing:
        cpu_reference = compute_cpu_reference(model, read_logits_fn, input_args)
        cpu_output_logits = cpu_reference.output_logits

    # Transfer model to device
    model = model.to(device, dtype=torch.bfloat16)

    # Shard model if shard spec function is provided
    mesh = None
    if is_multichip:
        shard_specs = sharding_config.shard_spec_fn(model_loader, model)
        mesh = get_mesh(model_loader, sharding_config.mesh_config_fn)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # Apply sharding constraint on lm_head output to all_gather logits
        if hasattr(model, "lm_head") and model.lm_head is not None:
            hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
            model.lm_head.register_forward_hook(hook)

    # Attach the (possibly None) mesh now that sharding setup is complete.
    session.mesh = mesh

    # Set XLA compilation options
    num_layers_override = getattr(model_loader, "num_layers", None)
    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=num_layers_override,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
    )
    options = build_compile_options(
        compile_config,
        export_model_name=export_model_name,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
    )
    torch_xla.set_custom_compile_options(options)

    apply_weight_dtypes(model, model_loader, weight_dtype_overrides)

    # ========================================================
    # ON-DEVICE PERF + PCC/TOPK BENCHMARK
    # ========================================================

    ground_truth_for_benchmark = (
        token_accuracy.reference_tokens if accuracy_testing else None
    )
    run_ctx = DeviceRunContext(
        session=session,
        model=model,
        read_logits_fn=read_logits_fn,
        decode_only=decode_only,
        pcc=pcc,
        accuracy_testing=accuracy_testing,
        golden=cpu_reference,
        ground_truth_tokens=ground_truth_for_benchmark,
        tokenizer=tokenizer,
    )

    iteration_times = run_perf_phase(
        run_ctx, initial_inputs=input_args, max_output_tokens=max_output_tokens
    )
    output_logits = run_logits_phase(run_ctx, max_output_tokens=max_output_tokens)

    # ========================================================
    # METRICS & VALIDATION
    # ========================================================

    perf = summarize_perf(iteration_times, decode_only)

    metadata = get_benchmark_metadata()

    model_type = "text-generation"
    dataset_name = (
        "Tale of Two Cities (Reference Data)" if accuracy_testing else "Random Data"
    )

    # Extract number of layers from model config if available
    num_layers = (
        model.config.num_hidden_layers
        if hasattr(model.config, "num_hidden_layers")
        else -1
    )

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=perf.decode_total_time,
        total_samples=perf.decode_total_tokens,
        samples_per_sec=perf.tokens_per_second,
        batch_size=batch_size,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        ttft_ms=perf.ttft_ms,
    )

    evaluation_score = 0.0
    custom_measurements = [
        {
            "measurement_name": "ttft",
            "value": perf.ttft_ms,
            "target": -1,
        },
    ]

    if accuracy_testing:
        evaluation_score, accuracy_measurements = evaluate_accuracy(
            output_logits, token_accuracy
        )
        custom_measurements.extend(accuracy_measurements)
    else:
        evaluate_pcc(
            output_logits,
            cpu_output_logits,
            required_pcc,
            decode_only=decode_only,
            assert_prefill=pcc.assert_prefill,
            assert_decode=pcc.assert_decode,
        )

    # Get device count and mesh info for metrics
    arch = get_xla_device_arch()
    device_count = xr.global_runtime_device_count()
    mesh_shape = tuple(mesh.shape()) if mesh is not None else None

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=(input_sequence_length,),
        loop_count=loop_count,
        data_format=data_format,
        total_time=perf.decode_total_time,
        total_samples=perf.decode_total_tokens,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=compile_config.optimization_level,
        program_cache_enabled=True,
        trace_enabled=compile_config.trace_enabled,
        experimental_weight_dtype=compile_config.experimental_weight_dtype,
        model_info=full_model_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=arch,
        input_is_image=False,
        input_sequence_length=input_sequence_length,
        device_count=device_count,
        mesh_shape=mesh_shape,
    )

    if check_fusions_enabled and expected_ops:
        check_fusions(
            expected_ops=expected_ops,
            export_model_name=export_model_name,
            modules_dir=MODULE_EXPORT_PATH,
        )

    return result
