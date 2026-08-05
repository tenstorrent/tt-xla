# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from typing import TypeAlias

    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.config.cache import BlockSize
    from vllm.inputs import PromptType
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams

    ParamsType: TypeAlias = SamplingParams | PoolingParams
else:
    BlockSize = None
    VllmConfig = None
    PoolingParams = None
    ParamsType = None

from torch_xla import runtime as xrt

from .logger import tt_init_logger

logger = tt_init_logger(__name__)

USE_TPU_INFERENCE = False


@dataclass
class TTConfig:
    # We allow the user of the plugin to toggle consteval in tt-mlir. We would like for this to be on at all times as it results in a more performant model.
    # However, the results of the consteval graphs are stored on device permanently. When pre-compiling multiple graphs for multiple sequence lengths, we
    # will essentially end up storing the entire model on device once per graph. This can easily lead to OOM errors.
    # There is an issue tracking this in tt-mlir: https://github.com/tenstorrent/tt-mlir/issues/3888
    enable_const_eval: bool = True

    # Enables hoisting const-eval subgraphs to CPU module. When enabled, const-eval
    # operations are hoisted to be executed on the CPU instead of being executed
    # on the device.
    enable_const_eval_on_cpu: bool = True

    min_context_len: int = 128

    # Minimum request-batch size to preallocate and precompile for. This is
    # used as the lower bound for request-count shape warmup.
    # If unset, it resolves to scheduler_config.max_num_seqs.
    min_num_seqs: Optional[int] = None

    # Maximum request-batch size for *prefill* compilation and tracing.
    # When set to a value smaller than max_num_seqs, prefill graphs compile
    # and trace at this batch shape instead of max_num_seqs. With
    # enable_trace=True, each traced bucket keeps its peak activations
    # resident in DRAM; capping prefill here is the primary knob for
    # reducing prefill DRAM footprint. Decode is unaffected (always
    # compiles at max_num_seqs). Must satisfy min_num_seqs <=
    # max_prefill_num_seqs <= max_num_seqs. Defaults to max_num_seqs.
    max_prefill_num_seqs: Optional[int] = None

    # b1-prefill batch threshold. When <= this many prefills are
    # pending, AscendScheduler admits at most min_num_seqs fresh prefills/step
    # (small/b1 graph, served serially) instead of one wasted-row b32 batch;
    # above it, prefills batch as usual. 0 = off; needs min_num_seqs < max.
    prefill_batch_threshold: int = 0

    # KV-cache high-watermark for *fresh* prefill admission, as a fraction of
    # the block pool (tt-xla: large-context concurrency thrash). AscendScheduler
    # stops admitting NEW prefills once doing so would leave less than this
    # fraction of the pool free, reserving headroom for in-flight requests to
    # finish decoding instead of evicting (preempting) them and re-prefilling.
    # Continuation chunks of already-started prefills and decode are NOT gated
    # (decode may use the pool to 100%). A forward-progress guard always admits
    # at least one prefill when nothing is running, so a single large prompt is
    # never starved. 0.0 = off (legacy 1% watermark for all prefills).
    # Default 0.25 (reserve 25% free => stop admitting above ~75% usage).
    # Override at runtime with env var TTXLA_PREFILL_KV_WATERMARK_PERCENT (a
    # percent, e.g. 25), which takes precedence over additional_config.
    # Resolved/validated in TTPlatform.check_and_update_config.
    prefill_kv_watermark: float = 0.25

    # Serving telemetry. When True, the scheduler and the v1 model
    # runner emit JSON-lines telemetry (batch occupancy, prefill/decode pass
    # split, decode rate, preemption, queue depth, KV/batch utilization) for
    # offline analysis. Zero-cost when off (a single bool check on the hot path).
    # No per-step disk I/O: records buffer in memory and flush on an interval /
    # at request completion / at shutdown. See vllm_tt/telemetry.py.
    # Override at runtime with env var TTXLA_TELEMETRY (truthy), which takes
    # precedence over additional_config. Resolved in check_and_update_config.
    telemetry_enabled: bool = False

    # Directory for telemetry JSON-lines sinks (scheduler.jsonl, runner.jsonl,
    # runner_snapshot.json). Env override: TTXLA_TELEMETRY_DIR. Default ./tt_telemetry.
    telemetry_dir: str = "./tt_telemetry"

    # Minimum gap (milliseconds) between telemetry disk flushes. Env override:
    # TTXLA_TELEMETRY_FLUSH_MS. Default 1000.
    telemetry_flush_ms: float = 1000.0

    batch_size: int = 1
    enable_precompile_all: bool = True

    # Per-step prefill chunk size (caps max_num_batched_tokens); 0 (default) =
    # opt-out. When set, long prompts split into chunks of this many tokens,
    # bounding compile time + peak prefill DRAM by the chunk, not max_model_len.
    prefill_chunk_size: int = 0

    # Flag to enable data parallel execution of a model. It will require
    # - max_num_seqs > 1
    # Only supported for pooling/embedding models.
    enable_data_parallel: bool = False

    # Flag to enable tensor parallel execution of a model. We are relying on
    # TPU model loader to share the model across multiple devices.
    enable_tensor_parallel: bool = False

    # Set False to fall back to the v1 runner. Ignored for pooling models.
    use_v2_model_runner: bool = True

    # Optimization level (0, 1, or 2) that controls multiple optimization passes.
    # Level 0: All optimizations disabled
    # Level 1: Basic optimizations (optimizer + Conv2d fusion)
    # Level 2: Advanced optimizations (optimizer + memory layout + Conv2d fusion)
    optimization_level: int = 1

    # Target dtype for weight conversion (e.g. "bfp_bf8", "bfp_bf4"). Empty disables.
    experimental_weight_dtype: str = ""

    # Per-tensor weight dtype overrides for mixed precision. Either a dict
    # mapping fnmatch globs over (vLLM) parameter names to dtype strings
    # ("bfp_bf4"/"bfp_bf8"/"bf16"), with an optional "default" key for
    # unmatched weights, or a path to a JSON file of the same shape. Takes
    # precedence over experimental_weight_dtype for matched tensors. None
    # disables.
    weight_dtype_overrides: Optional[Union[dict, str]] = None

    # Toggle the tt-mlir permute+matmul fusion optimization. Mirrors the PJRT
    # compile option; defaults to True to match the PJRT default.
    experimental_enable_permute_matmul_fusion: bool = True

    # Enable fp32 destination accumulation in matmul/reduction kernels.
    #
    # PREFER LEAVING THIS None (unset). Do NOT set it to False as a default /
    # convenience in a model config or test. It is a GRAPH-WIDE override applied
    # to every matmul, including tiny index-arithmetic matmuls (e.g. the per-user
    # last-token gather that lowers to `flat_index = indices @ strides` ->
    # embedding). Forcing bf16 destination accumulation there rounds flat
    # indices >= 512 to the wrong value -> wrong gathered row -> wrong logits ->
    # per-user divergence in batched greedy decoding (tt-xla #5116). Unset is
    # better for accuracy: ttnn picks fp32 accumulation for fp32-output ops
    # (exact index math) and bf16 for bf16 compute matmuls (no memory
    # regression). Only set True/False deliberately for a validated reason.
    fp32_dest_acc_en: Optional[bool] = None

    # Override the on-device KV cache element dtype.
    experimental_kv_cache_dtype: Optional[str] = None

    # Perform token sampling on CPU instead of compiling a sampling graph for device
    cpu_sampling: bool = False

    # When True, `capture_model` precompiles only the graphs needed for decode
    # (num_tokens == 1):
    #   - `_precompile_backbone` is restricted to the decode shape.
    #   - `_precompile_select_hidden_states` is restricted to the decode shape.
    #   - `_precompile_mm_encoder` and `_precompile_structured_decoding` are
    #     skipped entirely (multimodal and structured-output decoding are not
    #     exercised in a plain decode-only run).
    # Useful for speeding up startup when only decode performance matters
    # (e.g. local debugging of decode-only tests).
    decode_only: bool = False

    # Generate fused decode graphs containing both the model forward and post-processing
    # (e.g. sampling, compute_logits, applying grammar constraints) in a single
    # graph. This will generate 4 graphs capturing greedy/non-greedy sampling
    # with and without grammar constraints.
    enable_decode_fused_graphs: bool = False

    # Override number of hidden layers (0 = use model default)
    # For debugging and testing purposes, we allow overriding the number of hidden
    # layers in the model config to enable testing with smaller models or to
    # simulate the behavior of larger models. This is done by directly modifying
    # the vllm_config before the model is loaded. We also store the original and
    # target number of layers to filter the weights accordingly during loading.
    num_hidden_layers: int = 0

    # Flag to enable 2D mesh for tensor parallel execution.
    use_2d_mesh: bool = True

    # Explicit (batch, model) SPMD mesh shape for tensor/data parallel
    # execution. When None, use_2d_mesh / parallel_mode determine the shape.
    # When set, it overrides use_2d_mesh.
    mesh_shape: Optional[list[int]] = None

    # When True, weight partition specs include the "batch" (DP) axis —
    # FSDP-style sharding that saves memory at the cost of extra communication.
    # When False (default), weights are sharded only on the "model" (TP) axis
    # and replicated across DP replicas (classic DP+TP), which incurs fewer
    # CCLs. Enable batch-axis sharding only for models that otherwise don't fit
    # on a machine.
    shard_weights_on_batch_axis: bool = False

    # Flatten model I/O to a flat token stream at the model-call boundary
    # (needed by HF forwards like Gemma-4's PLE path).
    flat_model_io: bool = False

    enable_trace: bool = False

    # PJRT IR export — when set, MLIR is dumped to `export_path` keyed by `export_model_name`.
    export_path: Optional[str] = None
    export_model_name: Optional[str] = None

    def get_pjrt_compile_config(self) -> dict:
        cfg = {
            "enable_const_eval": self.enable_const_eval,
            "enable_const_eval_on_cpu": self.enable_const_eval_on_cpu,
            "optimization_level": self.optimization_level,
            "experimental_weight_dtype": self.experimental_weight_dtype,
            "enable_trace": "true" if self.enable_trace else "false",
            "experimental_enable_permute_matmul_fusion": self.experimental_enable_permute_matmul_fusion,
        }
        if self.fp32_dest_acc_en is not None:
            cfg["fp32_dest_acc_en"] = self.fp32_dest_acc_en
        if self.experimental_kv_cache_dtype is not None:
            cfg["experimental-kv-cache-dtype"] = self.experimental_kv_cache_dtype
        if self.export_path:
            cfg["export_path"] = self.export_path
        if self.export_model_name:
            name = self.export_model_name
            if self.enable_tensor_parallel:
                name = f"{name}_g{xrt.global_ordinal()}"
            cfg["export_model_name"] = name
        return cfg


class TTPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "xla"
    device_type: str = "xla"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TT"
    dist_backend: str = "gloo"
    device_control_env_var: str = "TT_VISIBLE_DEVICES"
    simple_compile_backend: str = "tt"

    supported_quantization: list[str] = [
        # "fp8", "tpu_int8", "compressed-tensors"
    ]

    additional_env_vars: list[str] = [
        # "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"
    ]

    # Stashed from the active TTConfig so validate_request() can decide
    # whether seeded sampling is supported (host path supports it; the
    # device-side tt::sampling kernel does not yet).
    _cpu_sampling: bool = False

    @classmethod
    def _validate_speculative_decode_config(cls, vllm_config: VllmConfig) -> None:
        """Validate the first TT speculative-decode support slice.

        For now, TT only supports method='ngram' in synchronous scheduling.
        All other methods and async scheduling are rejected explicitly.
        """
        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            return

        method = getattr(speculative_config, "method", None)
        if method != "ngram":
            raise NotImplementedError(
                "TT speculative decoding currently supports only "
                "speculative_config.method='ngram'."
            )

        if getattr(vllm_config.scheduler_config, "async_scheduling", False):
            raise NotImplementedError(
                "TT ngram speculative decoding currently supports only "
                "synchronous scheduling (async_scheduling=False)."
            )

        if hasattr(speculative_config, "use_ngram_gpu") and callable(
            speculative_config.use_ngram_gpu
        ):
            if speculative_config.use_ngram_gpu():
                raise NotImplementedError(
                    "TT speculative decoding does not support ngram_gpu yet. "
                    "Use method='ngram' (CPU proposer path)."
                )

        logger.info(
            "[TT] Enabling speculative decoding with method='ngram' "
            "(synchronous scheduling only)."
        )

    def __post_init__(self):
        torch._dynamo.config.ignore_logging_methods(logger.info)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if attn_selector_config.use_sparse:
            raise NotImplementedError(
                "Sparse Attention is not supported on TT devices."
            )
        if attn_selector_config.use_mla:
            logger.info("Using TT MLA Attention backend.")
            return AttentionBackendEnum.FLASH_ATTN_MLA.get_path()
        if selected_backend != AttentionBackendEnum.CUSTOM:
            logger.info("Cannot use %s backend on TT devices.", selected_backend)

        logger.info("Using TT Attention layer.")
        return AttentionBackendEnum.CUSTOM.get_path()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        cls.device = device

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return f"xla:{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        """Return ``(free, total)`` memory in bytes.

        Some upstream multimodal models call this to size a
        memory-safe chunk for the vision encoder. TT device DRAM is managed by
        tt-metal and not queryable through this CUDA-style hook, so we report
        host memory: the value only scales the encoder chunk size (smaller ->
        more, smaller passes), so a host-memory estimate is always safe.
        """
        import psutil

        vm = psutil.virtual_memory()
        return vm.available, vm.total

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        # Emit per-attention-type kv_cache_groups (like every other platform);
        # the base default False downgrades sliding-window layers to full
        # attention, making them pay the full max_model_len KV cost.
        return True

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # The CPU wrapper is the only non-Triton LoRA punica implementation.
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_infinity_values(cls, dtype: torch.dtype) -> tuple[float, float]:
        return torch.finfo(dtype).min, torch.finfo(dtype).max

    @classmethod
    def can_update_inplace(cls):
        return False

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        return 1

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def validate_request(cls, prompt, params, processed_inputs) -> None:
        """Reject requests this platform can't currently handle.

        SamplingParams(seed=...) is not supported on the device sampler
        because the tt::sampling kernel passes a single scalar seed to
        all 32 cores, breaking per-row reproducibility. The host-side
        path (cpu_sampling=True) handles seeded sampling correctly via
        the legacy Sampler.random_sample() Gumbel-max chain. Raising
        here returns a clean per-request error to the caller without
        killing the engine; remove this once the kernel grows per-row
        seed support (see perf_debug/SEEDED_SAMPLING_NOTES.md).
        """
        if cls._cpu_sampling:
            return
        seed = getattr(params, "seed", None)
        if seed is not None:
            raise ValueError(
                "SamplingParams(seed=...) is not supported by the TT "
                "device sampler — the tt::sampling kernel does not honor "
                "per-row seeds yet. Set additional_config="
                "{'cpu_sampling': True} to use the host-side sampler "
                "which handles seeded sampling correctly."
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationMode, CUDAGraphMode

        if vllm_config.additional_config is None:
            vllm_config.additional_config = {}
        additional_config = vllm_config.additional_config
        tt_config = TTConfig(**additional_config)
        if "batch_size" in additional_config:
            logger.warning(
                "additional_config['batch_size'] is deprecated and will be removed "
                "in a future release. Use max_num_seqs instead."
            )

        max_num_seqs = vllm_config.scheduler_config.max_num_seqs

        # Resolve/validate max_prefill_num_seqs first so min_num_seqs can default to it.
        if tt_config.max_prefill_num_seqs is None:
            additional_config["max_prefill_num_seqs"] = max_num_seqs
            tt_config.max_prefill_num_seqs = additional_config["max_prefill_num_seqs"]
        elif tt_config.max_prefill_num_seqs < 1:
            raise ValueError(
                "additional_config['max_prefill_num_seqs'] must be >= 1 for the TT backend."
            )
        elif tt_config.max_prefill_num_seqs > max_num_seqs:
            raise ValueError(
                "additional_config['max_prefill_num_seqs'] must be <= max_num_seqs "
                "for the TT backend."
            )

        # Resolve/validate min_num_seqs after max_prefill_num_seqs.
        if tt_config.min_num_seqs is None:
            additional_config["min_num_seqs"] = tt_config.max_prefill_num_seqs
            tt_config.min_num_seqs = additional_config["min_num_seqs"]
        elif tt_config.min_num_seqs < 1:
            raise ValueError(
                "additional_config['min_num_seqs'] must be >= 1 for the TT backend."
            )
        elif tt_config.min_num_seqs > max_num_seqs:
            raise ValueError(
                "additional_config['min_num_seqs'] must be <= max_num_seqs "
                "for the TT backend."
            )

        if tt_config.max_prefill_num_seqs < tt_config.min_num_seqs:
            raise ValueError(
                "additional_config['max_prefill_num_seqs'] must be >= min_num_seqs "
                "for the TT backend."
            )

        # b1-prefill batch threshold: resolve default (0 = off)
        # and persist so the AscendScheduler sees a concrete value; validate >= 0.
        if additional_config.get("prefill_batch_threshold") is None:
            additional_config["prefill_batch_threshold"] = 0
        elif int(additional_config["prefill_batch_threshold"]) < 0:
            raise ValueError(
                "additional_config['prefill_batch_threshold'] must be >= 0."
            )

        # Resolve prefill_kv_watermark to a concrete fraction the scheduler can
        # read directly: default, then env override (percent), then validate.
        # See TTConfig.prefill_kv_watermark.
        if additional_config.get("prefill_kv_watermark") is None:
            additional_config["prefill_kv_watermark"] = TTConfig.prefill_kv_watermark
        env_wm = os.environ.get("TTXLA_PREFILL_KV_WATERMARK_PERCENT")
        if env_wm is not None:
            additional_config["prefill_kv_watermark"] = float(env_wm) / 100.0
        wm = float(additional_config["prefill_kv_watermark"])
        if not (0.0 <= wm < 1.0):
            raise ValueError(
                "prefill_kv_watermark (TTXLA_PREFILL_KV_WATERMARK_PERCENT / 100) "
                f"must be in [0, 1); got {wm}."
            )
        additional_config["prefill_kv_watermark"] = wm

        # Written back so the runner (typed TTConfig) and the scheduler (raw
        # additional_config, separate process) read the same settings.
        tele_enabled = bool(additional_config.get("telemetry_enabled", False))
        env_tele = os.environ.get("TTXLA_TELEMETRY")
        if env_tele is not None:
            tele_enabled = env_tele.strip().lower() in {"1", "true", "yes", "on"}
        additional_config["telemetry_enabled"] = tele_enabled

        env_tele_dir = os.environ.get("TTXLA_TELEMETRY_DIR", "").strip()
        # reset_sinks joins this path, so an empty value would delete
        # sink-named files from the process CWD.
        additional_config["telemetry_dir"] = (
            env_tele_dir
            or (additional_config.get("telemetry_dir") or "").strip()
            or TTConfig.telemetry_dir
        )

        env_tele_flush = os.environ.get("TTXLA_TELEMETRY_FLUSH_MS")
        if env_tele_flush is not None:
            try:
                additional_config["telemetry_flush_ms"] = float(env_tele_flush)
            except ValueError:
                # A telemetry knob must never crash the engine.
                additional_config["telemetry_flush_ms"] = TTConfig.telemetry_flush_ms
        elif additional_config.get("telemetry_flush_ms") is None:
            additional_config["telemetry_flush_ms"] = TTConfig.telemetry_flush_ms

        # Must precede every collector: the scheduler's is built after warmup
        # and the snapshot is never truncated on init.
        if additional_config["telemetry_enabled"]:
            from .telemetry import reset_sinks

            reset_sinks(additional_config["telemetry_dir"])

        vllm_config.additional_config = additional_config

        # Stash cpu_sampling so validate_request() can read it without
        # rebuilding TTConfig per request.
        cls._cpu_sampling = bool(
            (vllm_config.additional_config or {}).get("cpu_sampling", False)
        )

        # Use AscendScheduler as the default scheduler for TT (except for pooling models)
        if vllm_config.model_config.runner_type != "pooling":
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_tt.scheduler.AscendScheduler"
            )

        # TT does not support asynchronous scheduling (is_async_output_supported
        # is False). v0.25.1 auto-enables it when left unset, which drives the
        # AscendScheduler stale-request path into an endless preempt/retry loop.
        # Force it off.
        vllm_config.scheduler_config.async_scheduling = False

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 32)
        compilation_config = vllm_config.compilation_config

        # TT only supports DYNAMO_TRACE_ONCE compilation level
        if compilation_config.mode != CompilationMode.DYNAMO_TRACE_ONCE:
            logger.info(
                "[TT] Forcing DYNAMO_TRACE_ONCE compilation level, and "
                "disabling cudagraph."
            )
            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE

        if (
            compilation_config.cudagraph_mode is None
            or compilation_config.cudagraph_mode.max_cudagraph_mode()
            != CUDAGraphMode.NONE
        ):
            logger.info(
                "[TT] CUDA graph is not supported on TT, " "disabling cudagraphs."
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if compilation_config.backend == "":
            compilation_config.backend = "tt"

        cls._validate_speculative_decode_config(vllm_config)

        model_config = vllm_config.model_config
        if model_config is not None and model_config.dtype in (
            torch.float16,
            torch.float32,
        ):
            logger.warning(
                "The TT backend currently does not support %s. "
                "Using bfloat16 instead.",
                model_config.dtype,
            )
            model_config.dtype = torch.bfloat16

        from .attention_impls.attention import TTAttentionBackend

        cache_config.block_size = TTAttentionBackend.get_page_size(
            vllm_config
        )  # type: ignore[assignment]

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_tt.worker.TTWorker"

        if (
            scheduler_config.is_multimodal_model
            and not scheduler_config.disable_chunked_mm_input
        ):
            logger.warning(
                "TT does not support running Multimodal models"
                " without setting `--disable_chunked_mm_input`. "
                "Forcing --disable_chunked_mm_input."
            )
            scheduler_config.disable_chunked_mm_input = True

        if model_config and model_config.use_mla:
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled."
            )
            vllm_config.scheduler_config.enable_chunked_prefill = False
            vllm_config.scheduler_config.tt_chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )
        elif model_config is not None and model_config.runner_type != "pooling":
            # Opt-in via prefill_chunk_size. tt_chunked_prefill_enabled is a
            # TT-only attr (vLLM has no such field) gating the TT path; vLLM's
            # enable_chunked_prefill defaults True and can't stand in for it.
            chunk_size = int(additional_config.get("prefill_chunk_size", 0))
            if chunk_size > 0:
                # PER-SEQUENCE cap bounding the prefill activation + token-padding
                # ladder, not a batch-wide sum. Floor at one block for alignment.
                per_seq_chunk = max(chunk_size, cache_config.block_size)

                # Per-step batch-wide budget: one chunk per user, so scale by
                # max_num_seqs to batch all users' same-stage chunks in one step.
                budget = per_seq_chunk * scheduler_config.max_num_seqs
                logger.info(
                    "[TT] Chunked prefill: per-seq chunk %d, "
                    "max_num_batched_tokens %d -> %d (= chunk x max_num_seqs %d) "
                    "so up to %d users prefill per step.",
                    per_seq_chunk,
                    scheduler_config.max_num_batched_tokens,
                    budget,
                    scheduler_config.max_num_seqs,
                    scheduler_config.max_num_seqs,
                )
                scheduler_config.enable_chunked_prefill = True
                scheduler_config.tt_chunked_prefill_enabled = True
                # TT-internal per-sequence chunk cap, read by AscendScheduler
                # (per-request chunk sizing) and TTModelRunner (bucket ladder).
                scheduler_config.tt_prefill_chunk_size = per_seq_chunk
                # Derived value: a user-supplied max_num_batched_tokens is
                # intentionally overridden here (logged above).
                scheduler_config.max_num_batched_tokens = budget
                scheduler_config.max_num_encoder_input_tokens = budget
                scheduler_config.encoder_cache_size = budget

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> int:
        # TT backend requires a block size divisible by 32 for optimal performance.
        return 32

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TT.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def validate_request(
        cls,
        prompt: "PromptType",
        params: "ParamsType",
    ) -> None:
        pass

    @classmethod
    @torch.compile(backend="tt")
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        torch.ops.xla.dynamo_set_buffer_donor_(dst_cache, True)
        dst_cache[dst_block_indices] = src_cache[src_block_indices].to(dst_cache.device)

    @classmethod
    @torch.compile(backend="tt")
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """tpu blocks to cpu blocks"""
        torch.ops.xla.dynamo_set_buffer_donor_(src_cache, True)
        dst_cache[dst_block_indices] = src_cache[src_block_indices].cpu()

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        return True

    @classmethod
    def check_max_model_len(cls, max_model_len: int) -> int:
        """
        Check max_model_len for the current platform.
        """
        logger.warning(
            "--max-model-len is not specified, "
            "it's currently using model's default length %d, "
            "which might be too large."
            "Please input with --max-model-len based on your "
            "request input length and output length, to avoid "
            "unnecessary degradation.",
            max_model_len,
        )
        return max_model_len

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """Set RNG seed across all devices for the current platform. Set in
        worker after initializing the device.
        """
        return


def install_tt_accelerator_memory_info() -> None:
    """Route ``torch.accelerator.get_memory_info`` to the TT host-memory estimate.

    vLLM 0.25.1's Gemma4 multimodal encoder sizes its vision-encoder chunks with
    ``torch.accelerator.get_memory_info()``, which asserts on the TT (xla/"jax")
    device because it has no torch ``DeviceAllocator``. Reuse the same host-memory
    estimate as ``TTPlatform.mem_get_info`` (the value only scales the encoder
    chunk size, so a host-memory estimate is always safe).
    """
    if not hasattr(torch, "accelerator"):
        return

    def _tt_get_memory_info(device=None) -> tuple[int, int]:
        return TTPlatform.mem_get_info()

    torch.accelerator.get_memory_info = _tt_get_memory_info
