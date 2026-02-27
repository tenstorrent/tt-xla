# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from vllm.inputs import ProcessorInputs, PromptType
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from typing import TypeAlias

    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.config.cache import BlockSize
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
    enable_const_eval_on_cpu: bool = False

    min_context_len: int = 128
    batch_size: int = 1
    enable_precompile_all: bool = True

    # Flag to enable data parallel execution of a model. It will require
    # - batch_size > 1
    # - max_num_seqs > 1
    # Only supported for pooling/embedding models.
    enable_data_parallel: bool = False

    # Flag to enable tensor parallel execution of a model. We are relying on
    # TPU model loader to share the model across multiple devices.
    enable_tensor_parallel: bool = False

    # Optimization level for tt-mlir compilation.
    optimization_level: int = 0

    # Target dtype for weight conversion (e.g. "bfp8", "bfp4"). Empty disables.
    experimental_weight_dtype: str = ""

    def get_pjrt_compile_config(self) -> dict:
        return {
            "enable_const_eval": self.enable_const_eval,
            "enable_const_eval_on_cpu": self.enable_const_eval_on_cpu,
            "optimization_level": self.optimization_level,
            "experimental_weight_dtype": self.experimental_weight_dtype,
        }


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

    def __post_init__(self):
        torch._dynamo.config.ignore_logging_methods(logger.info)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> str:
        if attn_selector_config.use_sparse:
            raise NotImplementedError(
                "Sparse Attention is not supported on TT devices."
            )
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
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return NotImplementedError
        # return "vllm.lora.punica_wrapper.punica_tpu.PunicaWrapperTPU"

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
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        from vllm.config import CompilationMode, CUDAGraphMode

        # Use AscendScheduler as the default scheduler for TT (except for pooling models)
        if vllm_config.model_config.runner_type != "pooling":
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_tt.scheduler.AscendScheduler"
            )

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

        assert (
            vllm_config.speculative_config is None
        ), "TT does not support speculative decoding"

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

        from .attention import TTAttentionBackend

        cache_config.block_size = TTAttentionBackend.get_page_size(
            vllm_config
        )  # type: ignore[assignment]

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_tt.worker.TTWorker"

        assert (
            not vllm_config.speculative_config
        ), "Speculative decoding is not yet supported for TT backend"

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
            vllm_config.scheduler_config.chunked_prefill_enabled = False
            vllm_config.scheduler_config.max_num_batched_tokens = max(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

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
        prompt: PromptType,
        params: ParamsType,
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams, SamplingType

        if (
            isinstance(params, SamplingParams)
            and params.sampling_type == SamplingType.RANDOM_SEED
        ):
            raise ValueError("Torch XLA does not support per-request seed.")

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
