# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from vllm.inputs import ProcessorInputs, PromptType
from vllm.platforms.interface import Platform, PlatformEnum, _Backend
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import DEFAULT_MAX_NUM_BATCHED_TOKENS

if TYPE_CHECKING:
    from vllm.config import BlockSize, ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
else:
    BlockSize = None
    ModelConfig = None
    VllmConfig = None
    PoolingParams = None

from torch_xla import runtime as xrt

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


@dataclass
class TTConfig:
    # We allow the user of the plugin to toggle consteval in tt-mlir. We would like for this to be on at all times as it results in a more performant model.
    # However, the results of the consteval graphs are stored on device permanently. When pre-compiling multiple graphs for multiple sequence lengths, we
    # will essentially end up storing the entire model on device once per graph. This can easily lead to OOM errors.
    # There is an issue tracking this in tt-mlir: https://github.com/tenstorrent/tt-mlir/issues/3888
    enable_const_eval: bool = True
    min_context_len: int = 128
    batch_size: int = 1
    enable_precompile_all: bool = True

    # Flag to enable data parallel execution of a model. It will require
    # - batch_size > 1
    # - max_num_seqs > 1
    # Only supported for pooling/embedding models.
    is_data_parallel: bool = False

    def get_pjrt_compile_config(self) -> dict:
        return {
            "enable_const_eval": self.enable_const_eval,
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
        selected_backend: _Backend,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_v1: bool,
        use_mla: bool,
        has_sink,
    ) -> str:
        if not use_v1:
            raise ValueError("TT backend only supports V1.")
        logger.info("Using TT Attention layer.")
        return "vllm_tt.attention.TTAttentionBackend"

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
        from vllm.config import CompilationLevel, CUDAGraphMode

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 32)
        compilation_config = vllm_config.compilation_config

        # TT only supports DYNAMO_ONCE compilation level
        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info(
                "[TT] Forcing DYNAMO_ONCE compilation level, and "
                "disabling cudagraph."
            )
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

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
                vllm_config.scheduler_config.max_model_len,
                DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TT.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TPU is experimental
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        if (
            isinstance(params, SamplingParams)
            and params.sampling_type == SamplingType.RANDOM_SEED
        ):
            raise ValueError("Torch XLA does not support per-request seed.")

    @classmethod
    def is_kv_cache_dtype_supported(
        cls, kv_cache_dtype: str, model_config: "ModelConfig"
    ) -> bool:
        return True

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
