# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import suppress
from typing import TYPE_CHECKING, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from .model_runner import TTModelRunner
from vllm.v1.worker.worker_base import WorkerBase
import torch
import torch_xla.core.xla_model as xm

# from vllm.worker.tt_worker import (close_device,
#                                    get_num_available_blocks_tt,
#                                    open_device)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)

from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
)

import torch_xla.runtime as xr
from vllm.platforms import current_platform

import vllm.envs as envs


class TTWorker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )

        self.parallel_config = vllm_config.parallel_config
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        self.original_parallel_config = None
        if self.use_spmd:
            # Under SPMD mode, distributed env is initialized as if there is
            # only one worker/device.
            self.original_parallel_config = self.parallel_config
            self.parallel_config.tensor_parallel_size = 1
            self.parallel_config.pipeline_parallel_size = 1
            self.parallel_config.world_size = 1

        self.parallel_config.rank = rank

        # Initialized by init_device
        self.device = None

        # Whether to use ttnn tracing for model execution
        override_tt_config = None  # self.model_config.override_tt_config
        trace_key = "trace_mode"
        self.trace_mode = True
        if override_tt_config and trace_key in override_tt_config:
            assert override_tt_config[trace_key] in [
                True,
                False,
            ], f"Invalid {trace_key}: {override_tt_config[trace_key]}"
            self.trace_mode = override_tt_config[trace_key]

    def init_device(self) -> None:
        # self.device = open_device(
        #     self.model_config.override_tt_config, self.trace_mode)
        self.device = xm.xla_device()
        self.device_config.device = self.device

        # Init ModelRunner here, so that we have access to self.device.
        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
            trace_mode=self.trace_mode,
        )

        # Initialize the distributed environment.
        self._init_tpu_worker_distributed_environment(
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
        )

    def load_model(self):
        self.model_runner.load_model()

    # def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
    #     """
    #     For the GPU/TPU backends, this method generates the KVCacheSpec by
    #     parsing the kv cache format from each Attention module in the static
    #     forward context (compilation_config.static_forward_context).
    #     core/kv_cache_utils.py uses the KVCacheSpec along with available
    #     memory info from a profiling run to determine num blocks.

    #     For the TT backend, the static forward context is not populated since
    #     the modelling code is independent so we currently skip creating a
    #     kv cache spec for each layer, similar to the Spyre/Neuron backends.
    #     Currently we also don't run profiling to determine available memory.

    #     Return a dummy single layer KVCacheSpec and in the
    #     determine_available_memory function override num blocks using
    #     self.cache_config.num_gpu_blocks_override.
    #     """

    #     # TODO: Once we're able to populate a static forward context,
    #     # generate separate specs per layer (e.g. also sliding window, local
    #     # attention).

    #     model_config = self.model_config
    #     parallel_config = self.parallel_config
    #     cache_config = self.cache_config

    #     # Excludes TP factor since that is handled on the model side for TT.
    #     total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    #     head_size = model_config.get_head_size()
    #     dtype = (
    #         model_config.dtype
    #         if cache_config.cache_dtype == "auto"
    #         else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    #     )

    #     attn_spec = FullAttentionSpec(
    #         block_size=cache_config.block_size if cache_config.block_size else 32,
    #         num_kv_heads=total_num_kv_heads,
    #         head_size=head_size,
    #         dtype=dtype,
    #         use_mla=model_config.use_mla,
    #         sliding_window=model_config.get_sliding_window(),
    #     )
    #     kv_cache_spec: dict[str, KVCacheSpec] = {"foo": attn_spec}
    #     return kv_cache_spec

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def determine_available_memory(self) -> int:
        """
        For the GPU/TPU backends, this method runs profiling to determine
        available memory for the KV cache. The available memory is then used
        in conjunction with the output of get_kv_cache_spec to determine
        the number of kv cache blocks (total memory / page_size / num layers).

        Currenly we just return a large dummy number of bytes similar to the
        Spyre/Neuron backends and override the number of kv cache blocks.
        """

        # TODO: Once we can run profiling, return real available memory
        # instead of overriding the number of blocks.
        # num_tt_blocks = get_num_available_blocks_tt(self.vllm_config)
        # self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return 12 * 1024**3

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Cache is already initialized in initialize_from_config.
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        # Currently skip and compile/capture-trace during the first execution.
        pass

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        assert self.is_driver_worker, "There should only be one Worker for TT"
        output = self.model_runner.execute_model(scheduler_output)
        return output

    def check_health(self) -> None:
        # Worker will always be healthy as long as it's running.
        return

    def _init_tpu_worker_distributed_environment(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: Optional[str] = None,
        local_rank: int = -1,
    ) -> None:
        """Initialize the distributed environment."""
        if self.use_spmd:
            xr.use_spmd()
        # NOTE(woosuk): This is just to initialize the TP group and broadcast
        # the input objects on CPU. The all-reduce and all-gather ops on TPU
        # are invoked by `xm.all_reduce` and `xm.all_gather` which use their
        # own context.
        parallel_config = vllm_config.parallel_config
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=distributed_init_method,
            backend=current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
        )

        ensure_kv_transfer_initialized(vllm_config)

    def get_supported_tasks(self):
        return self.model_runner.get_supported_tasks()

    ## Destructor (used to close devices)

    def __del__(self):
        # Delete model runner first in case there are model artifacts
        with suppress(AttributeError):
            # attributes may be already torn down when destructor is called
            del self.model_runner

            # if self.device:
            #     close_device(self.device,
            #                       self.model_config.override_tt_config)
            #     del self.device

        if hasattr(super(), "__del__"):
            super().__del__()  # type: ignore
