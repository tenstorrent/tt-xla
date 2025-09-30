# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch_xla
import torch_xla.runtime as xr

# import ttnn

from typing import Any, Callable, Dict, List, Optional, Type, Union, Literal

from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm.forward_context import set_forward_context
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

# from .model_loader import TTModelLoader
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.tpu import TPUModelLoader
from .platform import TTPlatform
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsTensors,
    ModelRunnerOutput,
)
from .input_batch import CachedRequestState, InputBatch

# from vllm.worker.tt_model_runner import (TTModelInput, TTSamplingParams,
#                                          sample_tokens)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

import numpy as np

logger = init_logger(__name__)

from dataclasses import dataclass

from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase


from vllm.tasks import GenerationTask, SupportedTask

from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model,
    is_text_generation_model,
)

from .attention import TTAttentionBackend, TTMetadata

from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
    KVConnectorOutput,
)


from vllm.v1.worker.utils import (
    bind_kv_cache,
)

from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
)

from vllm.config import (
    ParallelConfig,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)

from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available, prev_power_of_2


def generate_attn_mask(
    context_lens: torch.Tensor,
    num_query_tokens: int,
    num_query_heads: int,
    max_model_len: int,
    dtype,
    device,
) -> torch.Tensor:
    L, S = num_query_tokens, max_model_len
    attn_mask = torch.zeros(L, S, dtype=dtype)

    length = context_lens[0].item()
    if L != 1:
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = attn_mask.masked_fill(
            temp_mask.logical_not(), torch.ones(()) * float("-inf")
        )

    attn_mask[:, length:] = float("-inf")
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    if L == 1:
        attn_mask = attn_mask.repeat(1, 1, num_query_heads, 1)
    torch.set_printoptions(threshold=1000000, linewidth=1000000)
    print(f"ATTN MASK: {attn_mask}")
    return attn_mask.detach().to(device)


def _make_src_and_dst_indices(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    src_device: Union[torch.device, str],
    dst_device: Union[torch.device, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    src_indices = torch.tensor(src_block_ids, device=src_device, dtype=torch.int64)
    dst_indices = torch.tensor(dst_block_ids, device=dst_device, dtype=torch.int64)
    return src_indices, dst_indices


@torch.compile(backend="tt")
def _insert_blocks_to_tpu(
    cpu_cache: torch.Tensor,
    tpu_cache: torch.Tensor,
    cpu_block_indices: torch.Tensor,
    tpu_block_indices: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(tpu_cache, True)
    tpu_cache[tpu_block_indices] = cpu_cache[cpu_block_indices].to(tpu_cache.device)


@torch.compile(backend="tt")
def _swap_out_tpu_blocks(
    tpu_cache: torch.Tensor,
    cpu_cache: torch.Tensor,
    tpu_block_indices: torch.Tensor,
    cpu_block_indices: torch.Tensor,
) -> None:
    """tpu blocks to cpu blocks"""
    torch.ops.xla.dynamo_set_buffer_donor_(tpu_cache, True)
    cpu_cache[cpu_block_indices] = tpu_cache[tpu_block_indices].cpu()


def copy_kv_blocks(
    src_kv_caches: dict[str, torch.Tensor],
    dst_kv_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
    direction: Literal["h2d", "d2h"],
) -> None:
    """Copy kv blocks between different buffers."""
    if (
        not src_kv_caches
        or not dst_kv_caches
        or not src_block_ids
        or not dst_block_ids
        or len(src_block_ids) != len(dst_block_ids)
    ):
        return

    src_device = next(iter(src_kv_caches.values())).device
    dst_device = next(iter(dst_kv_caches.values())).device

    src_indices, dst_indices = _make_src_and_dst_indices(
        src_block_ids=src_block_ids,
        dst_block_ids=dst_block_ids,
        src_device=src_device,
        dst_device=dst_device,
    )

    _copy_fn = _insert_blocks_to_tpu if direction == "h2d" else _swap_out_tpu_blocks
    for layer_name in src_kv_caches:
        src_tensor = src_kv_caches[layer_name]
        dst_tensor = dst_kv_caches[layer_name]
        _copy_fn(src_tensor, dst_tensor, src_indices, dst_indices)


@dataclass(frozen=True)
class TTSamplingParams:
    """
    Used by TTModelInput.
    """

    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class TTModelInput(ModelRunnerInputBase):
    """
    Used by the TTModelRunner.
    """

    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: Optional[List[int]]
    seq_groups: List[int]
    block_tables: torch.Tensor
    unpadded_batch_size: int
    tt_sampling_params: Optional[TTSamplingParams]
    sampling_params_list: Optional[List[Any]]
    compat_sampling_used: bool
    sampling_metadata: Optional["SamplingMetadata"]
    multi_modal_kwargs: Dict[str, Any]
    cross_block_tables: torch.Tensor
    is_first_multi_step: bool = True
    is_last_step: bool = True
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "prompt_lens": self.prompt_lens,
            "seq_groups": self.seq_groups,
            "block_tables": self.block_tables,
            "unpadded_batch_size": self.unpadded_batch_size,
            "tt_sampling_params": self.tt_sampling_params,
            "sampling_params_list": self.sampling_params_list,
            "compat_sampling_used": self.compat_sampling_used,
            "sampling_metadata": self.sampling_metadata,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "cross_block_tables": self.cross_block_tables,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
        }

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["TTModelInput"],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "TTModelInput":
        return cls(**tensor_dict)


from transformers import TopPLogitsWarper


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # Do not keep the entire vocab size after top k.
    # Instead, keep the k size tensor and record the associated indices.
    if k < 1:  # no top-k sampling if set to -1 or 0
        top_k_values, top_k_indices = logits, torch.arange(logits.shape[-1]).unsqueeze(
            0
        ).repeat(logits.shape[0], 1)
    else:
        top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = TopPLogitsWarper(top_p=p)(None, top_k_values)
    probs = torch.softmax(top_p_values / temperature, dim=-1)
    probs = torch.nan_to_num(
        probs
    )  # convert nan to num to prevent error in multinomial
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


def sample_tokens(logits, tt_sampling_params: TTSamplingParams):
    if tt_sampling_params.temperature == 0 or True:  # greedy decoding
        return torch.argmax(logits, dim=-1)
    else:  # top-k top-p sampling
        return top_pk_logits_efficient(
            logits,
            p=tt_sampling_params.top_p,
            k=tt_sampling_params.top_k,
            temperature=tt_sampling_params.temperature,
        )


class TTModelRunner(KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        trace_mode: bool,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        # self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config
        self.num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.kv_caches: list[torch.Tensor] = []
        self.use_spmd = False

        self.max_num_reqs = self.scheduler_config.max_num_seqs

        self.load_config.device = "cpu"

        self.kv_cache_dtype = self.model_config.dtype

        # Model-related.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, LayerBlockType.attention
        )
        self.num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.hidden_size = self.model_config.get_hidden_size()
        self.vocab_size = self.model_config.get_vocab_size()

        # Because of multiprocessing, the config-dependent
        # class attributes might not have been set in this process,
        # so we need to call this again.
        TTPlatform.check_and_update_config(vllm_config)

        # Currently, TT model runner doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.device = device
        self.trace_mode = trace_mode

        # Whether to sample on device
        self.sample_on_device_mode = TTPlatform.sample_on_device_mode

        logger.info(
            "TTModelRunner: trace_mode=%s, sample_on_device_mode=%s",
            self.trace_mode,
            self.sample_on_device_mode,
        )

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Cached request states. Request states are tracked in the runner so
        # they don't need to be re-sent every scheduling step. For requests
        # that have been scheduled before, only the diff is received from
        # the scheduler output.
        self.requests: dict[str, CachedRequestState] = {}
        self.pin_memory = is_pin_memory_available()

        # tensors for structured decoding
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory
        )

    def load_model(self) -> None:
        loader = get_model_loader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        ).to(self.device)
        self.model.compile(backend="tt")

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            if attn_module.attn_type == AttentionType.DECODER:
                if isinstance(attn_module, ChunkedLocalAttention):
                    logger.warning_once(
                        "Using irope in Pallas is not supported yet, it "
                        "will fall back to global attention for long context."
                    )
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=False,
                    )
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=False,
                    )
            elif attn_module.attn_type in (
                AttentionType.ENCODER,
                AttentionType.ENCODER_ONLY,
            ):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if len(kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet."
            )
        if isinstance(kv_cache_groups[0].kv_cache_spec, AttentionSpec):
            kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        else:
            raise TypeError("Expected AttentionSpec")

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert (
                len(kv_cache_tensor.shared_by) == 1
            ), "KV cache shared by multiple layers is not supported for TT"

        # Initialize persistent input batch with block size from kv_cache_spec.
        # The persistent batch optimization reduces overhead between steps
        # when consecutive batches contain mostly the same requests.
        max_num_reqs = self.scheduler_config.max_num_seqs
        print(f"max_num_reqs: {max_num_reqs}")
        max_model_len = self.model_config.max_model_len
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        self.input_batch = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            block_sizes=[kv_cache_spec.block_size],
        )

        # Make the assumption that we are tensor parallel by
        # min(number of devices, number of KV heads).
        # TODO: move this into model.allocate_kv_cache.
        model_config = self.model_config
        data_parallel = 1
        if (
            self.model_config.override_tt_config
            and "data_parallel" in model_config.override_tt_config
        ):
            data_parallel = model_config.override_tt_config["data_parallel"]
        num_devices = xr.global_runtime_device_count()
        total_kv_heads = kv_cache_spec.num_kv_heads
        num_kv_heads = total_kv_heads // min(num_devices, total_kv_heads)

        # kv_cache_shape = (
        #     kv_cache_config.num_blocks,
        #     num_kv_heads,
        #     kv_cache_spec.block_size,
        #     kv_cache_spec.head_size,
        # )
        # Allocate KV cache tensors.
        # self.kv_caches = self.model.allocate_kv_cache(kv_cache_shape, dtype,
        #                                               num_layers)

        # kv_caches = [
        #     torch.zeros(kv_cache_shape, dtype=dtype, device=self.device)
        #     for _ in range(num_layers)
        # ]

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in " "TPU."
            )
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: dict[str, torch.Tensor] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes  # noqa
                if isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        assert self.original_parallel_config is not None
                        tp_size = self.original_parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode"
                        )
                    kv_cache_shape = TTAttentionBackend.get_kv_cache_shape(
                        1,
                        num_kv_heads,
                        32,
                        kv_cache_spec.head_size,
                    )
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = torch.zeros(kv_cache_shape, dtype=dtype).to(
                        self.device
                    )
                    print(f"Layer name: {layer_name}")
                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        # print(f"KV CACHE SHAPE: {kv_cache_shape}")
        print(f"KV CACHES: {kv_caches.keys()}")
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            get_kv_transfer_group().set_host_xfer_buffer_ops(copy_kv_blocks)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the
        scheduler output.
        The updated states are used in `_prepare_model_inputs` to create the
        input tensors for the model.
        Based on _update_states for GPU/TPU backends.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert (
                new_req_data.sampling_params is not None
            ), "Pooling is not supported for TT yet"
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                # mm_inputs=new_req_data.mm_inputs,
                mm_kwargs=new_req_data.mm_kwargs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

    def _prepare_model_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> TTModelInput:

        assert scheduler_output.total_num_scheduled_tokens > 0
        input_batch = self.input_batch
        num_reqs = input_batch.num_reqs
        assert num_reqs > 0
        assert (
            len(input_batch.block_table.block_tables) == 1
        ), "Currently only supporting 1 KV cache group"

        # Second dim of block table kept as fixed size max_num_blocks_per_req
        # (ceil(max_model_len / block_size)) so ttnn tracing can work
        # (requires constant shape).
        block_tables = input_batch.block_table[0].get_cpu_tensor()[:num_reqs, :]

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0
        if is_prompt:
            # Assert no running requests
            assert (
                len(scheduler_output.scheduled_cached_reqs.req_ids) == 0
            ), "Currently only supporting all prefills or all decodes in batch"

            input_positions = 0
            max_prompt_tokens = max(input_batch.num_prompt_tokens[:num_reqs])
            input_tokens = input_batch.token_ids_cpu_tensor[
                :num_reqs, :max_prompt_tokens
            ]
            prompt_lens = torch.from_numpy(input_batch.num_prompt_tokens[:num_reqs])
        else:
            input_positions = torch.from_numpy(input_batch.num_tokens[:num_reqs] - 1)
            input_tokens = input_batch.token_ids_cpu_tensor[
                torch.arange(num_reqs), input_positions
            ].view(-1, 1)
            prompt_lens = None

            # TODO: Remove once TT models can support arbitrary batch sizes.
            # Pad batch to max_num_reqs.
            if input_tokens.shape[0] < input_batch.max_num_reqs:
                batch_pad = input_batch.max_num_reqs - input_tokens.shape[0]
                input_tokens = torch.cat(
                    [input_tokens, torch.zeros(batch_pad, 1, dtype=torch.int32)]
                )
                # Pad positions with -1 to indicate no position
                input_positions = torch.cat(
                    [input_positions, torch.ones(batch_pad, dtype=torch.int32) * -1]
                )
                block_tables = torch.cat(
                    [
                        block_tables,
                        torch.zeros(
                            batch_pad, block_tables.shape[1], dtype=torch.int32
                        ),
                    ]
                )
        print(f"input_tokens: {input_tokens}")

        def next_multiple_of_32(n):
            return (n + 31) & ~31

        def pad_input_ids(input_ids):
            batch_size, seq_len = input_ids.shape
            padded_seq_len = next_multiple_of_32(seq_len)
            padded_input_ids = torch.zeros(
                batch_size,
                padded_seq_len,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            padded_input_ids[:, :seq_len] = input_ids
            return padded_input_ids

        if prompt_lens is not None:
            input_tokens = pad_input_ids(input_tokens)
        print(f"input_tokens after padding shape: {input_tokens.shape}")
        print(f"input_tokens after padding: {input_tokens}")
        print(f"PROMT LENS: {prompt_lens}")
        print(f"input_positions: {input_positions}")

        # Sampling-related.
        temperature = input_batch.sampling.temperature_cpu[:num_reqs]
        top_p = input_batch.sampling.top_p_cpu[:num_reqs]
        top_k = input_batch.sampling.top_k_cpu[:num_reqs]
        if not np.all(temperature == temperature[0]):
            logger.warning(
                "Currently only supporting same temperature for all "
                "sequences in batch, falling back to first sequence's "
                "temperature (%s)",
                temperature[0],
            )
        if not np.all(top_k == top_k[0]):
            logger.warning(
                "Currently only supporting same top_k"
                "for all sequences in batch, "
                "falling back to first sequence's top_k (%s)",
                top_k[0],
            )
        if not np.all(top_p == top_p[0]):
            logger.warning(
                "Currently only supporting same top_p"
                "for all sequences in batch, "
                "falling back to first sequence's top_p (%s)",
                top_p[0],
            )
        tt_sampling_params = TTSamplingParams(
            temperature=temperature[0],
            top_k=top_k[0],
            top_p=top_p[0],
        )

        if isinstance(input_positions, np.ndarray):
            input_positions = torch.from_numpy(input_positions)
        if isinstance(input_positions, int):
            input_positions = torch.tensor(input_positions).reshape([1])

        assert (
            not TTPlatform.compat_sampling_possible
        ), "Compatibility sampling is not yet supported in V1 TT backend"
        sampling_params_list: list[Any] = []
        compat_sampling_used = False
        sampling_metadata = None

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            seq_groups=None,  # Not used in V1
            block_tables=block_tables,
            unpadded_batch_size=num_reqs,
            tt_sampling_params=tt_sampling_params,
            sampling_params_list=sampling_params_list,
            compat_sampling_used=compat_sampling_used,
            sampling_metadata=sampling_metadata,
            multi_modal_kwargs={},  # Not yet supported in V1
            cross_block_tables=None,  # Not yet supported in V1
        )

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def compute_logits(self, sample_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(sample_hidden_states, None)

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = scheduler_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(req_id)
            if mask_index is None:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.grammar_bitmask_cpu[struct_out_indices] = torch.from_numpy(
            grammar_bitmask[mask_indices]
        )
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        struct_out_indices = torch.tensor(struct_out_indices, dtype=torch.long)
        self.require_structured_out_cpu[struct_out_indices] = True
        return (
            self.require_structured_out_cpu[:num_reqs].to(logits.device),
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device),
            self.structured_decode_arange.to(logits.device),
        )

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        device = logits.device
        return torch.where(
            require_struct_decoding.to("cpu"),
            self.apply_grammar_bitmask(
                logits.to("cpu"), grammar_bitmask.to("cpu"), arange.to("cpu")
            ),
            logits.to("cpu"),
        ).to(device)

    def apply_grammar_bitmask(
        self, logits: torch.Tensor, grammar_bitmask: torch.Tensor, arange: torch.Tensor
    ):
        assert logits.shape[0] == grammar_bitmask.shape[0]
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (
                torch.bitwise_right_shift(grammar_bitmask[i][:, None], arange[None, :])
                & 1
            ) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[: self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf")
            )
        return logits_cloned

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        """Execute the model with the given scheduler output.
        Note: currently does not support chunked prefill."""

        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        # Prepare model inputs
        model_input = self._prepare_model_inputs(scheduler_output)
        is_decode = model_input.prompt_lens is None
        execute_model_kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
            **(model_input.multi_modal_kwargs or {}),
        }
        if not is_decode:
            execute_model_kwargs["prompt_lens"] = model_input.prompt_lens
        else:
            execute_model_kwargs["start_pos"] = model_input.input_positions
        if self.sample_on_device_mode == "all" or (
            self.sample_on_device_mode == "decode_only" and is_decode
        ):
            execute_model_kwargs["sampling_params"] = model_input.tt_sampling_params

        cur_pos = model_input.input_positions if is_decode else model_input.prompt_lens

        print(f"CUR POS: {cur_pos}")
        attn_metadata = TTMetadata(
            cur_pos=cur_pos,
            attn_mask=generate_attn_mask(
                cur_pos,
                model_input.input_tokens.shape[-1],
                self.num_query_heads,
                self.model_config.max_model_len,
                torch.bfloat16,
                self.device,
            ),
        )
        layer_names = get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata for layer_name in layer_names
        }

        torch_xla.core.xla_model.wait_device_ops()
        torch_xla.core.xla_model.mark_step()
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=scheduler_output.total_num_scheduled_tokens,
        ):
            self.maybe_setup_kv_connector(scheduler_output)
            # Execute model
            if not is_decode:
                positions = torch.zeros(32, dtype=torch.int32)
                positions[: model_input.prompt_lens[0]] = torch.arange(
                    0, model_input.prompt_lens[0]
                )
                hidden_states = self.model(
                    input_ids=model_input.input_tokens.to(self.device),
                    # positions=model_input.input_positions.to(self.device),
                    positions=positions.to(self.device),
                )
                # [batch_size, seq_len, vocab_size]
            else:
                # TODO: Add encoder-decoder support
                enc_dec_kwargs: dict[str, Any] = {}
                hidden_states = self.model(
                    input_ids=model_input.input_tokens.to(self.device),
                    positions=model_input.input_positions.to(self.device),
                )

            logits = self.compute_logits(hidden_states)

        logits = logits.to("cpu")
        # if scheduler_output.grammar_bitmask is not None:
        #     (
        #         require_struct_decoding,
        #         grammar_bitmask_padded,
        #         arange,
        #     ) = self.prepare_structured_decoding_input(logits, scheduler_output)
        #     logits = self.structured_decode(
        #         require_struct_decoding, grammar_bitmask_padded, logits, arange
        #     )
        # print(f"TT OUT: {tt_out}")
        # torch_xla.core.xla_model.mark_step()
        # torch_xla.core.xla_model.wait_device_ops()

        # if not self.sample_on_device_mode or (
        #     self.sample_on_device_mode == "decode_only" and not is_decode
        # ):
        #     next_idx = 0 if is_decode else model_input.prompt_lens[-1]-1
        #     next_logits = logits[
        #         : self.input_batch.num_reqs, next_idx, :
        #     ]  # unpadded batch, vocab of last token
        #     next_token_ids = sample_tokens(next_logits, model_input.tt_sampling_params)
        # else:
        #     next_token_ids = logits

        next_idx = 0 if is_decode else model_input.prompt_lens[-1] - 1
        next_logits = logits[
            : self.input_batch.num_reqs, next_idx, :
        ]  # unpadded batch, vocab of last token
        next_token_ids = sample_tokens(next_logits, model_input.tt_sampling_params)

        print(f"NEXT TOKEN IDS: {next_token_ids}")
        # next_token_ids = next_token_ids[::1]

        sampled_token_ids = [
            [int(next_token_ids[i])] for i in range(self.input_batch.num_reqs)
        ]

        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfers(
            scheduler_output
        )
        kv_connector_output = (
            None
            if (finished_sending is None and finished_recving is None)
            else KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
            )
        )

        output = self._generate_runner_output(sampled_token_ids, kv_connector_output)
        return output

    def _generate_runner_output(
        self,
        sampled_token_ids: list[list[int]],
        kv_connector_output: Optional[KVConnectorOutput] = None,
    ):
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        for req_idx, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.model_config.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.model_config.max_model_len}"
            )

            # Update persistent batch
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens[req_idx] = end_idx

            # Update request state
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Empty prompt log probs
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[: self.input_batch.num_reqs]:
            prompt_logprobs_dict[req_id] = None

        # Note: currently does not support speculative decoding, log probs,
        # or pooling.
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.model
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())

        return tuple(tasks)
