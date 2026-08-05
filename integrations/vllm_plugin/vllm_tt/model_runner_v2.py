# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT Model Runner v2 (MRv2): a fork of upstream ``vllm/v1/worker/gpu/model_runner.py``.

Upstream's v2 runner is Triton/CUDA/UVA-native, so TT forks it and substitutes
host-side mechanisms (numpy input-prep, ``@torch.compile(backend="tt")`` graphs,
TT attention metadata) instead of subclassing ``GPUModelRunner``. Two things
differ from a naive port:

* 2D forward layout: TT's model takes ``[num_reqs, padded_query_len]`` (reshaped
  to 1D at the call boundary for ``flat_model_io`` models), not upstream's flat
  ``[num_tokens]``. The runner builds those 2D tensors itself; ``TTInputBatch``
  is the flat per-step bookkeeping view consumed by ``from_v2_states`` and the
  attn build.
* Upstream v2 request semantics: requests keep a stable ``TTRequestState`` slot
  for their lifetime (no condense); preempted requests are freed and resumed
  ones return via ``scheduled_new_reqs``, so ``update_requests`` has no resumed
  branch.

Deferred: multi-device SPMD/mesh, ``get_mm_embeddings``, LoRA, and the grammar /
cpu_sampling / prompt-logprobs branches.
"""

from __future__ import annotations

import bisect
import contextlib
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from tt_torch.sharding import sharding_constraint_tensor
from vllm.sampling_params import SamplingType
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from .logger import tt_init_logger
from .vllm_distributed_utils import ParallelismMode, safe_mark_sharding, shard_model

logger = tt_init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """First padding >= x (the per-step query-length bucket)."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def _adjust_min_token(min_token_size: int) -> int:
    """Round min_token_size up to a power of two that is >= 32 (32B alignment)."""
    if (min_token_size & (min_token_size - 1)) == 0 and min_token_size >= 32:
        return min_token_size

    # Default fallback is 32 (smallest valid input length).
    adjusted_value = 32
    if min_token_size > 32:
        # Round up to the next power of two.
        adjusted_value = 1 << (min_token_size - 1).bit_length()

    logger.warning(
        f"Flag min_context_len={min_token_size} is not a power of two and divisible by 32. "
        f"Adjusting to the next power of two. Using min_context_len={adjusted_value}."
    )
    return adjusted_value


def _get_token_paddings(min_token_size: int, max_token_size: int) -> list[int]:
    """Exponential token-length ladder from min_token_size up past max_token_size.

    Always starts with 1 to support single-token decode steps.
    """
    num = _adjust_min_token(min_token_size)
    paddings = [1]
    while True:
        paddings.append(num)
        if num >= max_token_size:
            break
        num *= 2
    return paddings


def replace_set_lora(model):
    """Wrap each LoRA layer's set_lora/reset_lora with a torch_xla.sync.

    Ported from the v1 runner: the integer LoRA index would otherwise trigger a
    recompilation, so a sync captures the input/metadata updates around it.
    """
    import torch_xla
    from vllm.lora.layers import BaseLayerWithLoRA

    def _tpu_set_lora(self, index, lora_a, lora_b, embeddings_tensor, bias=None):
        self._original_set_lora(index, lora_a, lora_b, embeddings_tensor)
        torch_xla.sync(wait=False)

    def _tpu_reset_lora(self, index):
        self._original_reset_lora(index)
        torch_xla.sync(wait=False)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(module, module.__class__)


class TTModelRunnerV2(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    """Tenstorrent MRv2 model runner (see module docstring)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        original_parallel_config=None,
    ) -> None:
        """Construct the split v2 state + config scalars.

        When TP/DP is enabled the SPMD device mesh is built here (see
        ``_build_device_mesh``); ``original_parallel_config`` carries the real
        (pre-collapse) parallel sizes from the worker. The model itself is
        loaded and compiled in ``load_model``; ``model``/``model_state``/``sampler``
        are None until then.
        """
        from vllm.v1.worker.block_table import MultiGroupBlockTable

        from .attention_impls.attention import (
            TPU_STR_DTYPE_TO_TORCH_DTYPE,
            TTAttentionBackend,
        )
        from .input_batch_v2 import TTInputBuffers
        from .platform import TTConfig
        from .request_state import TTRequestState
        from .sampling_state_v2 import TTSamplingStates

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.load_config = vllm_config.load_config
        self.lora_config = vllm_config.lora_config
        self.device = device

        # Speculative decode (ngram). Mirrors v1: num_spec_tokens gates the whole
        # path, drafter is only built for the ngram method.
        self.speculative_config = getattr(vllm_config, "speculative_config", None)
        self.num_spec_tokens = 0
        self.drafter = None
        self._draft_token_ids: list[list[int]] | None = None
        self._draft_token_req_ids: list[str] | None = None
        if self.speculative_config is not None:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            if self.speculative_config.method == "ngram":
                from vllm.v1.spec_decode.ngram_proposer import NgramProposer

                self.drafter = NgramProposer(vllm_config)

        # additional_config arrives as a plain dict; build the typed TTConfig
        # (as the v1 runner does) so the field reads below see real values.
        self.tt_config = TTConfig(**vllm_config.additional_config)

        # Override number of hidden layers if specified in TTConfig. Must run
        # before load_model so only the target layers are built; the weight
        # loader is filtered to match in load_model.
        from .vllm_utils import apply_hidden_layer_override

        self._original_num_layers, self._target_num_layers = (
            apply_hidden_layer_override(
                self.model_config.hf_config, self.tt_config.num_hidden_layers
            )
        )

        tt = self.tt_config
        self.enable_tensor_parallel = bool(getattr(tt, "enable_tensor_parallel", False))
        self.enable_data_parallel = bool(getattr(tt, "enable_data_parallel", False))
        self.use_2d_mesh = getattr(tt, "use_2d_mesh", True)
        self.is_sharded_compute_logits = False
        self.original_parallel_config = original_parallel_config

        # max_num_reqs may be rounded up to a dp_size multiple by the mesh build,
        # so it must be set before it sizes the state tables below.
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.dp_size = 1
        self.mesh = None
        self.parallel_mode = ParallelismMode.DISABLED
        if self.enable_tensor_parallel or self.enable_data_parallel:
            self._build_device_mesh()

        self.dtype = self.model_config.dtype
        if self.cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = (
                TPU_STR_DTYPE_TO_TORCH_DTYPE[self.dtype]
                if isinstance(self.dtype, str)
                else self.dtype
            )
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]
        # 1-byte accounting stand-in so vLLM budgets blocks for the BFP8 footprint.
        self.kv_cache_spec_dtype = (
            torch.uint8
            if getattr(tt, "experimental_kv_cache_dtype", None) == "bfp_bf8"
            else self.kv_cache_dtype
        )

        self.block_size = self.cache_config.block_size
        self.max_model_len = self.model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)

        # Prefill request-count bucketing bounds (decode always uses max_num_reqs).
        self.min_num_reqs = getattr(tt, "min_num_seqs", None) or self.max_num_reqs
        self.max_prefill_num_reqs = (
            getattr(tt, "max_prefill_num_seqs", None) or self.max_num_reqs
        )

        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        min_context_len = getattr(tt, "min_context_len", 32)
        if max_num_batched_tokens < min_context_len:
            min_context_len = max_num_batched_tokens
        self.prefill_chunk_budget = min(
            getattr(
                self.scheduler_config, "tt_prefill_chunk_size", max_num_batched_tokens
            )
            or max_num_batched_tokens,
            self.max_model_len,
        )
        self.num_tokens_paddings = _get_token_paddings(
            min_context_len, self.prefill_chunk_budget
        )
        if getattr(tt, "decode_only", False):
            self.num_tokens_paddings = [1]
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model dims.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention"
        )
        self.num_query_heads = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.vocab_size = self.model_config.get_vocab_size()
        if self.lora_config is not None:
            # lora_extra_vocab_size was removed in vllm 0.20.2; add it only when
            # the installed LoRAConfig still exposes it.
            self.vocab_size += getattr(self.lora_config, "lora_extra_vocab_size", 0)
        # M-RoPE models feed 3D [3, reqs, tokens] position ids to the model.
        self.uses_mrope = self.model_config.uses_mrope
        self.supports_mm_inputs = bool(
            getattr(self.model_config, "is_multimodal_model", False)
        )
        self.mm_budget = None  # set when multimodal profiling lands

        # SMEM row caps consumed by _select_batch.
        self.num_reqs_max_model_len = min(
            TTAttentionBackend.get_max_num_seqs(self.max_model_len, self.block_size),
            self.max_num_reqs,
        )

        # Driver / graph knobs read later.
        self.use_flat_model_io = bool(getattr(tt, "flat_model_io", False))
        self.cpu_sampling = bool(getattr(tt, "cpu_sampling", False))
        self.enable_decode_fused_graphs = bool(
            getattr(tt, "enable_decode_fused_graphs", False)
        )
        self.sampling_device = torch.device("cpu") if self.cpu_sampling else self.device

        # Split v2 state.
        self.req_states = TTRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_spec_tokens,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.sampling_states = TTSamplingStates(
            max_num_reqs=self.max_num_reqs, vocab_size=self.vocab_size
        )
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[self.block_size],
            kernel_block_sizes=[self.block_size],
        )
        self.input_buffers = TTInputBuffers(
            self.max_num_reqs, self.max_num_tokens, self.device
        )

        self.encoder_cache: dict = {}
        self.num_prompt_logprobs: dict = {}
        # Partial prompt-logprobs results for chunked prefills, keyed by req_id.
        self.in_progress_prompt_logprobs: dict = {}
        # Active LoRA request per stable slot (None/absent -> base model).
        self.lora_requests_by_slot: dict = {}
        # Multimodal feature list per stable slot (for the encoder run + gather).
        self.mm_features_by_slot: dict = {}

        # Structured-output (grammar) buffers. Filled per pass in
        # prepare_structured_decoding_input, keyed by batch position.
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
        )
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1), dtype=torch.bool, device="cpu"
        )
        # Power-of-2 masks for on-device bitmask unpacking via bitwise_and
        # (bitwise_right_shift is unsupported on TT). uint32->int32 view avoids
        # overflow at bit 31.
        _bitmask_values = np.array([1 << i for i in range(32)], dtype=np.uint32).view(
            np.int32
        )
        self.structured_decode_bitmasks = torch.from_numpy(_bitmask_values.copy())
        self.kv_caches: list = []
        self.kv_cache_config = None
        # layer_name -> target layer for cross-layer KV sharing (get_kv_cache_spec).
        self.shared_kv_cache_layers: dict = {}
        # Per-step handoff between the two-phase execute_model / sample_tokens.
        self.scheduler_output = None

        # Filled by load_model.
        self.model = None
        self.model_state = None
        self.sampler = None
        self._attention_layer_names: tuple = ()
        self.attention_layer_names: tuple = ()

    def _build_device_mesh(self) -> None:
        """Select the parallelism mode and build the SPMD device mesh.

        Ported from the v1 runner: disables TP/DP that the available device count
        can't support, derives the ``(batch, model)`` mesh via
        ``determine_mesh_shape``, sets ``dp_size`` (>1 only in DP modes), and
        rounds ``max_num_reqs`` up to a ``dp_size`` multiple. No-ops to the
        single-device path (mesh stays None) when both flags end up disabled.
        """
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr

        from .vllm_utils import determine_mesh_shape

        num_devices = xr.global_runtime_device_count()
        if self.enable_tensor_parallel and num_devices == 1:
            logger.warning("Tensor parallel needs >1 device; found 1. Disabling TP.")
            self.enable_tensor_parallel = False
        if self.enable_data_parallel and (self.max_num_reqs <= 1 or num_devices == 1):
            logger.warning(
                "Data parallel needs >1 device and max_num_seqs > 1. Disabling DP."
            )
            self.enable_data_parallel = False

        if self.enable_data_parallel and self.enable_tensor_parallel:
            self.parallel_mode = ParallelismMode.DATA_TENSOR_PARALLEL
        elif self.enable_data_parallel:
            self.parallel_mode = ParallelismMode.DATA_PARALLEL_ONLY
        elif self.enable_tensor_parallel:
            # An explicit 2D mesh_shape (no size-1 axis) forces TP-2D even when
            # use_2d_mesh is unset, matching the mesh determine_mesh_shape builds.
            explicit_2d_mesh = (
                self.tt_config.mesh_shape is not None
                and 1 not in self.tt_config.mesh_shape
            )
            self.parallel_mode = (
                ParallelismMode.TENSOR_PARALLEL_ONLY_2D
                if (self.use_2d_mesh or explicit_2d_mesh)
                else ParallelismMode.TENSOR_PARALLEL_ONLY_1D
            )
        else:
            self.parallel_mode = ParallelismMode.DISABLED
            return

        mesh_shape = determine_mesh_shape(
            num_devices, self.parallel_mode, self.tt_config.mesh_shape
        )
        device_ids = np.array(range(num_devices))
        self.mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))
        # mesh_shape[0] ("batch" axis) is a DP replica count only in DP modes; in
        # pure-TP the batch axis is a TP axis, so dp_size stays 1.
        if self.parallel_mode in (
            ParallelismMode.DATA_PARALLEL_ONLY,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ):
            self.dp_size = mesh_shape[0]
        self.use_2d_mesh = 1 not in mesh_shape
        xs.set_global_mesh(self.mesh)

        if self.enable_data_parallel and self.dp_size > 1:
            remainder = self.max_num_reqs % self.dp_size
            if remainder != 0:
                adjusted = self.max_num_reqs + self.dp_size - remainder
                logger.warning(
                    "Data parallel requires max_num_reqs divisible by dp_size; "
                    "adjusting from %d to %d.",
                    self.max_num_reqs,
                    adjusted,
                )
                self.max_num_reqs = adjusted

    def load_model(self) -> None:
        """Load, place, and compile the model; build ModelState + sampler.

        Under TP the weights are sharded across the mesh (via ``shard_model``)
        and the embedding load is wrapped in the vocab TP-rank patch. LoRA and
        per-tensor weight-dtype overrides are still deferred. Needs a real model
        + loader, so it is validated at engine stand-up.
        """
        from contextlib import nullcontext

        from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.model_loader import get_model_loader

        from .model_state import TTModelState
        from .overrides import repair_stale_moe_closures, replace_modules
        from .rejection_sampler import RejectionSampler
        from .sampler import Sampler

        logger.info("Loading model %s ...", self.model_config.model)
        loader = get_model_loader(self.load_config)

        # Under a layer override the checkpoint still holds every layer's
        # weights; filter the loader so it only feeds the built layers.
        if self._original_num_layers is not None:
            original_get_all_weights = loader.get_all_weights

            def filtered_get_all_weights(model_config, model):
                return self._filter_weights_for_layer_override(
                    original_get_all_weights(model_config, model)
                )

            loader.get_all_weights = filtered_get_all_weights

        # xm's rank assignment differs from gloo's; the embedding all-gather
        # ordering depends on the xm rank, so patch it only for the weight load.
        vocab_rank_patch = nullcontext()
        if self.enable_tensor_parallel:
            from unittest.mock import patch

            import torch_xla.runtime as xr

            vocab_rank_patch = patch(
                "vllm.model_executor.layers.vocab_parallel_embedding."
                "get_tensor_model_parallel_rank",
                return_value=xr.global_ordinal(),
            )

        with set_current_vllm_config(self.vllm_config), vocab_rank_patch:
            model = loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            ).eval()
        replace_modules(model)
        self.model = model.to(self.device)

        # Repair MoE routing closures that captured CPU tensors before to(device).
        repair_stale_moe_closures(self.model)

        # Wrap with LoRA layers after the model is on device, before compile
        # (ported from the v1 runner). replace_set_lora adds the torch_xla.sync
        # the integer LoRA index needs.
        if self.lora_config is not None:
            logger.info("Loading LoRA model...")
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
            replace_set_lora(self.model)

        # Shard weights across the mesh before compile so the annotations are
        # captured in the traced graph. Pure-TP always shards on the batch axis
        # (it is itself a TP axis there); DP+TP honours the config knob.
        if self.enable_tensor_parallel:
            shard_on_batch_axis = (
                self.tt_config.shard_weights_on_batch_axis
                if self.parallel_mode == ParallelismMode.DATA_TENSOR_PARALLEL
                else True
            )
            shard_model(self.model, self.mesh, shard_on_batch_axis)

        # MM models nest lm_head, so walk the tree for any ParallelLMHead.
        self.is_sharded_compute_logits = self.enable_tensor_parallel and any(
            isinstance(m, ParallelLMHead) for m in self.model.modules()
        )

        self.model.compile(backend="tt", dynamic=False)
        logger.info("Model loaded and registered for tt compilation.")
        self.sampler = Sampler()
        self.rejection_sampler = RejectionSampler(self.sampler)

        encoder_cache = self.encoder_cache if self.supports_mm_inputs else None
        self.model_state = TTModelState(
            self.vllm_config, self.model, encoder_cache, self.device
        )

        # Cache the attention layer names for the per-step prepare_attn fan-out.
        self._attention_layer_names = tuple(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        self.attention_layer_names = self._attention_layer_names

    def _filter_weights_for_layer_override(self, weights_iterator):
        """Drop checkpoint weights for layers beyond the override target."""
        if self._original_num_layers is None or self._target_num_layers is None:
            yield from weights_iterator
            return
        for weight_name, weight_tensor in weights_iterator:
            skip = False
            if "layers." in weight_name:
                parts = weight_name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            skip = int(parts[i + 1]) >= self._target_num_layers
                        except ValueError:
                            skip = False
                        break
            if not skip:
                yield weight_name, weight_tensor

    def _remove_request(self, req_id: str) -> None:
        """Free a request's slot across every per-slot table. Idempotent."""
        slot = self.req_states.req_id_to_index.get(req_id)
        if slot is None:
            return
        self.req_states.remove_request(req_id)
        self.sampling_states.remove_request(slot)
        self.block_table.clear_row(slot)
        self.num_prompt_logprobs.pop(req_id, None)
        self.lora_requests_by_slot.pop(slot, None)
        self.mm_features_by_slot.pop(slot, None)

    def finish_requests(self, scheduler_output: "SchedulerOutput") -> None:
        """Remove finished and preempted requests, freeing their slots.

        Preempted requests are dropped (not kept at their slot as in v1): the
        scheduler resends them through ``scheduled_new_reqs`` when resumed.
        """
        for req_id in scheduler_output.finished_req_ids:
            self._remove_request(req_id)
        for req_id in scheduler_output.preempted_req_ids:
            self._remove_request(req_id)
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

    def add_requests(self, scheduler_output: "SchedulerOutput") -> None:
        """Register newly scheduled requests into the per-slot tables."""
        new_reqs = scheduler_output.scheduled_new_reqs
        for new in new_reqs:
            sampling_params = new.sampling_params
            assert sampling_params is not None, "Pooling not supported in the v2 runner"

            req_id = new.req_id
            # A re-added id (streaming abort+resubmit) must clear its stale slot.
            self._remove_request(req_id)

            prompt_len = len(new.prompt_token_ids)
            # prefill_token_ids is the full known prefix (prompt + already-computed
            # tokens for resumed/PD requests); prompt_token_ids for a fresh req.
            all_token_ids = (
                new.prefill_token_ids
                if new.prefill_token_ids is not None
                else new.prompt_token_ids
            )
            self.req_states.add_request(
                req_id, prompt_len, all_token_ids, new.num_computed_tokens
            )
            slot = self.req_states.req_id_to_index[req_id]

            # Seeded requests carry their own generator; the rest use global RNG.
            generator = None
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(sampling_params.seed)
            self.sampling_states.add_request(slot, sampling_params, generator)

            self.block_table.add_row(new.block_ids, slot)

            if new.lora_request is not None:
                self.lora_requests_by_slot[slot] = new.lora_request

            if self.supports_mm_inputs and new.mm_features:
                self.mm_features_by_slot[slot] = new.mm_features

            if self.model_state is not None:
                # No-op for the common text path (rope_state is None).
                self.model_state.add_request(slot, new)

            if sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

        if new_reqs:
            # No-ops under the numpy substitution, kept for interface parity.
            self.req_states.apply_staged_writes()
            if self.model_state is not None:
                self.model_state.apply_staged_writes()

    def update_requests(self, scheduler_output: "SchedulerOutput") -> None:
        """Advance computed-token counts and append new blocks for running reqs."""
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            slot = self.req_states.req_id_to_index[req_id]
            num_computed = cached.num_computed_tokens[i]
            self.req_states.num_computed_tokens[slot] = num_computed
            # Clamp prefill progress to the prefill length (upstream update_requests).
            self.req_states.num_computed_prefill_tokens[slot] = min(
                num_computed, int(self.req_states.prefill_len[slot])
            )
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                self.block_table.append_row(new_block_ids, slot)

    def _order_scheduled_reqs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order this step's scheduled requests decodes-first (by token count).

        Returns parallel ``(slots, num_scheduled_tokens)`` arrays. Sorting by
        ascending scheduled-token count (upstream v2 convention) groups the
        1-token decodes ahead of the prefills so the multi-pass loop can emit
        pure decode passes (dispatched to the decode graph) before prefills.
        """
        sched = scheduler_output.num_scheduled_tokens
        if self.dp_size > 1:
            # DP: a request's batch position picks its DP replica, so it must equal
            # the request's stable slot for its whole life (KV is written at prefill
            # and read at decode on the same replica). Return the FULL replica grid
            # -- every slot 0..max_num_reqs-1, position b == slot b -- with per-slot
            # token counts (0 for inactive or unscheduled slots); _select_batch runs
            # it as one full-width pass. Listing only active slots (or ranking them)
            # would shift a lone prefilling request to position 0 and read its KV off
            # the wrong replica at decode.
            idx_to_req = self.req_states.index_to_req_id
            slots = np.arange(self.max_num_reqs, dtype=np.int32)
            ntoks_np = np.array(
                [sched.get(idx_to_req.get(s), 0) for s in range(self.max_num_reqs)],
                dtype=np.int32,
            )
            return slots, ntoks_np

        slots: list[int] = []
        ntoks: list[int] = []
        for req_id, n in sched.items():
            slot = self.req_states.req_id_to_index.get(req_id)
            if slot is None:
                continue
            slots.append(slot)
            ntoks.append(n)
        ntoks_np = np.array(ntoks, dtype=np.int32)
        order = np.argsort(ntoks_np, kind="stable")
        return np.array(slots, dtype=np.int32)[order], ntoks_np[order]

    def _select_batch(
        self,
        ordered_slots: np.ndarray,
        ordered_num_tokens: np.ndarray,
        start_index: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """Pick the sub-batch for one pass, applying the SMEM row caps.

        Mirrors the v1 fork's per-pass clamping over the decode-first ordering.
        The row cap is always the max-model-len cap; prefill passes are
        additionally capped at ``max_prefill_num_reqs`` and the multi-pass loop
        picks up the rest. Returns ``(idx_mapping,
        num_scheduled_tokens, target_num_reqs, padded_query_len, end_index)``.
        """
        if self.dp_size > 1:
            # DP grid is one full-width pass; positions already map to replicas
            # (see _order_scheduled_reqs), so no SMEM re-clamp or reorder here.
            max_scheduled = max(int(ordered_num_tokens.max()), 1)
            padded_query_len = _get_padded_token_len(
                self.num_tokens_paddings, max_scheduled
            )
            return (
                ordered_slots,
                ordered_num_tokens,
                self.max_num_reqs,
                padded_query_len,
                len(ordered_slots),
            )

        row_cap = self.num_reqs_max_model_len

        end_index = min(len(ordered_slots), start_index + row_cap)
        num_scheduled = ordered_num_tokens[start_index:end_index]

        # Prefill pass over the cap -> trim; the next pass takes the remainder.
        if (
            len(num_scheduled) > self.max_prefill_num_reqs
            and int(num_scheduled.max()) > 1
        ):
            end_index = start_index + self.max_prefill_num_reqs
            num_scheduled = ordered_num_tokens[start_index:end_index]

        idx_mapping = ordered_slots[start_index:end_index]
        max_scheduled = int(num_scheduled.max())

        if max_scheduled == 1:
            # Decode always runs at the max request bucket.
            target_num_reqs = self.max_num_reqs
        else:
            actual = len(num_scheduled)
            target_num_reqs = (
                self.min_num_reqs
                if actual <= self.min_num_reqs
                else self.max_prefill_num_reqs
            )

        padded_query_len = _get_padded_token_len(
            self.num_tokens_paddings, max_scheduled
        )
        return idx_mapping, num_scheduled, target_num_reqs, padded_query_len, end_index

    def _prepare_input_tokens(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        target_num_reqs: int,
        padded_query_len: int,
        spec_decode_metadata=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Host substitute for the v2 Triton input-prep kernels.

        Builds 2D input_ids/positions [target_num_reqs, padded_query_len] plus flat
        query_start_loc/seq_lens from TTRequestState, indexed by idx_mapping_np[b]
        -> stable slot. Prefill and decode share one gather
        (all_token_ids[computed:computed+n]); the sampled token was already
        appended by the writeback, and staged drafts sit right after it. Padding
        stays zero.

        logits_indices is flat [target_num_reqs] (one logit at each request's last
        scheduled token), or packed [target_num_reqs, num_spec_tokens+1] under spec
        decode so draft and bonus logits come out of a single pass.
        """
        num_reqs = len(idx_mapping_np)
        rs = self.req_states

        input_ids = np.zeros((target_num_reqs, padded_query_len), dtype=np.int32)
        positions = np.zeros((target_num_reqs, padded_query_len), dtype=np.int32)
        seq_lens = np.zeros(target_num_reqs, dtype=np.int32)
        logits_indices = np.zeros(target_num_reqs, dtype=np.int32)

        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            n = int(num_scheduled_tokens[b])
            computed = int(rs.num_computed_tokens[slot])
            # Same gather for prefill and decode: the scheduled tokens for this
            # step are all_token_ids[computed : computed + n].
            input_ids[b, :n] = rs.all_token_ids[slot, computed : computed + n]
            positions[b, :n] = np.arange(n, dtype=np.int32) + computed
            seq_lens[b] = computed + n
            # One logit per request, at its last scheduled token (within-row).
            logits_indices[b] = n - 1

        if spec_decode_metadata is not None:
            # Packed [reqs, spec_width]: draft rows index the first draft_len
            # positions, every remaining column points at the bonus logit.
            spec_width = self.num_spec_tokens + 1
            packed = np.zeros((target_num_reqs, spec_width), dtype=np.int32)
            for b in range(num_reqs):
                bonus_idx = int(num_scheduled_tokens[b]) - 1
                packed[b, :] = bonus_idx
                draft_len = spec_decode_metadata.num_draft_tokens[b]
                if draft_len:
                    packed[b, :draft_len] = np.arange(draft_len, dtype=np.int32)
            logits_indices = packed

        query_start_loc = np.zeros(target_num_reqs + 1, dtype=np.int32)
        np.cumsum(num_scheduled_tokens, out=query_start_loc[1 : num_reqs + 1])
        # Non-decreasing pad tail (matches TTInputBatch.make_dummy).
        query_start_loc[num_reqs + 1 :] = query_start_loc[num_reqs]

        if self.uses_mrope:
            # M-RoPE position ids are 3D [3, reqs, tokens]. For text-only inputs
            # every plane is identical, making it equivalent to 1D RoPE (see
            # https://arxiv.org/abs/2409.12191). Mirrors the v1 runner.
            positions = np.broadcast_to(
                positions, (3, target_num_reqs, padded_query_len)
            ).copy()

        return input_ids, positions, query_start_loc, seq_lens, logits_indices

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput") -> None:
        """Run the multimodal encoder for this step and cache outputs by mm_hash.

        Ported from the v1 runner (_execute_mm_encoder). Reads mm features from
        the per-slot table instead of self.requests, and assumes whole mm items
        (no scatter_mm_placeholders dynamism), like the v1 runner.
        """
        from typing import cast

        import torch_xla
        from vllm.model_executor.models.interfaces import SupportsMultiModal
        from vllm.multimodal.utils import group_mm_kwargs_by_modality
        from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs

        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        mm_kwargs = []
        mm_hashes_pos = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            slot = self.req_states.req_id_to_index[req_id]
            mm_features = self.mm_features_by_slot.get(slot, [])
            for mm_input_id in encoder_input_ids:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_hashes_pos.append((mm_feature.identifier, mm_feature.mm_position))

        model = cast(SupportsMultiModal, self.model)
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs, device=self.device, pin_memory=False
        ):
            torch_xla.sync(wait=False)
            curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)
            torch_xla.sync(wait=False)
            sanity_check_mm_encoder_outputs(
                curr_group_outputs, expected_num_items=num_items
            )
            for output in curr_group_outputs:
                encoder_outputs.append(output)

        for (mm_hash, _pos), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = output

    def _gather_mm_embeddings(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        padded_query_len: int,
    ):
        """Build the per-pass mm-embed mask + flat scatter indices (v1
        _gather_mm_embeddings port). The mask is [target_num_reqs,
        padded_query_len] to match input_ids (each request's scheduled tokens
        start at column 0); mm_indices are its row-major True positions, consumed
        by get_mm_embeddings' index_copy merge.
        """
        target_num_reqs = len(idx_mapping_np)
        is_mm_embed = torch.zeros(
            (target_num_reqs, padded_query_len), dtype=torch.bool, device="cpu"
        )
        mm_embeds = []

        for b in range(target_num_reqs):
            n = int(num_scheduled_tokens[b])
            if n == 0:
                continue
            slot = int(idx_mapping_np[b])
            num_computed_tokens = int(self.req_states.num_computed_tokens[slot])
            for mm_feature in self.mm_features_by_slot.get(slot, []):
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                if start_pos >= num_computed_tokens + n:
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(num_computed_tokens - start_pos + n, num_encoder_tokens)
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                if curr_embeds_start == curr_embeds_end:
                    continue

                encoder_output = self.encoder_cache.get(mm_feature.identifier)
                assert (
                    encoder_output is not None
                ), f"Encoder cache miss for {mm_feature.identifier}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                # Column of this request's row where the placeholder begins
                # (input_ids stores scheduled tokens from column 0).
                col_base = start_pos - num_computed_tokens
                if is_embed is None:
                    is_mm_embed[b, col_base + start_idx : col_base + end_idx] = True
                else:
                    is_mm_embed[
                        b, col_base + start_idx : col_base + end_idx
                    ] |= is_embed
                mm_embeds.append(mm_embeds_item)

        mm_indices = is_mm_embed.flatten().nonzero(as_tuple=True)[0].to(self.device)
        is_mm_embed = is_mm_embed.to(self.device)
        return mm_embeds, is_mm_embed, mm_indices

    def _apply_scheduled_drafts(self, scheduler_output: "SchedulerOutput") -> None:
        """Stage this step's scheduled draft tokens into the slot table.

        The scheduler already counts drafts in num_scheduled_tokens, so the input
        gather reads them straight out of all_token_ids once staged here.
        """
        if not self.num_spec_tokens:
            return
        rs = self.req_states
        scheduled = (
            getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        )
        for req_id, slot in rs.req_id_to_index.items():
            drafts = scheduled.get(req_id)
            if drafts:
                rs.set_draft_tokens(slot, [int(t) for t in drafts])
            else:
                rs.clear_draft_tokens(slot)

    def _build_spec_decode_metadata(self, idx_mapping_np: np.ndarray):
        """Per-pass SpecDecodeMetadata over the packed [reqs, spec_width] logits.

        Returns None when spec decode is off or no request in this pass carries
        drafts, which keeps the non-spec path on the flat one-logit-per-request
        layout. Mirrors the v1 builder.
        """
        if not self.num_spec_tokens:
            return None
        rs = self.req_states
        num_draft = [int(rs.num_draft_tokens[int(s)]) for s in idx_mapping_np]
        if not any(num_draft):
            return None

        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        spec_width = self.num_spec_tokens + 1
        flat: list[int] = []
        target_logits_indices: list[int] = []
        bonus_logits_indices: list[int] = []
        for b, draft_len in enumerate(num_draft):
            base = b * spec_width
            slot = int(idx_mapping_np[b])
            flat.extend(int(t) for t in rs.draft_tokens[slot, :draft_len])
            target_logits_indices.extend(range(base, base + draft_len))
            bonus_logits_indices.append(base + spec_width - 1)

        dev = self.device
        cu_num_draft = np.cumsum(np.array(num_draft, dtype=np.int32), dtype=np.int32)
        cu_num_sampled = np.cumsum(
            np.array([d + 1 for d in num_draft], dtype=np.int32), dtype=np.int32
        )
        return SpecDecodeMetadata(
            draft_token_ids=torch.tensor(flat, dtype=torch.int32).to(dev),
            num_draft_tokens=num_draft,
            cu_num_draft_tokens=torch.from_numpy(cu_num_draft).to(dev),
            cu_num_sampled_tokens=torch.from_numpy(cu_num_sampled).to(dev),
            target_logits_indices=torch.tensor(
                target_logits_indices, dtype=torch.int32
            ).to(dev),
            bonus_logits_indices=torch.tensor(
                bonus_logits_indices, dtype=torch.int32
            ).to(dev),
            logits_indices=torch.arange(
                len(num_draft) * spec_width, dtype=torch.int32
            ).to(dev),
        )

    def postprocess(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """Write sampled tokens back into TTRequestState (post_update substitute).

        A request emits a token only once past its prefill (seq_len >= prefill_len);
        still-prefilling rows are discarded. The token lands at total_len (== seq_len
        then) so the next step's gather reads it; num_computed is advanced by the
        scheduler via update_requests, not here. Returns per-batch sampled ids with
        discarded rows emptied.
        """
        num_reqs = len(idx_mapping_np)
        rs = self.req_states
        valid: list[list[int]] = [list(row) for row in sampled_token_ids[:num_reqs]]

        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            if int(num_scheduled_tokens[b]) == 0:
                # DP padding row (unscheduled this step): emits no token.
                valid[b] = []
                continue
            seq_len = int(rs.num_computed_tokens[slot]) + int(num_scheduled_tokens[b])
            if seq_len < int(rs.prefill_len[slot]):
                # Still prefilling: ignore the sampled token from this partial req.
                valid[b] = []
                continue
            # Spec decode accepts 1..num_spec_tokens+1 tokens per request; the
            # non-spec path always yields exactly one.
            tokens = [int(t) for t in valid[b]]
            if not tokens:
                continue
            pos = int(rs.total_len[slot])
            rs.all_token_ids[slot, pos : pos + len(tokens)] = tokens
            rs.total_len[slot] = pos + len(tokens)
            rs.last_sampled_tokens[slot, 0] = tokens[-1]

        return valid

    def _prepare_attn_tensors(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        seq_lens: np.ndarray,
        target_num_reqs: int,
        num_blocks_per_req: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Host substitute for the v2 block-table / slot-mapping kernels.

        Builds the paged-attention tensors TTModelState.prepare_attn packages into
        a TTMetadata: read-path page_table (gathered in batch order via the slot
        mapping), write-path fill_page_table (prefix rolled), and cache_position
        (seq_lens - 1). Padding rows are null (page_table 0, cache_position -1).
        """
        num_reqs = len(idx_mapping_np)
        block_table_cpu = self.block_table[0].get_cpu_tensor()
        if hasattr(block_table_cpu, "numpy"):
            block_table_cpu = block_table_cpu.numpy()

        page_table = np.zeros((target_num_reqs, num_blocks_per_req), dtype=np.int32)
        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            page_table[b, :] = block_table_cpu[slot, :num_blocks_per_req]

        # Decode/write position per user; -1 for padding rows.
        cache_position = np.full(target_num_reqs, -1, dtype=np.int32)
        cache_position[:num_reqs] = seq_lens[:num_reqs] - 1

        # Prefix caching: roll each row so paged_fill_cache writes the suffix
        # blocks instead of overwriting shared prefix blocks. Per-row because
        # prefix lengths differ.
        offsets = np.zeros(num_reqs, dtype=np.int64)
        for b in range(num_reqs):
            slot = int(idx_mapping_np[b])
            offsets[b] = (
                int(self.req_states.num_computed_tokens[slot]) // self.block_size
            )
        if np.any(offsets > 0):
            fill_page_table = page_table.copy()
            for b in range(num_reqs):
                if offsets[b] > 0:
                    fill_page_table[b] = np.roll(page_table[b], -int(offsets[b]))
        else:
            fill_page_table = page_table

        # A zero-scheduled (already-prefilled, re-batched) row would clobber its
        # real KV with padding; redirect its fill to the null block. The read
        # path keeps the real blocks.
        zero_rows = np.nonzero(num_scheduled_tokens[:num_reqs] == 0)[0]
        if len(zero_rows) > 0:
            if fill_page_table is page_table:
                fill_page_table = page_table.copy()
            fill_page_table[zero_rows, :] = 0

        return page_table, fill_page_table, cache_position

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ):
        """Phase 1 of the step: apply the scheduler delta via the lifecycle and
        stash the step. Returns None (the forward + sampling run in sample_tokens),
        or an empty output when nothing is scheduled.
        """
        assert (
            self.scheduler_output is None
        ), "execute_model called before sample_tokens consumed the prior step"
        # TT compiles one graph per (bucket) shape; keep dynamic shapes off so a
        # new shape recompiles rather than silently falling back.
        torch._dynamo.config.dynamic_shapes = False

        self.finish_requests(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)

        if scheduler_output.total_num_scheduled_tokens == 0:
            from vllm.distributed.kv_transfer import has_kv_transfer_group
            from vllm.v1.worker.gpu_model_runner import EMPTY_MODEL_RUNNER_OUTPUT

            if has_kv_transfer_group():
                # Nothing to run, but the connector still has sends/recvs to drive.
                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
            return EMPTY_MODEL_RUNNER_OUTPUT

        self._apply_scheduled_drafts(scheduler_output)

        if self.supports_mm_inputs and scheduler_output.scheduled_encoder_inputs:
            # Run the vision/multimodal encoder once per step; outputs are cached
            # by mm_hash and consumed per pass in _run_model_pass. Text-only
            # requests skip this (no scheduled encoder inputs).
            self._execute_mm_encoder(scheduler_output)

        self.scheduler_output = scheduler_output
        return None

    def sample_tokens(self, grammar_output):
        """Phase 2 of the step: run the decode-first, SMEM-clamped multi-pass batch
        loop around the _run_model_pass hardware leaf, then assemble the
        ModelRunnerOutput. A step may span several passes (start_index advances).
        """
        if self.scheduler_output is None:
            # PP non-final rank / nothing stashed: output is unused.
            return None
        scheduler_output = self.scheduler_output
        self.scheduler_output = None

        ordered_slots, ordered_num_tokens = self._order_scheduled_reqs(scheduler_output)

        out_req_ids: list[str] = []
        out_sampled: list[list[int]] = []
        # Capture per-request prompt hidden states (keyed by slot) when any
        # scheduled request wants prompt logprobs.
        want_prompt_hs = bool(self.num_prompt_logprobs)
        prompt_lp_hs: dict[int, torch.Tensor] = {}

        kv_connector_output = None
        with contextlib.ExitStack() as stack:
            if self._has_kv_transfer_group():
                # Once per step, not per pass: get_finished() reports a transfer
                # only on its first call, so per-pass entry would drop finished
                # send/recv ids. start_load_kv needs an active forward context,
                # so push a token-less one as kv_connector_no_forward does.
                from vllm.forward_context import set_forward_context

                stack.enter_context(set_forward_context(None, self.vllm_config))
                kv_connector_output = stack.enter_context(
                    self.maybe_get_kv_connector_output(scheduler_output)
                )
            self._run_pass_loop(
                ordered_slots,
                ordered_num_tokens,
                grammar_output,
                want_prompt_hs,
                out_req_ids,
                out_sampled,
                prompt_lp_hs,
            )

        self.propose_draft_token_ids(out_req_ids, out_sampled)

        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            prompt_lp_hs, scheduler_output
        )

        from vllm.v1.outputs import ModelRunnerOutput

        return ModelRunnerOutput(
            req_ids=out_req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(out_req_ids)},
            sampled_token_ids=out_sampled,
            prompt_logprobs_dict=prompt_logprobs_dict,
            kv_connector_output=kv_connector_output,
        )

    def propose_draft_token_ids(
        self, out_req_ids: list[str], out_sampled: list[list[int]]
    ) -> None:
        """Run the ngram proposer over this step's accepted tokens.

        Drafts are cached for the scheduler to pick up via take_draft_token_ids.
        Host-only (the proposer is CPU/numpy), so nothing is compiled here.
        """
        self._draft_token_req_ids = []
        self._draft_token_ids = []
        if not self.drafter or not self.num_spec_tokens or not out_req_ids:
            return
        if not any(out_sampled):
            return

        rs = self.req_states
        # num_tokens_no_spec: committed context length, drafts excluded. total_len
        # already excludes staged drafts and includes this step's accepted tokens.
        slots = [rs.req_id_to_index[rid] for rid in out_req_ids]
        num_tokens_no_spec = np.array(
            [int(rs.total_len[s]) for s in slots], dtype=np.int32
        )
        token_ids_cpu = np.zeros((len(slots), self.max_model_len), dtype=np.int32)
        for i, slot in enumerate(slots):
            n = int(rs.total_len[slot])
            token_ids_cpu[i, :n] = rs.all_token_ids[slot, :n]

        drafts = self.drafter.propose(
            [list(row) for row in out_sampled], num_tokens_no_spec, token_ids_cpu
        )
        for req_id, draft in zip(out_req_ids, drafts):
            if draft:
                self._draft_token_req_ids.append(req_id)
                self._draft_token_ids.append(draft)

    def take_draft_token_ids(self):
        """Hand this step's cached drafts to the scheduler (once)."""
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        from vllm.v1.outputs import DraftTokenIds

        out = DraftTokenIds(self._draft_token_req_ids, self._draft_token_ids)
        self._draft_token_ids = None
        self._draft_token_req_ids = None
        return out

    @staticmethod
    def _has_kv_transfer_group() -> bool:
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        return has_kv_transfer_group()

    def _run_pass_loop(
        self,
        ordered_slots,
        ordered_num_tokens,
        grammar_output,
        want_prompt_hs: bool,
        out_req_ids: list,
        out_sampled: list,
        prompt_lp_hs: dict,
    ) -> None:
        """Decode-first multi-pass loop; appends into the caller's output lists."""
        start_index = 0
        while start_index < len(ordered_slots):
            (
                idx_mapping,
                num_scheduled,
                target_num_reqs,
                padded_query_len,
                end_index,
            ) = self._select_batch(ordered_slots, ordered_num_tokens, start_index)

            spec_md = self._build_spec_decode_metadata(idx_mapping)

            input_ids, positions, _query_start_loc, seq_lens, logits_indices = (
                self._prepare_input_tokens(
                    idx_mapping,
                    num_scheduled,
                    target_num_reqs,
                    padded_query_len,
                    spec_md,
                )
            )
            page_table, fill_page_table, cache_position = self._prepare_attn_tensors(
                idx_mapping,
                num_scheduled,
                seq_lens,
                target_num_reqs,
                self.max_num_blocks_per_req,
            )

            pass_result = self._run_model_pass(
                idx_mapping,
                num_scheduled,
                target_num_reqs,
                padded_query_len,
                input_ids,
                positions,
                logits_indices,
                page_table,
                fill_page_table,
                cache_position,
                want_prompt_hs,
                grammar_output,
                spec_md,
            )
            if want_prompt_hs:
                sampled, hidden_states = pass_result
                # Copy only the rows for requests that need prompt logprobs
                # (keyed by stable slot), so the big hidden tensor never fully
                # leaves the device.
                for b in range(len(idx_mapping)):
                    slot = int(idx_mapping[b])
                    req_id = self.req_states.index_to_req_id.get(slot)
                    if req_id in self.num_prompt_logprobs:
                        prompt_lp_hs[slot] = hidden_states[b].cpu()
            else:
                sampled = pass_result

            valid = self.postprocess(idx_mapping, num_scheduled, sampled)
            for b in range(len(idx_mapping)):
                if int(num_scheduled[b]) == 0:
                    # DP padding row: not scheduled this step, don't report it.
                    continue
                slot = int(idx_mapping[b])
                out_req_ids.append(self.req_states.index_to_req_id[slot])
                out_sampled.append(valid[b])

            start_index = end_index

    def _run_model_pass(
        self,
        idx_mapping_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        target_num_reqs: int,
        padded_query_len: int,
        input_ids: np.ndarray,
        positions: np.ndarray,
        logits_indices: np.ndarray,
        page_table: np.ndarray,
        fill_page_table: np.ndarray,
        cache_position: np.ndarray,
        want_prompt_hs: bool = False,
        grammar_output=None,
        spec_decode_metadata=None,
    ):
        """Run one compiled forward+sample pass for the selected sub-batch.

        Copies host arrays to device, builds attention metadata (prepare_attn) +
        sampling metadata (from_v2_states), runs the compiled graph, and returns
        the batch-ordered sampled token ids. Common text path only. When
        ``want_prompt_hs`` is set, also returns the per-batch pre-selection hidden
        states (device tensor [target_num_reqs, padded_query_len, H]) for prompt
        logprobs.
        """
        from types import SimpleNamespace

        from .metadata import XLASupportedSamplingMetadata
        from .rejection_sampler import RejectionSampler

        dev = self.device
        input_ids_dev = torch.from_numpy(input_ids).to(dev)
        positions_dev = torch.from_numpy(positions).to(dev)
        page_table_dev = torch.from_numpy(page_table).to(dev)
        fill_page_table_dev = torch.from_numpy(fill_page_table).to(dev)
        cache_position_dev = torch.from_numpy(cache_position).to(dev)
        logits_indices_dev = torch.from_numpy(logits_indices).to(dev)
        batch_idx_dev = torch.from_numpy(np.arange(target_num_reqs, dtype=np.int32)).to(
            dev
        )

        # Pin input shardings eagerly (not inside the compiled graph, whose
        # dynamo trace can't run safe_mark_sharding's replicate-fallback logging).
        self._pin_input_shardings(input_ids_dev, positions_dev, None)

        if self.parallel_mode in (
            ParallelismMode.DATA_PARALLEL_ONLY,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ):
            # These share the DP-sharded K/V input's per-device leading dim;
            # batch_idx feeds paged_fill_cache, whose verifier requires dim0 to
            # equal the per-device batch.
            safe_mark_sharding(page_table_dev, self.mesh, ("batch", None))
            safe_mark_sharding(cache_position_dev, self.mesh, ("batch",))
            safe_mark_sharding(batch_idx_dev, self.mesh, ("batch",))
            safe_mark_sharding(fill_page_table_dev, self.mesh, ("batch", None))

        attn_metadata = self.model_state.prepare_attn(
            self.attention_layer_names,
            page_table_dev,
            cache_position_dev,
            fill_page_table=fill_page_table_dev,
            batch_idx=batch_idx_dev,
            num_users=target_num_reqs,
            dp_size=self.dp_size,
        )
        # from_v2_states only reads num_reqs + idx_mapping_np off the view.
        batch_view = SimpleNamespace(
            num_reqs=len(idx_mapping_np), idx_mapping_np=idx_mapping_np
        )
        sampling_metadata = XLASupportedSamplingMetadata.from_v2_states(
            self.req_states,
            self.sampling_states,
            batch_view,
            target_num_reqs,
            self.sampling_device,
            vocab_size=self.vocab_size,
        )

        if self.lora_config is not None:
            prompt_map, token_map, lora_reqs = self._make_lora_inputs(
                idx_mapping_np, num_scheduled_tokens
            )
            self._set_active_loras(prompt_map, token_map, lora_reqs)

        # Multimodal: merge gathered encoder embeds into the text embeds and feed
        # the model inputs_embeds (input_ids=None). Text-only requests on a mm
        # model still take this path (mm_embeds empty -> pure text embeddings).
        model_input_ids = input_ids_dev
        inputs_embeds_dev = None
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed, mm_indices = self._gather_mm_embeddings(
                idx_mapping_np, num_scheduled_tokens, padded_query_len
            )
            inputs_embeds_dev = self.model_state.get_mm_embeddings(
                input_ids_dev, mm_embeds, is_mm_embed, mm_indices
            )
            model_input_ids = None

        # Structured output: build the per-pass grammar tensors; apply_grammar is
        # a plain python bool so dynamo specializes a separate graph (no-grammar
        # path unaffected).
        apply_grammar = grammar_output is not None
        spec_active = spec_decode_metadata is not None
        if apply_grammar and spec_active:
            # Same limitation as v1: tenstorrent/tt-xla#5701.
            logger.warning_once(
                "Speculative decoding with grammar is not supported yet. "
                "Disabling grammar for this step."
            )
            apply_grammar = False
        require_struct = grammar_bitmask = bitmasks = None
        if apply_grammar:
            require_struct, grammar_bitmask, bitmasks = (
                self.prepare_structured_decoding_input(
                    grammar_output, idx_mapping_np, target_num_reqs
                )
            )

        result = self._forward_and_sample(
            model_input_ids,
            positions_dev,
            logits_indices_dev,
            attn_metadata,
            sampling_metadata,
            want_prompt_hs,
            inputs_embeds_dev,
            apply_grammar,
            require_struct,
            grammar_bitmask,
            bitmasks,
            logits_only=spec_active,
        )
        if want_prompt_hs:
            forward_out, hidden_states = result
        else:
            forward_out, hidden_states = result, None
        if spec_active:
            # Rejection sampling runs outside torch.compile: variable draft
            # lengths and early exit on rejection are request-wise dynamic
            # control flow dynamo cannot trace into a stable fullgraph.
            selected = self.rejection_sampler(
                spec_decode_metadata,
                None,
                forward_out,
                sampling_metadata=sampling_metadata,
            ).sampled_token_ids
            sampled, _ = RejectionSampler.parse_output(selected, self.vocab_size)
            sampled = sampled[: len(idx_mapping_np)]
            if want_prompt_hs:
                return sampled, hidden_states
            return sampled
        if self.cpu_sampling:
            # forward_out is logits [target_num_reqs, vocab]; mask (grammar) and
            # sample on host. Device sampling masks inside _sample_compiled.
            selected = self.sample_from_logits_cpu(
                forward_out,
                sampling_metadata,
                apply_grammar,
                require_struct,
                grammar_bitmask,
                bitmasks,
            )
        else:
            selected = forward_out
        # [target_num_reqs, 1] -> per active-req list; drop padding rows.
        # Transfer first, then slice on host: a device-side slice of the ROW_MAJOR
        # sampled-id tensor forces a to_layout typecast tt-mlir can't lower (v1-aligned).
        sampled = selected.cpu()[: len(idx_mapping_np)].tolist()
        if want_prompt_hs:
            return sampled, hidden_states
        return sampled

    def _forward_and_sample(
        self,
        input_ids,
        positions,
        logits_indices,
        attn_metadata,
        sampling_metadata,
        want_prompt_hs=False,
        inputs_embeds=None,
        apply_grammar=False,
        require_struct_decoding=None,
        grammar_bitmask=None,
        bitmasks=None,
        logits_only=False,
    ):
        """Run the forward->logits graph, then (device path) the sampling graph.

        The forward and the post-processing are compiled as two separate graphs so
        sampling variants don't re-specialize the expensive model forward:
        ``_forward_to_logits_compiled`` keys only on shape and ``want_prompt_hs``,
        while ``_sample_compiled`` keys on ``apply_grammar`` and the sampling mode.
        Under cpu_sampling the sampling graph is skipped -- the caller masks and
        samples on the host (see ``sample_from_logits_cpu``).
        """
        from vllm.forward_context import set_forward_context

        # Multimodal passes inputs_embeds with input_ids=None; both are 2D
        # [reqs, tokens(, H)] so the token count is the leading two dims.
        ref = input_ids if input_ids is not None else inputs_embeds
        num_tokens = ref.shape[0] * ref.shape[1]
        with set_forward_context(
            attn_metadata, self.vllm_config, num_tokens=num_tokens
        ):
            fwd = self._forward_to_logits_compiled(
                input_ids, positions, logits_indices, inputs_embeds, want_prompt_hs
            )
        logits, hidden_states = fwd if want_prompt_hs else (fwd, None)

        if self.cpu_sampling or logits_only:
            # Stop at logits; grammar + sampling run on the host (spec decode
            # rejection-samples them outside the compiled region).
            out = logits
        else:
            out = self._sample_compiled(
                logits,
                sampling_metadata,
                apply_grammar,
                require_struct_decoding,
                grammar_bitmask,
                bitmasks,
            )
        if want_prompt_hs:
            return out, hidden_states
        return out

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def _forward_to_logits_compiled(
        self,
        input_ids,
        positions,
        logits_indices,
        inputs_embeds=None,
        want_prompt_hs=False,
    ):
        """Compiled model forward -> last-token select -> logits.

        Keys only on shape and ``want_prompt_hs`` (which additionally returns the
        pre-selection hidden states for prompt logprobs); grammar and sampling
        live in ``_sample_compiled``, so the heavy forward is never re-specialized
        per post-processing combo. ``inputs_embeds`` is set (with ``input_ids=None``)
        on the multimodal path.
        """
        model_input_ids, model_positions, model_embeds, restore_shape = (
            self._prepare_model_call_tensors(input_ids, positions, inputs_embeds)
        )
        hidden_states = self.model(
            input_ids=model_input_ids,
            positions=model_positions,
            inputs_embeds=model_embeds,
        )
        hidden_states = self._restore_model_hidden_states(hidden_states, restore_shape)
        selected_states = self._select_hidden_states(hidden_states, logits_indices)
        logits = self.compute_logits(selected_states)
        if want_prompt_hs:
            return logits, hidden_states
        return logits

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def _sample_compiled(
        self,
        logits,
        sampling_metadata,
        apply_grammar=False,
        require_struct_decoding=None,
        grammar_bitmask=None,
        bitmasks=None,
    ):
        """Compiled grammar-mask + device sample over precomputed logits.

        ``apply_grammar`` is a plain python bool so dynamo specializes the
        grammar / no-grammar paths separately; only this small post-logits graph
        recompiles, not the forward.
        """
        if apply_grammar:
            logits = self.structured_decode(
                require_struct_decoding, grammar_bitmask, logits, bitmasks
            )
        return self._sample_from_logits(logits, sampling_metadata)

    def _pin_input_shardings(self, input_ids, positions, inputs_embeds) -> None:
        """Pin model inputs on the batch/mesh axis so warmup and inference trace
        the same graph. No-op off the SPMD path (ported from the v1 runner)."""
        if not self.enable_tensor_parallel and self.parallel_mode not in (
            ParallelismMode.DATA_PARALLEL_ONLY,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ):
            return
        # 2D mesh: batch dim -> "batch"; pure 1D-TP: "batch" is size 1, use "model".
        batch_axis = "batch" if self.use_2d_mesh else "model"
        if input_ids is not None:
            safe_mark_sharding(input_ids, self.mesh, (batch_axis, None))
        if inputs_embeds is not None:
            safe_mark_sharding(inputs_embeds, self.mesh, (batch_axis, None, None))
        # positions: pin only for DP modes; under pure-TP it drives GSPMD into a
        # batch-axis reduce_scatter that hits a tt-mlir to_layout bug.
        if self.parallel_mode in (
            ParallelismMode.DATA_PARALLEL_ONLY,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ):
            safe_mark_sharding(positions, self.mesh, (batch_axis, None))

    def _prepare_model_call_tensors(self, input_ids, positions, inputs_embeds):
        """Optionally flatten the 2D [reqs, tokens] tensors to 1D for flat_model_io."""
        if not self.use_flat_model_io:
            return input_ids, positions, inputs_embeds, None
        restore_shape = None
        if input_ids is not None and input_ids.ndim > 1:
            restore_shape = input_ids.shape
            input_ids = input_ids.reshape(-1)
        if inputs_embeds is not None and inputs_embeds.ndim > 2:
            if restore_shape is None:
                restore_shape = torch.Size(inputs_embeds.shape[:-1])
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        if positions.ndim > 1:
            if self.uses_mrope:
                assert positions.ndim == 3 and positions.shape[0] == 3
                positions = positions.reshape(3, -1)
            else:
                positions = positions.reshape(-1)

        assert (
            restore_shape is not None
        ), "restore_shape should be set if any input is flattened."
        return input_ids, positions, inputs_embeds, restore_shape

    def _restore_model_hidden_states(self, hidden_states, restore_shape):
        if restore_shape is None or hidden_states.ndim != 2:
            return hidden_states
        return hidden_states.reshape(*restore_shape, hidden_states.shape[-1])

    def _select_hidden_states(self, hidden_states, indices_do_sample):
        # Gather each request's last-token hidden state: hidden is [reqs, tokens, H].
        if indices_do_sample.ndim == 1:
            batch_indices = torch.arange(indices_do_sample.shape[0], dtype=torch.int32)
            result = hidden_states[batch_indices, indices_do_sample, :]
        else:
            # Spec decode: [reqs, spec_width] indices -> flat [reqs*spec_width, H]
            # so the rejection sampler's flat target/bonus indices line up.
            batch_indices = torch.arange(
                indices_do_sample.shape[0], dtype=torch.int32
            ).unsqueeze(1)
            batch_indices = batch_indices.expand_as(indices_do_sample)
            result = hidden_states[batch_indices, indices_do_sample, :]
            result = result.reshape(-1, result.shape[-1])
        if self.enable_tensor_parallel and self.use_2d_mesh:
            result = sharding_constraint_tensor(result, self.mesh, (None, None))
        return result

    def _sample_from_logits(self, logits, sampling_metadata):
        # Greedy fast-path (argmax) avoids the fused sampling kernel; the sampler
        # handles temperature/top-k/p/penalties/seeds otherwise.
        if (
            sampling_metadata.all_greedy
            and sampling_metadata.no_penalties
            and sampling_metadata.no_logit_bias
            and sampling_metadata.no_bad_words
            and sampling_metadata.no_allowed_token_ids
            and sampling_metadata.no_min_tokens
            and sampling_metadata.no_generators
        ):
            return torch.argmax(logits, dim=-1, keepdim=True)
        return self.sampler(logits, sampling_metadata).sampled_token_ids

    def prepare_structured_decoding_input(
        self, grammar_output, idx_mapping_np, num_reqs
    ):
        """Build the per-pass structured-decoding tensors (ported from v1).

        Keyed by batch position b (not the v1 persistent input_batch index): the
        request at slot ``idx_mapping_np[b]`` fills row b. ``grammar_bitmask`` has
        one row per structured request (scheduler order), so index it by the
        enumerate position so passes that hold only some structured requests stay
        aligned.
        """
        grammar_bitmask = grammar_output.grammar_bitmask
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        pos_by_req = {}
        for b in range(len(idx_mapping_np)):
            rid = self.req_states.index_to_req_id.get(int(idx_mapping_np[b]))
            if rid is not None:
                pos_by_req[rid] = b

        for mask_idx, req_id in enumerate(grammar_output.structured_output_request_ids):
            b = pos_by_req.get(req_id)
            if b is None:
                continue
            self.grammar_bitmask_cpu[b] = torch.from_numpy(grammar_bitmask[mask_idx])
            self.require_structured_out_cpu[b] = True

        return (
            self.require_structured_out_cpu[:num_reqs].to(self.device),
            self.grammar_bitmask_cpu[:num_reqs].to(self.device),
            self.structured_decode_bitmasks.to(self.device),
        )

    def structured_decode(
        self, require_struct_decoding, grammar_bitmask, logits, bitmasks
    ):
        """Mask logits to grammar-allowed tokens for rows requiring it (v1 port)."""
        return torch.where(
            require_struct_decoding,
            self._apply_grammar_bitmask(logits, grammar_bitmask, bitmasks),
            logits,
        )

    def _apply_grammar_bitmask(self, logits, grammar_bitmask, bitmasks):
        # Unpack the bitmask on-device with bitwise_and against power-of-2 masks
        # (bitwise_right_shift is unsupported on TT). grammar_bitmask:
        # [batch, ceil(vocab/32)] int32; bitmasks: [32] int32 = [1,2,...,2^31].
        bits = grammar_bitmask.unsqueeze(-1) & bitmasks
        allowed = (bits != 0).reshape(logits.shape[0], -1)[:, : self.vocab_size]
        return torch.where(allowed, logits, torch.full_like(logits, float("-inf")))

    def sample_from_logits_cpu(
        self,
        logits,
        sampling_metadata,
        apply_grammar=False,
        require_struct_decoding=None,
        grammar_bitmask=None,
        bitmasks=None,
    ):
        """Sample on CPU instead of compiling a device sampling graph.

        Ported from the v1 runner: supports greedy, temperature, top-k/top-p and
        penalty-based sampling. Uses Gumbel-max (argmax of logits + Gumbel noise)
        rather than torch.multinomial, which serializes over the batch.
        """
        logits = logits.cpu()

        # Grammar masking for the host path (the device path masks inside
        # _sample_compiled); move the grammar tensors to host to match logits.
        if apply_grammar:
            logits = self.structured_decode(
                require_struct_decoding.cpu(),
                grammar_bitmask.cpu(),
                logits,
                bitmasks.cpu(),
            )

        if not sampling_metadata.no_penalties:
            output_counts = sampling_metadata.output_token_counts
            occurred_output = output_counts > 0
            prompt_mask = sampling_metadata.prompt_token_mask
            rep_pen = sampling_metadata.repetition_penalties.unsqueeze(1)
            rep_mask = occurred_output | prompt_mask
            penalty_factor = torch.where(logits > 0, torch.reciprocal(rep_pen), rep_pen)
            logits = torch.where(rep_mask, logits * penalty_factor, logits)
            freq_pen = sampling_metadata.frequency_penalties.unsqueeze(1)
            logits -= freq_pen * output_counts.to(logits.dtype)
            pres_pen = sampling_metadata.presence_penalties.unsqueeze(1)
            logits -= pres_pen * occurred_output.to(logits.dtype)

        if sampling_metadata.all_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)

        temp = sampling_metadata.temperature
        temp = torch.where(temp < 1e-6, torch.ones_like(temp), temp)
        logits = logits / temp.unsqueeze(1)

        has_topk = sampling_metadata.top_k is not None
        has_topp = sampling_metadata.top_p is not None
        top_k = sampling_metadata.top_k if has_topk else None
        top_p = sampling_metadata.top_p if has_topp else None

        # Fast path: all requests share the same k > 0 -- single batched topk
        # reduces vocab from 128K to k before the top-p sort.
        uniform_k = 0
        if has_topk:
            k = int(top_k[0].item())
            if 0 < k < logits.size(1) and (top_k == k).all():
                uniform_k = k

        if uniform_k > 0:
            topk_vals, topk_idx = torch.topk(logits, uniform_k, dim=-1)

            if has_topp:
                for i in range(topk_vals.size(0)):
                    p = float(top_p[i].item())
                    if p < 1.0:
                        sorted_vals, sort_idx = torch.sort(
                            topk_vals[i], descending=True
                        )
                        probs = torch.softmax(sorted_vals, dim=-1)
                        mask = torch.cumsum(probs, dim=-1) - probs >= p
                        sorted_vals[mask] = float("-inf")
                        topk_vals[i].scatter_(0, sort_idx, sorted_vals)

            greedy = torch.argmax(topk_vals, dim=-1)
            gumbel = -torch.log(
                -torch.log(torch.rand_like(topk_vals.float()) + 1e-20) + 1e-20
            )
            random = torch.argmax(topk_vals + gumbel, dim=-1)
            sampled_local = torch.where(temp < 1e-6, greedy, random)
            return topk_idx.gather(-1, sampled_local.unsqueeze(-1))

        # Slow path: mixed k values or k=0 -- per-request filtering on full vocab.
        if has_topk:
            for i in range(logits.size(0)):
                k = int(top_k[i].item())
                if k > 0 and k < logits.size(1):
                    topk_vals, _ = torch.topk(logits[i], k)
                    logits[i][logits[i] < topk_vals[-1]] = float("-inf")

        if has_topp:
            for i in range(logits.size(0)):
                p = float(top_p[i].item())
                if p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits[i], descending=True
                    )
                    probs = torch.softmax(sorted_logits, dim=-1)
                    mask = torch.cumsum(probs, dim=-1) - probs >= p
                    sorted_logits[mask] = float("-inf")
                    logits[i].scatter_(0, sorted_indices, sorted_logits)

        greedy = torch.argmax(logits, dim=-1)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits.float()) + 1e-20) + 1e-20)
        random = torch.argmax(logits + gumbel, dim=-1)
        return torch.where(temp < 1e-6, greedy, random).unsqueeze(-1)

    def compute_logits(self, sample_hidden_states):
        """Vocab logits for the given hidden states (ported from the v1 runner)."""
        logits = self.model.compute_logits(sample_hidden_states)
        # Replicate sharded logits for SPMD: hooks can't reach ParallelLMHead
        # (quant_method.apply bypasses __call__) and all_gather is a no-op at
        # world_size 1, so the constraint must sit inside the compiled graph.
        if self.enable_tensor_parallel and self.is_sharded_compute_logits:
            logits = sharding_constraint_tensor(logits, self.mesh, (None, None))
        return logits

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def compute_logits_compiled(self, sample_hidden_states):
        """Compiled wrapper for compute_logits warmup / prompt-logprobs reuse."""
        return self.compute_logits(sample_hidden_states)

    def gather_logprobs(self, logits, token_ids):
        """Top-k logprobs + the given tokens' logprobs (ported from the v1 runner).

        Uses a fixed ``max_logprobs`` width so one compiled graph serves every
        request; callers trim to their per-request count on CPU.
        """
        logprobs = self.sampler.compute_logprobs(logits)
        return self.sampler.gather_logprobs(
            logprobs,
            self.model_config.max_logprobs,
            token_ids=token_ids.squeeze(-1),
        )

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def gather_logprobs_compiled(self, logits, token_ids):
        """Compiled wrapper for gather_logprobs warmup / prompt-logprobs reuse."""
        return self.gather_logprobs(logits, token_ids)

    def _get_prompt_logprobs_dict(self, prompt_lp_hs, scheduler_output):
        """Compute prompt logprobs for prefilling requests (ported from v1).

        For each request with prompt_logprobs enabled, processes this step's
        prompt positions in batches of max_num_reqs through compute_logits /
        gather_logprobs so the vocab-sized tensors stay on device; only the small
        gathered result moves to CPU. ``prompt_lp_hs`` holds per-slot hidden
        states [padded_query_len, H] captured during the step's forward passes.
        """
        if not self.num_prompt_logprobs:
            return {}

        from vllm.v1.outputs import LogprobsTensors

        in_progress = self.in_progress_prompt_logprobs
        result: dict[str, Optional[LogprobsTensors]] = {}
        completed: list[str] = []

        batch_hs_buf: Optional[torch.Tensor] = None
        batch_tgt_buf = torch.zeros(self.max_num_reqs, 1, dtype=torch.int64)

        for req_id, num_plp in self.num_prompt_logprobs.items():
            # gather_logprobs is compiled for max_logprobs; clamp so the trim
            # slice below never reads past the gathered columns.
            num_plp = min(num_plp, self.model_config.max_logprobs)
            slot = self.req_states.req_id_to_index.get(req_id)
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id)
            if slot is None or num_tokens is None:
                continue

            num_prompt_tokens = int(self.req_states.prompt_len[slot])
            prompt_token_ids = self.req_states.all_token_ids[slot, :num_prompt_tokens]

            logprobs_tensors = in_progress.get(req_id)
            if logprobs_tensors is None:
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_plp + 1
                )
                in_progress[req_id] = logprobs_tensors

            start_idx = int(self.req_states.num_computed_tokens[slot])
            start_tok = start_idx + 1
            num_remaining = num_prompt_tokens - start_tok

            if num_tokens <= num_remaining:
                num_logits = num_tokens
            else:
                num_logits = num_remaining
                completed.append(req_id)
                result[req_id] = logprobs_tensors

            if num_logits <= 0:
                continue

            hs_cpu = prompt_lp_hs.get(slot)
            assert hs_cpu is not None, (
                f"req {req_id} (slot {slot}) not found in prompt_lp_hs — "
                f"state inconsistency"
            )
            hs_cpu = hs_cpu[:num_logits, :]

            if batch_hs_buf is None or batch_hs_buf.shape[-1] != hs_cpu.shape[-1]:
                batch_hs_buf = torch.zeros(
                    self.max_num_reqs, hs_cpu.shape[-1], dtype=hs_cpu.dtype
                )

            all_tgt_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            for batch_start in range(0, num_logits, self.max_num_reqs):
                batch_end = min(batch_start + self.max_num_reqs, num_logits)
                batch_size = batch_end - batch_start

                batch_hs_buf.zero_()
                batch_hs_buf[:batch_size] = hs_cpu[batch_start:batch_end]
                batch_hs_dev = batch_hs_buf.to(self.device)

                logits = self.compute_logits(batch_hs_dev)

                batch_tgt_buf.zero_()
                batch_tgt_buf[:batch_size, 0] = torch.tensor(
                    all_tgt_ids[batch_start:batch_end], dtype=torch.int64
                )
                batch_tgt_dev = batch_tgt_buf.to(self.device)

                lp_tensors = self.gather_logprobs(logits, batch_tgt_dev)

                ids_cpu = lp_tensors.logprob_token_ids.cpu()
                lps_cpu = lp_tensors.logprobs.cpu()
                ranks_cpu = lp_tensors.selected_token_ranks.cpu()

                dest = slice(start_idx + batch_start, start_idx + batch_end)
                logprobs_tensors.logprob_token_ids[dest] = ids_cpu[
                    :batch_size, : num_plp + 1
                ]
                logprobs_tensors.logprobs[dest] = lps_cpu[:batch_size, : num_plp + 1]
                logprobs_tensors.selected_token_ranks[dest] = ranks_cpu[:batch_size]

        for req_id in completed:
            del self.num_prompt_logprobs[req_id]
            del in_progress[req_id]

        return result

    def _allocate_kv_caches(self, kv_cache_config: "KVCacheConfig") -> dict:
        """Allocate the per-layer KV cache tensors on the TT device.

        The device-coupled core of ``initialize_kv_cache`` (salvaged from the v1
        fork), split out so it can be exercised on-device without the engine
        wrappers. Standard attention gets separate ``[k_cache, v_cache]`` tensors
        (avoids slice/concat in the compiled graph); MLA gets a single latent
        tensor. Only one KV-cache group (no hybrid) and one owner per tensor are
        supported. Returns ``layer_name -> tensor | [k, v]``.
        """
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            MLAAttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        def _layer_spec(group_spec, name):
            # Heterogeneous same-type layers (e.g. Gemma-4's mixed head dims) are
            # merged into one group whose spec is a UniformTypeKVCacheSpecs; the
            # real per-layer spec (with its own page_size/head_size) is inside.
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                return group_spec.kv_cache_specs[name]
            return group_spec

        from .attention_impls.attention import TTAttentionBackend
        from .attention_impls.attention_mla import TTMLAAttentionBackend

        groups = kv_cache_config.kv_cache_groups
        if len(groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )
        if len(groups) == 0:
            # Valid when only a subset of layers is compiled (layer override).
            return {}

        assert groups[0].kv_cache_spec.block_size == self.block_size, (
            f"KV cache block_size {groups[0].kv_cache_spec.block_size} must match "
            f"the runner block_size {self.block_size}"
        )

        kv_cache_sizes: dict[str, int] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            assert (
                len(tensor.shared_by) == 1
            ), "A KV cache tensor shared by multiple layers is not supported on TT."
            kv_cache_sizes[tensor.shared_by[0]] = tensor.size

        kv_caches: dict = {}
        for group in groups:
            for layer_name in group.layer_names:
                spec = _layer_spec(group.kv_cache_spec, layer_name)
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % spec.page_size_bytes == 0
                num_blocks = tensor_size // spec.page_size_bytes
                if isinstance(spec, MLAAttentionSpec):
                    # Single concatenated latent KV tensor (num_kv_heads == 1).
                    shape = TTMLAAttentionBackend.get_kv_cache_shape(
                        num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size
                    )
                    kv_caches[layer_name] = torch.zeros(shape, dtype=spec.dtype).to(
                        self.device
                    )
                elif isinstance(spec, AttentionSpec):
                    if self.enable_tensor_parallel:
                        tp_size = self.original_parallel_config.tensor_parallel_size
                        assert spec.num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {spec.num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD"
                        )
                    shape = TTAttentionBackend.get_kv_cache_shape(
                        num_blocks, spec.block_size, spec.num_kv_heads, spec.head_size
                    )
                    # spec.dtype may be a 1-byte accounting dtype; the device
                    # buffers use the real transfer dtype.
                    k = torch.zeros(shape, dtype=self.kv_cache_dtype).to(self.device)
                    v = torch.zeros(shape, dtype=self.kv_cache_dtype).to(self.device)
                    kv_caches[layer_name] = [k, v]
                else:
                    raise NotImplementedError(
                        f"Unsupported KV cache spec: {type(spec)}"
                    )
        return kv_caches

    def _maybe_setup_cross_layer_kv_sharing(
        self, kv_caches: dict, kv_cache_config: "KVCacheConfig"
    ) -> None:
        """Point cross-layer KV-sharing (child) layers at the target's cache.

        Gemma-4's last layers allocate no cache of their own and reuse an earlier
        layer's; without this they keep an empty placeholder and decode indexes
        it (kv_cache[0]) on an empty tensor. Mirrors the v1 runner.
        """
        if not self.shared_kv_cache_layers:
            return
        from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers, kv_cache_config.kv_cache_groups
        )
        for child, target in self.shared_kv_cache_layers.items():
            kv_caches[child] = kv_caches[target]

    def get_kv_cache_spec(self) -> dict:
        """Build the per-layer KVCacheSpec from the model's attention modules.

        Emits a Full / SlidingWindow / MLA spec per layer, skipping
        cross-layer-shared (recorded in shared_kv_cache_layers) and encoder-only
        layers. The engine uses it to budget KV blocks and build the KVCacheConfig.
        """
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
            SlidingWindowSpec,
        )

        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        block_size = self.vllm_config.cache_config.block_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        kv_cache_spec: dict = {}
        for layer_name, attn_module in layers.items():
            if isinstance(attn_module, Attention):
                if attn_module.kv_sharing_target_layer_name is not None:
                    # Reuses another layer's KV cache; allocate nothing here.
                    self.shared_kv_cache_layers[layer_name] = (
                        attn_module.kv_sharing_target_layer_name
                    )
                    continue
                if attn_module.attn_type == AttentionType.DECODER:
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_spec_dtype,
                            sliding_window=attn_module.sliding_window,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_spec_dtype,
                        )
                elif attn_module.attn_type in (
                    AttentionType.ENCODER,
                    AttentionType.ENCODER_ONLY,
                ):
                    continue  # encoder-only attention needs no KV cache
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError(
                        "Encoder-decoder attention is not supported yet."
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_module.attn_type}")
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_spec_dtype,
                    cache_dtype_str=cache_dtype_str,
                )
        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: "KVCacheConfig") -> None:
        """Allocate KV caches, bind them into the forward context, register them
        with the KV-transfer group, and (under TP) shard them across the mesh.
        """
        from vllm.v1.worker.utils import bind_kv_cache

        kv_caches = self._allocate_kv_caches(kv_cache_config)
        if not kv_caches:
            return
        self._maybe_setup_cross_layer_kv_sharing(kv_caches, kv_cache_config)
        logger.info("Allocated KV cache for %d attention layer(s).", len(kv_caches))
        self.kv_cache_config = kv_cache_config
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )
        self._register_kv_caches_for_transfer(kv_caches)

        if self.parallel_mode == ParallelismMode.DATA_TENSOR_PARALLEL:
            # DP+TP: leave replicated; each device writes its own slice via
            # paged_update_cache. A TP-only spec puts block_size on the DP axis
            # and fails ttir.paged_update_cache (follow-up).
            return
        if self.enable_tensor_parallel:
            import torch_xla.distributed.spmd as xs

            for entry in self.kv_caches:
                is_pair = isinstance(entry, (list, tuple))
                for cache in entry if is_pair else [entry]:
                    assert cache.ndim == 4, "KV cache tensor must be 4D."
                    if is_pair:
                        # Shard standard K/V on num_kv_heads over the "model" axis.
                        safe_mark_sharding(
                            cache, self.mesh, (None, "model", None, None)
                        )
                    else:
                        # Replicate the MLA latent KV cache.
                        xs.mark_sharding(cache, self.mesh, (None, None, None, None))

    def _register_kv_caches_for_transfer(self, kv_caches: dict) -> None:
        """Hand the allocated caches to the KV-connector, if one is configured.

        Without this a disaggregated-serving setup silently transfers nothing.
        """
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )
        from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks

        if not has_kv_transfer_group():
            return
        get_kv_transfer_group().register_kv_caches(kv_caches)
        get_kv_transfer_group().set_host_xfer_buffer_ops(copy_kv_blocks)
        logger.info(
            "Registered %d KV cache layer(s) with the KV connector.", len(kv_caches)
        )

    def get_model(self):
        return self.model

    def reset_mm_cache(self) -> None:
        if self.mm_budget is not None:
            self.mm_budget.reset_cache()

    # add_lora / maybe_setup_dummy_loras / maybe_select_dummy_loras /
    # list_loras / remove_lora / pin_lora come from LoRAModelRunnerMixin.

    def _set_active_loras(
        self, prompt_lora_mapping, token_lora_mapping, lora_requests, *args, **kwargs
    ) -> None:
        """Activate LoRAs, bracketing with torch_xla.sync (ported from v1).

        The syncs capture the input/metadata updates around the integer LoRA
        index, which would otherwise force a recompile.
        """
        import torch_xla

        torch_xla.sync(wait=False)
        super()._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests, *args, **kwargs
        )
        torch_xla.sync(wait=False)

    def _make_lora_inputs(self, idx_mapping_np, num_scheduled_tokens):
        """Build (prompt, token) LoRA id mappings + active requests for a pass.

        Host substitute for InputBatch.make_lora_inputs: repeats each request's
        LoRA id (0 = base model) by its sampled-token count (1) and scheduled-
        token count, gathered in batch order via idx_mapping.
        """
        prompt_lora_mapping: list[int] = []
        token_lora_mapping: list[int] = []
        lora_requests = set()
        for b in range(len(idx_mapping_np)):
            slot = int(idx_mapping_np[b])
            lora_req = self.lora_requests_by_slot.get(slot)
            lora_id = lora_req.lora_int_id if lora_req is not None else 0
            prompt_lora_mapping.append(lora_id)
            token_lora_mapping.extend([lora_id] * int(num_scheduled_tokens[b]))
            if lora_req is not None:
                lora_requests.add(lora_req)
        return tuple(prompt_lora_mapping), tuple(token_lora_mapping), lora_requests

    def get_supported_tasks(self) -> tuple:
        """Report the generation tasks this model supports (via TTModelState)."""
        if self.model_config.runner_type != "generate":
            raise NotImplementedError(
                "Only the generate runner type is supported in the v2 runner."
            )
        assert (
            self.model_state is not None
        ), "load_model must run before get_supported_tasks"
        return tuple(self.model_state.get_supported_generation_tasks())

    def reset_dynamo_cache(self) -> None:
        """Clear the compiled-model dynamo cache so it re-traces cleanly."""
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        if self.model_config.is_multimodal_model:
            compiled_model = self.model.get_language_model().model
        else:
            compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWithNoGuardsWrapper):
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object()
            )
            compiled_model.compiled = False
            TorchCompileWithNoGuardsWrapper.__init__(compiled_model)

    def update_config(self, overrides: dict) -> None:
        from vllm.config import update_config as _update_config

        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            setattr(
                self,
                config_name,
                _update_config(getattr(self, config_name), config_overrides),
            )

    def reload_weights(self) -> None:
        from vllm.model_executor.model_loader import get_model_loader

        assert (
            getattr(self, "model", None) is not None
        ), "Cannot reload weights before model is loaded."
        logger.info("Reloading weights inplace...")
        get_model_loader(self.load_config).load_weights(
            self.model, model_config=self.model_config
        )

    def ensure_kv_transfer_shutdown(self) -> None:
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        from vllm.distributed.kv_transfer.kv_transfer_state import (
            ensure_kv_transfer_shutdown,
        )

        if has_kv_transfer_group():
            ensure_kv_transfer_shutdown()

    def _warmup_buckets(self) -> list[tuple[int, int]]:
        """(target_num_reqs, padded_query_len) pairs to precompile.

        Decode always runs at max_num_reqs; prefill runs at the min / max prefill
        buckets. Each request-count bucket is compiled at every token-length
        padding, matching the shapes _select_batch can produce at runtime.
        """
        targets = sorted(
            {self.max_num_reqs, self.min_num_reqs, self.max_prefill_num_reqs}
        )
        return [(t, q) for t in targets for q in self.num_tokens_paddings]

    def capture_model(self) -> None:
        """Precompile the forward/sample graph at every runtime bucket shape.

        Warms the graph once with the DP input/KV shardings pinned so the first
        real request pays no compile latency and traces the same graph warmup did.
        Uses valid dummy attention metadata (non-zero cache_position, real
        batch_idx) — all-zeros dummies fail to compile the paged-attention op.
        Greedy (argmax) path only; multimodal warmup is deferred. Under
        cpu_sampling the graph returns logits (host sampling is eager), so the
        same precompile warms the correct graph.
        """
        import time

        import torch_xla

        torch._dynamo.config.dynamic_shapes = False
        start = time.perf_counter()
        buckets = self._warmup_buckets()
        logger.info("MRv2 warmup: precompiling %d bucket(s).", len(buckets))
        # Activate dummy LoRAs for the warmup so the compiled graphs match the
        # LoRA runtime shapes (no-op context when lora_config is None).
        with self.maybe_setup_dummy_loras(self.lora_config):
            for target_num_reqs, padded_query_len in buckets:
                logger.info(
                    "MRv2 warmup: num_reqs=%d, query_len=%d",
                    target_num_reqs,
                    padded_query_len,
                )
                if self.lora_config is not None:
                    self.maybe_select_dummy_loras(
                        self.lora_config,
                        np.array([target_num_reqs], dtype=np.int32),
                    )
                self._precompile_bucket(target_num_reqs, padded_query_len)
                # Sync per bucket so prefill and decode graphs stay separate.
                torch_xla.sync()
        torch_xla.sync()
        logger.info("MRv2 warmup finished in %.2f [secs].", time.perf_counter() - start)

    def _precompile_bucket(self, target_num_reqs: int, padded_query_len: int) -> None:
        """Trace the fused forward+sample graph for one bucket with valid dummies.

        Mirrors _run_model_pass's device-tensor + sharding sequence so the warmed
        graph matches inference exactly (input shardings pinned eagerly; page /
        cache / batch_idx sharded on the DP batch axis).
        """
        from .metadata import XLASupportedSamplingMetadata

        dev = self.device
        input_ids = torch.zeros(
            (target_num_reqs, padded_query_len), dtype=torch.int32
        ).to(dev)
        positions_shape = (
            (3, target_num_reqs, padded_query_len)
            if self.uses_mrope
            else (target_num_reqs, padded_query_len)
        )
        positions = torch.zeros(positions_shape, dtype=torch.int32).to(dev)
        logits_indices = torch.zeros(target_num_reqs, dtype=torch.int32).to(dev)
        page_table = torch.zeros(
            (target_num_reqs, self.max_num_blocks_per_req), dtype=torch.int32
        ).to(dev)
        fill_page_table = torch.zeros(
            (target_num_reqs, self.max_num_blocks_per_req), dtype=torch.int32
        ).to(dev)
        # Non-zero write position: all-zeros makes the paged cache op fail to
        # compile (matches v1's _dummy_run, which uses ones).
        cache_position = torch.ones(target_num_reqs, dtype=torch.int32).to(dev)
        # from_numpy (not on-device arange) so the DP "batch" sharding sticks.
        batch_idx = torch.from_numpy(np.arange(target_num_reqs, dtype=np.int32)).to(dev)

        self._pin_input_shardings(input_ids, positions, None)
        if self.parallel_mode in (
            ParallelismMode.DATA_PARALLEL_ONLY,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ):
            safe_mark_sharding(page_table, self.mesh, ("batch", None))
            safe_mark_sharding(cache_position, self.mesh, ("batch",))
            safe_mark_sharding(batch_idx, self.mesh, ("batch",))
            safe_mark_sharding(fill_page_table, self.mesh, ("batch", None))

        attn_metadata = self.model_state.prepare_attn(
            self.attention_layer_names,
            page_table,
            cache_position,
            fill_page_table=fill_page_table,
            batch_idx=batch_idx,
            num_users=target_num_reqs,
            dp_size=self.dp_size,
        )
        # Defaults are all-greedy -> warms the argmax fast-path.
        sampling_metadata = XLASupportedSamplingMetadata(all_greedy=True)
        self._forward_and_sample(
            input_ids, positions, logits_indices, attn_metadata, sampling_metadata
        )
