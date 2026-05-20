# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ModelAdapter protocol.

Each (model, variant) pair (e.g. DeepSeek V4 Flash, Llama-3-8B, Qwen3-MoE)
provides one implementation. `streaming.core.run_streaming` takes an
instance and orchestrates streaming load + compile + inference.

Notes on `forward_*` methods:

  * `call_model` is used in `whole_graph` mode (one `torch.compile(model)`
    call wraps the full forward).
  * `forward_pre_layers / forward_layer / forward_post_layers` split the
    forward into three stages that the `layer_eager` mode compiles + executes
    independently. The default `call_model` chains them so adapters that
    only implement the three stages still work in both modes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import torch
import torch.nn as nn


class ModelAdapter(Protocol):
    """Each model variant implements this."""

    # ---- identification ----
    name: str  # e.g. "deepseek-v4-flash"
    tokenizer_repo_id: str  # HF repo for tokenizer

    # ---- model construction ----
    def build_skeleton(
        self,
        num_layers: Optional[int],
        bsz: int,
        prompt_len: int,
        max_new_tokens: int,
    ) -> nn.Module: ...

    def get_layers(self, model: nn.Module) -> List[nn.Module]: ...

    # ---- per-layer load ----
    def load_block_state_dict(self, layer_id: int) -> Dict[str, torch.Tensor]: ...

    def post_load_block(
        self,
        block: nn.Module,
        layer_id: int,
        mesh_shape: Tuple[int, ...],
    ) -> None:
        """Optional hook AFTER state_dict load. e.g. MoE swap. Default: no-op."""
        return

    # ---- sharding ----
    def block_shard_spec(
        self,
        block: nn.Module,
        mesh,
    ) -> Dict[torch.Tensor, Tuple]: ...

    def top_level_shard_spec(
        self,
        model: nn.Module,
    ) -> Dict[torch.Tensor, Tuple]: ...

    # ---- dummy flush ----
    def dummy_block_inputs(
        self,
        model: nn.Module,
        bsz: int,
        prompt_len: int,
        device,
        mesh,
    ) -> Tuple[torch.Tensor, ...]:
        """Args fed to `block.forward(...)` during the per-layer dummy flush.
        Shape must match real prefill so the HLO graph matches."""
        ...

    def mutable_buffer_names(self) -> Set[str]:
        """Buffer names that the per-block forward MUTATES (need re-zero
        before real prefill). Default: just `kv_cache`. Override for window
        attention, compressor, etc."""
        return {"kv_cache"}

    def mutable_buffer_init_value(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the fresh CPU tensor used to re-initialize a mutable buffer
        between dummy flush and real prefill. Default: zeros. Override for
        buffers whose semantics require a non-zero start (e.g. softmax-masked
        score states init'd to -inf)."""
        return torch.zeros(shape, dtype=dtype)

    def block_output_spec(self, block: nn.Module) -> Tuple:
        """Partition spec applied to block forward output during the dummy
        flush. Default: fully replicated. Override to match the activation
        flow when the block produces a sharded output (e.g. hidden-shard)."""
        return (None, None, None, None)

    # ---- post-skeleton optimization ----
    def dedupe_per_block_tensors(self, layers: List[nn.Module]) -> int:
        """Optional hook AFTER skeleton + before top-level ship. Lets the
        adapter alias tensors that are logically identical across layers
        (e.g. positional embeddings keyed by an attribute on each block)
        so the per-layer dummy flush hits the compile cache instead of
        retracing every layer. Returns the number of aliased layers for
        logging. Default: no-op."""
        return 0

    # ---- tokenization + inference ----
    def tokenize_prompts(
        self,
        prompts: List[str],
        bsz: int,
        prompt_len: int,
    ) -> torch.Tensor:
        """Returns (bsz, prompt_len) input_ids."""
        ...

    # ---- whole_graph mode ----
    def call_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Whole-model forward returning logits. Default: chains pre→layers
        →post so adapters only need to implement the per-stage methods."""
        h = self.forward_pre_layers(model, input_ids, start_pos)
        for layer in self.get_layers(model):
            h = self.forward_layer(layer, h, start_pos, input_ids)
        return self.forward_post_layers(model, h, input_ids)

    # ---- layer_eager mode (per-stage forward) ----
    def forward_pre_layers(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> Any:
        """Embed + any pre-block reshape. Returns the initial `h` that the
        first layer consumes."""
        ...

    def forward_layer(
        self,
        layer: nn.Module,
        h: Any,
        start_pos: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Any:
        """Single block forward."""
        ...

    def forward_post_layers(
        self,
        model: nn.Module,
        h: Any,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """norm + head + any final reshape. Returns logits (bsz, vocab)."""
        ...

    # ---- weight dtype overrides (model-specific paths) ----
    def weight_dtype_overrides(
        self,
        expert_dtype: str,
        attn_dtype: str,
    ) -> Dict[str, str]:
        """Map config dtype values (e.g. "bfp_bf8", "bfp_bf4") to
        `{module_glob: dtype_str}` for `apply_weight_dtype_overrides`. Each
        weight class (expert / attention / ...) is independently controllable;
        "bf16" / "" / "none" for that class disables overrides on those paths.
        Adapters may extend with more weight classes by adding kwargs."""
        return {}
