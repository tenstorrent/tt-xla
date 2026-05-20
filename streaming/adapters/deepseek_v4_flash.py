# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash adapter for `streaming.core.run_streaming`."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from tt_torch.sparse_mlp import enable_sparse_mlp

from streaming._helpers import _build_skeleton
from streaming.streaming_loader import _strip_cpu_golden_refs, _upload_with_sharding
from streaming.weight_loaders import deepseek_v4_flash as wl

# 32 canned prompts shared with test_deepseek_v4_full_e2e.py. The runner
# (or the adapter's `tokenize_prompts`) picks the first BATCH_SIZE entries.
PROMPTS = [
    "How are you today?",
    "What is the capital of France?",
    "Explain machine learning briefly.",
    "Who painted the Mona Lisa?",
    "What is two plus two?",
    "Tell me a fun fact about space.",
    "What is photosynthesis?",
    "How does a transformer model work?",
    "What is the speed of light?",
    "Name three programming languages.",
    "What causes earthquakes?",
    "How do you make pasta from scratch?",
    "What is the largest planet in our solar system?",
    "Who wrote the play Hamlet?",
    "What is gravity?",
    "How does the internet work?",
    "What is the human brain made of?",
    "Tell me about black holes.",
    "What is DNA?",
    "How does a car engine work?",
    "What is the meaning of recursion?",
    "Who was Albert Einstein?",
    "What is climate change?",
    "How do plants grow?",
    "What is quantum entanglement?",
    "Tell me a short story about a robot.",
    "What is a relational database?",
    "How does Wi-Fi work?",
    "What is the Pythagorean theorem?",
    "Name three renewable energy sources.",
    "What was the French Revolution about?",
    "How do volcanoes form?",
]


class DeepSeekV4FlashAdapter:
    """`streaming.adapters.base.ModelAdapter` implementation for the
    DeepSeek-V4-Flash HF release."""

    name = "deepseek-v4-flash"
    tokenizer_repo_id = wl.REPO_ID

    def __init__(self) -> None:
        # Cached after first build_skeleton call; needed by post_load_block
        # (enable_sparse_mlp wants the same args), dummy_block_inputs (uses
        # args.dim), and the few other adapter methods that reach into config.
        self._args = None
        # Cached after first build_skeleton; used by tokenize_prompts.
        self._tokenizer = None

    # ---- model construction ----
    def build_skeleton(
        self,
        num_layers: Optional[int],
        bsz: int,
        prompt_len: int,
        max_new_tokens: int,
    ) -> nn.Module:
        args = wl.load_config_args()
        args.n_mtp_layers = 0
        args.max_batch_size = bsz
        if num_layers is not None and num_layers < args.n_layers:
            args.n_layers = num_layers
            args.compress_ratios = args.compress_ratios[:num_layers]

        # max_seq_len rounding: compressor uses fixed-size kv_cache sized
        # at max_seq_len // ratio, so we need a multiple of every compress
        # ratio that's >= prompt + decode steps.
        max_cr = max(args.compress_ratios) if args.compress_ratios else 0
        needed = prompt_len + max_new_tokens
        if max_cr > 0:
            rounded = ((needed + max_cr - 1) // max_cr) * max_cr
            args.max_seq_len = max(rounded, 2 * max_cr)
        else:
            args.max_seq_len = ((needed + 31) // 32) * 32

        self._args = args
        return _build_skeleton(args)

    def get_layers(self, model: nn.Module) -> List[nn.Module]:
        return list(model.layers)

    # ---- per-layer load ----
    def load_block_state_dict(self, layer_id: int) -> Dict[str, torch.Tensor]:
        block_sd = wl.load_block_state_dict(layer_id)
        # Strip `layers.{N}.` prefix so the dict applies to a single Block.
        prefix = f"layers.{layer_id}."
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in block_sd.items()
        }

    def post_load_block(
        self,
        block: nn.Module,
        layer_id: int,
        mesh_shape: Tuple[int, ...],
    ) -> None:
        """MoE swap + strip CPU expert copies that sparse_mlp leaves behind."""
        enable_sparse_mlp(
            block,
            mesh=mesh_shape,
            cluster_axis=0,
            config=self._args,
            verbose=False,
        )
        _strip_cpu_golden_refs(block)

    # ---- sharding ----
    def block_shard_spec(
        self,
        block: nn.Module,
        mesh,
    ) -> Dict[torch.Tensor, Tuple]:
        compound = ("_axis_0", "_axis_1")
        specs: Dict[torch.Tensor, Tuple] = {}

        attn = block.attn
        specs[attn.wq_b.weight] = ("_axis_1", None)
        specs[attn.wo_a.weight] = ("_axis_1", None)
        specs[attn.wo_b.weight] = (None, "_axis_1")
        specs[attn.kv_cache] = ("_axis_0", None, None)
        if attn.compress_ratio:
            specs[attn.compressor.kv_cache] = ("_axis_0", None, None)
            specs[attn.compressor.kv_state] = ("_axis_0", None, None)
            specs[attn.compressor.score_state] = ("_axis_0", None, None)
            if getattr(attn, "indexer", None) is not None:
                specs[attn.indexer.wq_b.weight] = ("_axis_1", None)
                specs[attn.indexer.weights_proj.weight] = ("_axis_1", None)
                specs[attn.indexer.compressor.kv_cache] = ("_axis_0", None, None)
                specs[attn.indexer.compressor.kv_state] = ("_axis_0", None, None)
                specs[attn.indexer.compressor.score_state] = ("_axis_0", None, None)

        # MoE (post-swap A2aSparseMLPWithSharedExperts).
        a2a_with_shared = block.ffn
        mlp = a2a_with_shared.mlp
        specs[mlp.router.gate.weight] = (None, None)
        specs[mlp.experts.gate_proj] = (compound, None, None)
        specs[mlp.experts.up_proj] = (compound, None, None)
        specs[mlp.experts.down_proj] = (compound, None, None)

        shared = getattr(a2a_with_shared, "shared_experts", None)
        if shared is not None:
            specs[shared.w1.weight] = (None, None)
            specs[shared.w2.weight] = (None, None)
            specs[shared.w3.weight] = (None, None)
        return specs

    def top_level_shard_spec(
        self,
        model: nn.Module,
    ) -> Dict[torch.Tensor, Tuple]:
        return {
            model.embed.weight: (None, None),
            model.norm.weight: (None,),
            model.head.weight: (None, None),
            model.hc_head_fn: (None, None),
            model.hc_head_base: (None,),
            model.hc_head_scale: (None,),
        }

    # ---- dummy flush ----
    def dummy_block_inputs(
        self,
        model: nn.Module,
        bsz: int,
        prompt_len: int,
        device,
        mesh,
    ) -> Tuple[torch.Tensor, ...]:
        """DS V4 Block forward signature: `block(h, sp, ids)` where h is the
        hc-expanded hidden state (bsz, seqlen, hc_mult, dim)."""
        h_cpu = torch.zeros(
            bsz,
            prompt_len,
            model.hc_mult,
            self._args.dim,
            dtype=torch.bfloat16,
        )
        h = _upload_with_sharding(
            h_cpu,
            mesh,
            ("_axis_0", None, None, None),
            device,
        )
        sp = torch.tensor(0, dtype=torch.long).to(device)
        ids_cpu = torch.zeros(bsz, prompt_len, dtype=torch.long)
        ids = _upload_with_sharding(
            ids_cpu,
            mesh,
            ("_axis_0", None),
            device,
        )
        del h_cpu, ids_cpu
        return (h, sp, ids)

    def mutable_buffer_names(self) -> Set[str]:
        # DS V4: window attn `kv_cache`, plus compressor's `kv_cache` /
        # `kv_state` / `score_state` (and indexer.compressor.* with same
        # names). All three are read+written by the dummy forward.
        return {"kv_cache", "kv_state", "score_state"}

    def mutable_buffer_init_value(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # `score_state` is consumed via softmax(dim=1) inside the compressor;
        # the original module inits it to -inf so unwritten slots receive
        # zero probability mass after softmax. Re-zeroing would make them
        # equiprobable masked-fill candidates and silently corrupt routing.
        if name == "score_state":
            return torch.full(shape, float("-inf"), dtype=dtype)
        return torch.zeros(shape, dtype=dtype)

    def block_output_spec(self, block: nn.Module) -> Tuple:
        return ("_axis_0", None, None, None)

    # ---- post-skeleton optimization ----
    def dedupe_per_block_tensors(self, layers: List[nn.Module]) -> int:
        """Alias `freqs_cis` across layers that share the same compress
        ratio so the per-layer dummy flush compile-cache hits instead of
        retracing every layer. Also aliases the compressor's `freqs_cis`
        when present. Returns the number of aliased layers."""
        freqs_by_cr: Dict[int, torch.Tensor] = {}
        aliased = 0
        for blk in layers:
            if not hasattr(blk, "attn") or not hasattr(blk.attn, "freqs_cis"):
                break
            cr = getattr(blk.attn, "compress_ratio", 0)
            if cr in freqs_by_cr:
                blk.attn.freqs_cis = freqs_by_cr[cr]
                comp = getattr(blk.attn, "compressor", None)
                if comp is not None:
                    comp.freqs_cis = freqs_by_cr[cr]
                aliased += 1
            else:
                freqs_by_cr[cr] = blk.attn.freqs_cis
        return aliased

    # ---- tokenization ----
    def tokenize_prompts(
        self,
        prompts: List[str],
        bsz: int,
        prompt_len: int,
    ) -> torch.Tensor:
        """Manually wrap each prompt with <BOS><User>{prompt}<Assistant> so
        the model enters conversational-answer mode. V4-Flash's tokenizer
        has no built-in chat_template but the special tokens exist in vocab.
        """
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_repo_id)
        tok = self._tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        bos_id = tok.bos_token_id
        user_id = tok.convert_tokens_to_ids("<｜User｜>")
        asst_id = tok.convert_tokens_to_ids("<｜Assistant｜>")

        rows = []
        for i in range(bsz):
            prompt = prompts[i % len(prompts)]
            body = tok(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            wrap = torch.tensor(
                [bos_id, user_id] + body.tolist() + [asst_id],
                dtype=torch.long,
            )
            if wrap.shape[0] >= prompt_len:
                wrap = wrap[-prompt_len:]
            else:
                pad = torch.full(
                    (prompt_len - wrap.shape[0],),
                    pad_id,
                    dtype=torch.long,
                )
                wrap = torch.cat([pad, wrap], dim=0)
            rows.append(wrap)
        return torch.stack(rows, dim=0).contiguous()

    # ---- whole_graph forward ----
    def call_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> torch.Tensor:
        return model(input_ids, start_pos)

    # ---- layer_eager forward (per-stage) ----
    def forward_pre_layers(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Embed → unsqueeze + repeat for hc_mult streams."""
        h = model.embed(input_ids)
        return h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)

    def forward_layer(
        self,
        layer: nn.Module,
        h: torch.Tensor,
        start_pos: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return layer(h, start_pos, input_ids)

    def forward_post_layers(
        self,
        model: nn.Module,
        h: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return model.head(
            h,
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.norm,
        )

    # ---- weight dtype overrides ----
    def weight_dtype_overrides(
        self,
        expert_dtype: str,
        attn_dtype: str,
    ) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        if expert_dtype and expert_dtype.lower() not in ("bf16", "none", ""):
            # Routed MoE experts (stacked across n_routed_experts) + shared
            # experts (FFN-style Linear in every block).
            overrides.update(
                {
                    "layers.*.ffn.mlp.experts.gate_proj": expert_dtype,
                    "layers.*.ffn.mlp.experts.up_proj": expert_dtype,
                    "layers.*.ffn.mlp.experts.down_proj": expert_dtype,
                    "layers.*.ffn.shared_experts.w1.weight": expert_dtype,
                    "layers.*.ffn.shared_experts.w2.weight": expert_dtype,
                    "layers.*.ffn.shared_experts.w3.weight": expert_dtype,
                }
            )
        if attn_dtype and attn_dtype.lower() not in ("bf16", "none", ""):
            # MLA Linear weights that exist in every block. The compressor's
            # `wkv`/`wgate` are fp32 (not packable) and the indexer only
            # exists on a subset of layers, so they're intentionally left
            # off this list to avoid spurious unmatched-pattern warnings.
            overrides.update(
                {
                    "layers.*.attn.wq_a.weight": attn_dtype,
                    "layers.*.attn.wq_b.weight": attn_dtype,
                    "layers.*.attn.wkv.weight": attn_dtype,
                    "layers.*.attn.wo_a.weight": attn_dtype,
                    "layers.*.attn.wo_b.weight": attn_dtype,
                }
            )
        return overrides
