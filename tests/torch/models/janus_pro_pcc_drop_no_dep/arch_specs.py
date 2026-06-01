# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Layer-0 ``input_layernorm`` + ``self_attn`` dimensions (Janus-Pro decoder).

Pro-7B (from ``janus_decoder_arc.txt`` language_model layer 0)::

    (input_layernorm): LlamaRMSNorm((4096,), eps=1e-06)
    (self_attn): LlamaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
    )
    30 x LlamaDecoderLayer; MLP gate/up 4096 -> 11008

Pro-1B (forge ImageTokenStep decode tests; same op layout at hidden_size=2048)::

    LlamaRMSNorm((2048,), eps=1e-06)
    q/k/v/o_proj: Linear(2048, 2048); 16 heads (head_dim=128); 24 layers
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from transformers import LlamaConfig

HEAD_DIM = 128


@dataclass(frozen=True)
class Layer0ModuleSpec:
    """Submodule shapes for layer-0 LN + attention."""

    variant: str
    hidden_size: int
    rms_norm_eps: float
    q_proj: tuple[int, int]
    k_proj: tuple[int, int]
    v_proj: tuple[int, int]
    o_proj: tuple[int, int]
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    decode_batch: int = 2
    decode_seq_len: int = 1
    prefill_len: int = 128

    def to_llama_config(self) -> LlamaConfig:
        # Match Janus ``load_mmgpt(..., attn_implementation="eager")`` — required for
        # standalone ``LlamaAttention`` (transformers 5.5+ dispatches via ``_attn_implementation``).
        config = LlamaConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_hidden_layers=self.num_hidden_layers,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            attention_dropout=0.0,
            attention_bias=False,
            use_cache=True,
            attn_implementation="eager",
        )
        config._attn_implementation = "eager"
        return config

    def summary_lines(self) -> list[str]:
        h = self.hidden_size
        return [
            f"variant={self.variant}  hidden_size={h}  layers={self.num_hidden_layers}",
            f"  input_layernorm: LlamaRMSNorm(({h},), eps={self.rms_norm_eps})",
            "  self_attn: LlamaAttention(",
            f"    q_proj: Linear{self.q_proj}  k_proj: Linear{self.k_proj}",
            f"    v_proj: Linear{self.v_proj}  o_proj: Linear{self.o_proj}",
            f"  num_heads={self.num_attention_heads}  kv_heads={self.num_key_value_heads}  "
            f"head_dim={HEAD_DIM}",
        ]


JANUS_PRO_1B_LAYER0 = Layer0ModuleSpec(
    variant="Pro_1B",
    hidden_size=2048,
    rms_norm_eps=1e-6,
    q_proj=(2048, 2048),
    k_proj=(2048, 2048),
    v_proj=(2048, 2048),
    o_proj=(2048, 2048),
    num_attention_heads=16,
    num_key_value_heads=16,
    num_hidden_layers=24,
    intermediate_size=5632,
    max_position_embeddings=16384,
    vocab_size=102400,
)

JANUS_PRO_7B_LAYER0 = Layer0ModuleSpec(
    variant="Pro_7B",
    hidden_size=4096,
    rms_norm_eps=1e-6,
    q_proj=(4096, 4096),
    k_proj=(4096, 4096),
    v_proj=(4096, 4096),
    o_proj=(4096, 4096),
    num_attention_heads=32,
    num_key_value_heads=32,
    num_hidden_layers=30,
    intermediate_size=11008,
    max_position_embeddings=16384,
    vocab_size=102400,
)


def parse_layer0_from_janus_decoder_arc(arc_path: str | Path) -> Layer0ModuleSpec:
    """Parse Pro-7B layer-0 LN + attn Linear dims from ``janus_decoder_arc.txt``."""
    text = Path(arc_path).read_text()
    lm = text.split("(language_model): LlamaForCausalLM", 1)[-1]
    block = lm.split("(layers): ModuleList", 1)[-1]
    layer0 = block.split("(1):", 1)[0]

    rms_m = re.search(
        r"\(input_layernorm\): LlamaRMSNorm\(\((\d+),\),\s*eps=([\d.e+-]+)\)", layer0
    )
    if not rms_m:
        raise ValueError("input_layernorm not found in arc dump")
    hidden_size = int(rms_m.group(1))
    rms_eps = float(rms_m.group(2))

    def _proj(name: str) -> tuple[int, int]:
        m = re.search(
            rf"\({name}\): Linear\(in_features=(\d+), out_features=(\d+)", layer0
        )
        if not m:
            raise ValueError(f"{name} not found in arc dump")
        return int(m.group(1)), int(m.group(2))

    q_proj = _proj("q_proj")
    k_proj = _proj("k_proj")
    v_proj = _proj("v_proj")
    o_proj = _proj("o_proj")

    layers_m = re.search(r"\(0-(\d+)\):\s+(\d+)\s+x\s+LlamaDecoderLayer", block)
    num_layers = int(layers_m.group(2)) if layers_m else 30

    gate_m = re.search(
        r"\(gate_proj\): Linear\(in_features=\d+,\s*out_features=(\d+)", layer0
    )
    intermediate_size = int(gate_m.group(1)) if gate_m else hidden_size * 4

    embed_m = re.search(r"\(embed_tokens\): Embedding\((\d+),\s*(\d+)\)", lm)
    vocab_size = int(embed_m.group(1)) if embed_m else 102400

    num_heads = hidden_size // HEAD_DIM

    return Layer0ModuleSpec(
        variant="Pro_7B",
        hidden_size=hidden_size,
        rms_norm_eps=rms_eps,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        num_hidden_layers=num_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=16384,
        vocab_size=vocab_size,
    )


def resolve_janus_decoder_arc_path() -> Path | None:
    env = os.environ.get("JANUS_DECODER_ARC_PATH")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    candidates = [
        Path(__file__).resolve().parents[4] / "janus_decoder_arc.txt",
        Path("/proj_sw/user_dev/ctr-akannan/28_may_yyz/tt-xla/janus_decoder_arc.txt"),
        Path("/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla/janus_decoder_arc.txt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def get_layer0_spec(variant: str) -> Layer0ModuleSpec:
    if variant in ("Pro_1B", "1B"):
        return JANUS_PRO_1B_LAYER0
    if variant in ("Pro_7B", "7B"):
        arc = resolve_janus_decoder_arc_path()
        if arc is not None:
            parsed = parse_layer0_from_janus_decoder_arc(arc)
            assert parsed.hidden_size == JANUS_PRO_7B_LAYER0.hidden_size
            return parsed
        return JANUS_PRO_7B_LAYER0
    raise ValueError(f"Unknown variant {variant!r}; use Pro_1B or Pro_7B")
