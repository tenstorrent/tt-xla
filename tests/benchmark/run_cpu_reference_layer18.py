# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run the layer-18 GPT-OSS-120B 1-layer model on CPU in bf16 as a reference.

Same model construction as debug_bfp4_layer18.py but:
  * no TT / torch_xla / SPMD,
  * no weight_dtype_overrides (keep all weights bf16 — no bfp4 quantization),
  * same inputs (zeros input_ids, arange cache_position, batch=64, seq=128),
  * saves outputs to tests/benchmark/bfp4_layer18_output/cpu_reference_outputs.pt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "tests" / "benchmark"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARK_DIR))

import torch
from transformers.cache_utils import StaticCache

# Reuse the same loader + wrapper from the debug script.
from debug_bfp4_layer18 import (  # noqa: E402
    BATCH_SIZE,
    INPUT_SEQUENCE_LENGTH,
    MODEL_NAME,
    TARGET_LAYER,
    EXPORT_PATH,
    ModelWithCache,
    load_model_layer18_only,
)


def main():
    out_path = Path(EXPORT_PATH) / "cpu_reference_outputs.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[cpuref] Loading 1-layer model with layer 18's weights...")
    inner_model, config = load_model_layer18_only(MODEL_NAME, TARGET_LAYER)
    inner_model = inner_model.to(torch.bfloat16).eval()

    print(f"[cpuref] Wrapping with StaticCache (batch={BATCH_SIZE}, seq={INPUT_SEQUENCE_LENGTH})...")
    wrapped = ModelWithCache(
        inner_model, config,
        batch_size=BATCH_SIZE, max_cache_len=INPUT_SEQUENCE_LENGTH,
    ).to(torch.bfloat16).eval()
    wrapped._rebind_cache_tensors()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    prompt = "Here is an exaustive list of the best practices for writing clean code:"
    input_ids = tok(
        [prompt] * BATCH_SIZE,
        return_tensors="pt", max_length=INPUT_SEQUENCE_LENGTH,
        padding="max_length", truncation=True,
    )["input_ids"]
    assert input_ids.shape == (BATCH_SIZE, INPUT_SEQUENCE_LENGTH)
    cache_position = torch.arange(0, INPUT_SEQUENCE_LENGTH)

    print("[cpuref] Running forward pass on CPU...")
    with torch.no_grad():
        out = wrapped(input_ids, cache_position)

    # Build a list of every tensor the forward pass produced. Matches the
    # structure the TT run gathers (logits + KV cache updates).
    tensors = []

    # Logits: CausalLMOutputWithPast.logits, shape (batch, seq, vocab).
    if hasattr(out, "logits") and out.logits is not None:
        tensors.append(("logits", out.logits.detach().to(torch.bfloat16).cpu().contiguous()))

    # KV cache state after the forward pass. For 1 layer, there's one entry.
    cache = wrapped.cache
    for i, layer in enumerate(cache.layers):
        tensors.append((f"kv_keys_layer{i}", layer.keys.detach().to(torch.bfloat16).cpu().contiguous()))
        tensors.append((f"kv_values_layer{i}", layer.values.detach().to(torch.bfloat16).cpu().contiguous()))

    print(f"[cpuref] CPU reference produced {len(tensors)} tensors:")
    for name, t in tensors:
        print(f"  {name:28s} shape={tuple(t.shape)} dtype={t.dtype}")

    torch.save(
        {"tensors": [t for _, t in tensors], "names": [n for n, _ in tensors]},
        str(out_path),
    )
    print(f"[cpuref] Wrote {out_path}")


if __name__ == "__main__":
    main()
