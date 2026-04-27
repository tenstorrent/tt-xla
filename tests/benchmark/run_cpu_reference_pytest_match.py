# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the layer-18 pytest-match config (17-token prefill)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "tests" / "benchmark"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARK_DIR))

import torch
import tt_torch  # noqa: F401  (registers the "dense" experts impl with HF)
from transformers import AutoTokenizer

from debug_bfp4_layer18 import (
    MODEL_NAME, TARGET_LAYER, BATCH_SIZE,
    ModelWithCache, load_model_layer18_only,
)

MAX_CACHE_LEN = 128
EXPORT_PATH = str(BENCHMARK_DIR / "bfp4_layer18_output_pytest_match")


def main():
    out_path = Path(EXPORT_PATH) / "cpu_reference_outputs.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inner_model, config = load_model_layer18_only(MODEL_NAME, TARGET_LAYER)
    inner_model = inner_model.to(torch.bfloat16).eval()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    prompt = "Here is an exaustive list of the best practices for writing clean code:"
    tokens = tok([prompt], return_tensors="pt", max_length=MAX_CACHE_LEN, truncation=True)["input_ids"]
    prefill_len = tokens.shape[1]
    print(f"[cpuref] prefill_len = {prefill_len}")
    input_ids = tokens.expand(BATCH_SIZE, -1).contiguous()
    cache_position = torch.arange(0, prefill_len)

    wrapped = ModelWithCache(
        inner_model, config,
        batch_size=BATCH_SIZE, max_cache_len=MAX_CACHE_LEN,
    ).to(torch.bfloat16).eval()
    wrapped._rebind_cache_tensors()

    print("[cpuref] Forward pass...")
    with torch.no_grad():
        out = wrapped(input_ids, cache_position)

    tensors = []
    if hasattr(out, "logits") and out.logits is not None:
        tensors.append(("logits", out.logits.detach().to(torch.bfloat16).cpu().contiguous()))
    for i, layer in enumerate(wrapped.cache.layers):
        tensors.append((f"kv_keys_layer{i}",
                        layer.keys.detach().to(torch.bfloat16).cpu().contiguous()))
        tensors.append((f"kv_values_layer{i}",
                        layer.values.detach().to(torch.bfloat16).cpu().contiguous()))

    print(f"[cpuref] Produced {len(tensors)} tensors. Logits shape: {tensors[0][1].shape}")
    torch.save(
        {"tensors": [t for _, t in tensors], "names": [n for n, _ in tensors]},
        str(out_path),
    )
    print(f"[cpuref] Wrote {out_path}")


if __name__ == "__main__":
    main()
