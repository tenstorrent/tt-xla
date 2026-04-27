# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run the full 36-layer GPT-OSS-120B model on CPU in bf16 as a reference.

No TT, no SPMD, no bfp4 quantization. Same inputs as
`debug_bfp4_full_model.py` (zeros input_ids, batch=64, seq=128).
Saves outputs to tests/benchmark/bfp4_full_output/cpu_reference_outputs.pt.

Memory note: full GPT-OSS-120B in bf16 needs ~240 GB CPU RAM after
Mxfp4Config(dequantize=True) expansion. Host has 566 GB; fits.
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

from benchmarks.llm_benchmark import setup_model_and_tokenizer
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

from debug_bfp4_layer18 import ModelWithCache  # reuse cache wrapper

BATCH_SIZE = 64
INPUT_SEQUENCE_LENGTH = 128
EXPORT_PATH = str(BENCHMARK_DIR / "bfp4_full_output")


def main():
    out_path = Path(EXPORT_PATH) / "cpu_reference_outputs.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[cpuref] Loading full 36-layer GPT-OSS-120B on CPU (bf16)...")
    model_loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B)
    inner_model, _ = setup_model_and_tokenizer(model_loader, ModelVariant.GPT_OSS_120B)
    inner_model = inner_model.to(torch.bfloat16).eval()
    config = inner_model.config
    print(f"[cpuref] num_hidden_layers = {config.num_hidden_layers}")

    wrapped = ModelWithCache(
        inner_model, config,
        batch_size=BATCH_SIZE, max_cache_len=INPUT_SEQUENCE_LENGTH,
    ).to(torch.bfloat16).eval()
    wrapped._rebind_cache_tensors()

    input_ids = torch.zeros(BATCH_SIZE, INPUT_SEQUENCE_LENGTH, dtype=torch.long)
    cache_position = torch.arange(0, INPUT_SEQUENCE_LENGTH)

    print("[cpuref] Running forward pass on CPU (this takes a while)...")
    import time
    t0 = time.time()
    with torch.no_grad():
        out = wrapped(input_ids, cache_position)
    dt = time.time() - t0
    print(f"[cpuref] Forward pass took {dt:.1f} s")

    tensors = []
    if hasattr(out, "logits") and out.logits is not None:
        tensors.append(("logits", out.logits.detach().to(torch.bfloat16).cpu().contiguous()))
    for i, layer in enumerate(wrapped.cache.layers):
        tensors.append((f"kv_keys_layer{i}",
                        layer.keys.detach().to(torch.bfloat16).cpu().contiguous()))
        tensors.append((f"kv_values_layer{i}",
                        layer.values.detach().to(torch.bfloat16).cpu().contiguous()))

    print(f"[cpuref] Produced {len(tensors)} tensors. Saving to {out_path}")
    torch.save(
        {"tensors": [t for _, t in tensors], "names": [n for n, _ in tensors]},
        str(out_path),
    )
    print("[cpuref] Done.")


if __name__ == "__main__":
    main()
