# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal device-perf driver for the BGE-m3 flatten-inputs regression (issue #5756).

Runs a single BAAI/bge-m3 embedding forward under tracy so per-op device timings
(ops_perf_results_*.csv, column "DEVICE FW DURATION [ns]") can be captured for the
batched vs flattened model I/O layouts and diffed.

Controlled entirely by env vars so the two variants are byte-identical except for the
`flat_model_io` toggle:

    BGE_FLAT   0 (batched, default) | 1 (flattened)          -> additional_config["flat_model_io"]
    BGE_BATCH  number of prompts / max_num_seqs (default 1)

Usage (from repo root, venv active):

    BGE_FLAT=0 BGE_BATCH=1 tracy -p -r --sync-host-device -n batched_b1 \
        -m pytest -svv tests/integrations/vllm_plugin/pooling/test_flatten_perf.py

    BGE_FLAT=1 BGE_BATCH=1 tracy -p -r --sync-host-device -n flattened_b1 \
        -m pytest -svv tests/integrations/vllm_plugin/pooling/test_flatten_perf.py

No PCC assertion — this is a perf trace, not a correctness gate. PCC is printed if the
committed baseline is available so we can sanity-check the flattened path still computes
correct embeddings.
"""
import os

# Force the vLLM V1 engine to run IN-PROCESS (no EngineCore subprocess). Device op
# profiling under tracy captures host-side per-op zones from the process it launches;
# with the default multiprocessing engine those zones are emitted in a child process
# and never make it into the ops report (empty ops_perf_results.csv). Must be set
# before `import vllm`. Same knob used by generative/test_prefill_recompile.py.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm

MODEL_NAME = "BAAI/bge-m3"
MAX_MODEL_LEN = 64

# A single ~55-token prompt, repeated to fill the batch. Uniform length -> uniform
# padding -> clean, identical matmul shapes between the batched and flattened runs, so
# the only structural difference is tensor rank.
_BASE_PROMPT = (
    "We build computers for artificial intelligence and design high performance graph "
    "processors alongside configurable chips that run a robust software stack for real "
    "time machine learning workloads at very large scale across many devices today."
)


def test_bge_m3_flatten_perf():
    flat = os.environ.get("BGE_FLAT", "0") == "1"
    batch = int(os.environ.get("BGE_BATCH", "1"))
    # Optionally truncate the encoder to N identical layers to keep tracy-instrumented
    # in-process compilation fast. bge-m3's 24 layers are identical and the flatten
    # regression lives in the per-layer non-attention matmuls, so a couple of layers
    # reproduces the exact per-op signature at a fraction of the compile cost
    # (mirrors PROFILING.md's `--num-layers 1` for dense LLMs). 0/unset = full model.
    layers = int(os.environ.get("BGE_LAYERS", "0"))

    prompts = [_BASE_PROMPT] * batch

    llm_args = {
        "model": MODEL_NAME,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "disable_sliding_window": True,
        # Enough token budget to keep the whole batch in a single forward pass.
        "max_num_batched_tokens": max(MAX_MODEL_LEN * batch, MAX_MODEL_LEN),
        "max_num_seqs": batch,
        "enable_prefix_caching": False,
        "additional_config": {
            "flat_model_io": flat,
        },
    }
    if layers > 0:
        llm_args["hf_overrides"] = {"num_hidden_layers": layers}

    print(
        f"\n[bge-m3 flatten-perf] flat_model_io={flat} batch={batch} "
        f"max_model_len={MAX_MODEL_LEN} layers={layers or 'full'}"
    )
    model = vllm.LLM(**llm_args)

    output_embedding = model.embed(prompts)

    print(f"[bge-m3 flatten-perf] produced {len(output_embedding)} embeddings")
    for idx, out in enumerate(output_embedding[: min(2, batch)]):
        embeds = out.outputs.embedding
        print(f"  prompt{idx}: embedding_dim={len(embeds)} head={embeds[:4]}")

    # Optional correctness sanity-check against the committed baseline (single-prompt
    # baseline; only meaningful when the prompt matches, so we just report, never assert).
    baseline_path = os.path.join(
        os.path.dirname(__file__), "baseline", "bge_m3_baseline.pt"
    )
    if os.path.exists(baseline_path):
        try:
            baseline = torch.load(baseline_path)
            golden = baseline.get("prompt0")
            if golden is not None:
                got = torch.tensor(
                    output_embedding[0].outputs.embedding, dtype=torch.float32
                )
                if got.shape == golden.shape:
                    pcc = torch.corrcoef(torch.stack([got, golden]))[0, 1].item()
                    print(f"[bge-m3 flatten-perf] PCC vs baseline prompt0: {pcc:.5f}")
        except Exception as e:  # noqa: BLE001 - diagnostic only
            print(f"[bge-m3 flatten-perf] baseline check skipped: {e}")
