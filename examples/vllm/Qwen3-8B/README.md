# Qwen3-8B forge serving — EngineCore host-memory leak repro

Reproduces an unbounded **host-memory leak in the vLLM `EngineCore` process** on
the forge plugin: RSS climbs ~linearly with generated tokens and is **never
released**, even after all requests drain and the server goes idle. Only a
server restart frees it. On a small-RAM host this eventually OOMs the box.

## Files

- `service.sh` — standalone `vllm serve Qwen/Qwen3-8B` (forge plugin: BFP8
  weights + KV, trace, opt1, b32). **Defaults to the recommended fast repro:
  single layer + 2k context + no chunked prefill** (~2-min compile, ~100s flood,
  still leaks). Env knobs flip variants (see header): `NUM_HIDDEN_LAYERS` (`full`
  for all layers), `MAX_MODEL_LEN`, `PREFILL_CHUNK_SIZE` (>0 enables chunked
  prefill + b1-prefill), `MAX_NUM_SEQS`. Full production config:
  `NUM_HIDDEN_LAYERS=full MAX_MODEL_LEN=40960 PREFILL_CHUNK_SIZE=2048 ./service.sh`.
- `leak_probe.py` — floods the server with concurrent `/v1/chat/completions`
  and samples the `VLLM::EngineCore` RSS from `/proc`; prints the trend +
  start/peak/end Δ. Self-contained (stdlib only).

## Run

```
# terminal 1 — server (from the tt-xla venv)
cd ~/tt-xla && source venv/activate
TT_VISIBLE_DEVICES=0 examples/vllm/Qwen3-8B/service.sh

# terminal 2 — once it serves, flood it
python3 examples/vllm/Qwen3-8B/leak_probe.py --port 8000 \
    --num-prompts 200 --concurrency 32 --max-tokens 1024
```

A climbing Δ that does not flatten once the KV pool is full, and does not drop
after the flood drains, is the leak. A one-time rise that plateaus and releases
is just normal KV/activation warmup.

## Observed (Qwen3-8B, P150, conc 32, same ~204k-token flood)

| layers | seq len | chunked prefill + b1 | Δ EngineCore | per-token | flood time | released? |
|---|---|---|---:|---:|---:|---|
| full 36L | 40k | on | +2.11 GB | ~10.4 KB/tok | 505 s | no |
| full 36L | 2k | off | +1.99 GB | ~10.2 KB/tok | 491 s | no |
| 1 layer | 40k | on | +0.64 GB | ~3.1 KB/tok | 102 s | no |
| **1 layer** | **2k** | **off** (default) | +0.63 GB | ~3.1 KB/tok | **100 s** | no |

The leak reproduces standalone (not the tt-media-server wrapper), and the
per-token rate is **independent of seq-len (40k vs 2k) and chunked prefill
(on/off)** — it scales only with model depth (~2.9 KB/tok layer-independent
floor + ~0.2 KB/tok/layer). So this branch's chunked-prefill / long-seq work is
not the cause. Single layer + 2k + no chunked prefill is the fastest repro
(~2-min compile, ~100s flood) and still leaks, which is why it is the default.

## Note

Requires `fastapi < 0.137.0` (pinned in `integrations/vllm_plugin/`): FastAPI
0.137.x has a route-tree regression that makes the Prometheus instrumentator
500 every request.
