# Sampling Overhead Debug: Device vs CPU Sampling

Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940

## Problem

Non-greedy sampling on device is significantly slower than CPU sampling for vLLM on Wormhole single-chip. The overhead is large enough to cut throughput in half regardless of model size, confirming the bottleneck is in the compiled sampling graph itself.

## Setup

- Hardware: Wormhole single-chip
- vLLM direct LLM mode (no server overhead)
- batch_size=1, max_model_len=2048 (8B) / 1024 (OPT)
- 128 output tokens per prompt

---

# Part 1: E2E Throughput Measurements

## Reproduce

```bash
# Synthetic sampling benchmarks (no model loading, ~34s full suite)
pytest -v tests/integrations/vllm_plugin/sampling/test_sampling_perf.py

# E2E model benchmarks
python examples/vllm/opt-125m/chat.py --benchmark --temperature 0.8 --fast
python examples/vllm/opt-125m/chat.py --benchmark --temperature 0.8 --cpu-sampling --fast
python examples/vllm/opt-125m/chat.py --benchmark --temperature 0.0 --fast
```

## E2E Model Benchmarks

### OPT-125M (vocab_size=50272)

| Configuration | tok/s | ms/token | Sampling overhead |
|---|---|---|---|
| Greedy device (temp=0.0) | 11.61 | 86ms | baseline |
| Non-greedy CPU (temp=0.8, top_p=0.9) | 11.00 | 91ms | +5ms (negligible) |
| Non-greedy device (temp=0.8, top_p=0.9) | 5.94 | 168ms | **+82ms** |

### Llama-3.1-8B-Instruct (vocab_size=128256)

| Configuration | tok/s | ms/token | Sampling overhead |
|---|---|---|---|
| Greedy device (temp=0.0) | 9.82 | 102ms | baseline |
| Non-greedy CPU (temp=0.8, top_p=0.9) | 7.89 | 127ms | +25ms |
| Non-greedy device (temp=0.8, top_p=0.9) | 4.02 | 249ms | **+147ms** |

## Synthetic Sampling Graph (isolated, no model, 50 iterations)

| Test | Backend | Vocab | P50 ms | tok/s |
|---|---|---|---|---|
| greedy | device | 50K | 0.28 | 3599 |
| greedy | device | 128K | 0.30 | 3316 |
| greedy | cpu | 50K | 0.07 | 13403 |
| greedy | cpu | 128K | 0.18 | 5510 |
| non_greedy | device | 50K | 63.59 | 19 |
| **non_greedy** | **device** | **128K** | **144.23** | **8** |
| non_greedy | cpu | 50K | 3.58 | 255 |
| non_greedy | cpu | 128K | 8.15 | 123 |
| temp_only | device | 50K | 65.01 | 16 |
| **temp_only** | **device** | **128K** | **145.93** | **8** |
| temp_only | cpu | 50K | 3.53 | 245 |
| temp_only | cpu | 128K | 8.23 | 97 |

## Key Findings from E2E / Synthetic

1. **Greedy device is essentially free** — 0.28ms, no optimization needed.

2. **temp_only and non_greedy are the same speed on device** (65ms vs 64ms at 50K, 146ms vs 144ms at 128K). Top-p filtering adds zero overhead — the entire cost is in the shared path (temperature scaling + softmax + multinomial or sort that runs unconditionally).

3. **Consistent 18x device/CPU ratio** across all non-greedy configs and vocab sizes. The overhead is a fixed multiplier, not a fixed cost.

4. **Cost scales super-linearly with vocab size on device.** 50K→128K is 2.5x vocab but 2.3x p50 on device (64ms→144ms). On CPU the scaling is similar (3.6ms→8.2ms).

5. **Large gap between synthetic and e2e.** Synthetic non-greedy p50=64ms at 50K, but e2e is 168ms/token. ~100ms of non-sampling overhead exists in the vLLM pipeline (model forward, logits, metadata, scheduler).

## Suspicious: vLLM Pipeline Overhead Dominates Model Compute

Greedy device baselines are surprisingly close across very different model sizes:

| Model | Params | Greedy device ms/token |
|---|---|---|
| OPT-125M | 125M | 86ms |
| Llama-3.1-8B | 8B | 102ms |

An 8B model is 64x larger than 125M yet only 16ms slower. This suggests the actual model forward pass is a small fraction of the per-token time, and ~80-90ms of fixed vLLM pipeline overhead (scheduler, metadata construction, device-host transfers) dominates at batch=1. This is a separate performance issue from sampling but worth investigating — it sets the floor for tok/s regardless of sampling improvements.

---

# Part 2: Tracy Device Profiling Deep Dive

## How to capture

```bash
# Non-greedy on device
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tracy -p -r --sync-host-device \
  -o perf_debug_e2e/tracy/e2e_non_greedy_inproc_longprefill \
  examples/vllm/opt-125m/chat.py --benchmark --fast --temperature 0.8 --max-tokens 4

# Greedy on device
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tracy -p -r --sync-host-device \
  -o perf_debug_e2e/tracy/e2e_greedy_inproc_longprefill \
  examples/vllm/opt-125m/chat.py --benchmark --fast --temperature 0 --max-tokens 4
```

Key flags:
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` — run engine in-process so Tracy can see device ops
- `-p` — only profile enabled zones (avoids 32K source location overflow)
- `--sync-host-device` — correlate host and device timelines

## Important: Tracy GUI shows host data, not device data

The `.tracy` file (`tracy_profile_log_host.tracy`) contains **host-side** data only. The "Device 0: Core(x,y)" rows in the Tracy GUI show when the **host dispatched** ops to those cores, not when the device actually executed them. The actual device execution timing is only in the CSV files (`ops_perf_results_*.csv`) via device cycle timestamps.

The "clusters" visible in the Tracy GUI are host dispatch bursts. The "gaps" between clusters are real host idle time (Python/vLLM overhead). But the device runs continuously with no gaps — it queues dispatched ops and processes them back-to-back.

## Greedy vs Non-Greedy Comparison (OPT-125M, vocab=50272)

Measured from device cycle timestamps in `ops_perf_results_*.csv`. Averages across tokens 2 and 3 (steady-state decode).

### Per-Token Device Breakdown

```
                  Fwd Ops  Fwd FW  Fwd Span  Smp Ops  Smp FW  Smp Span  Tot FW  Tot Span  Tok2Tok
Greedy Tok2          382  88.4ms   315.7ms        1   0.8ms     0.8ms  89.1ms   316.5ms  316.5ms
Greedy Tok3          382  88.3ms   282.7ms        1   0.8ms     0.8ms  89.1ms   283.5ms  283.5ms
NonGreedy Tok2       420  93.7ms   316.2ms       66  94.0ms    67.3ms 187.6ms   383.5ms  383.5ms
NonGreedy Tok3       420  93.7ms   338.8ms       66  94.1ms    62.4ms 187.8ms   401.3ms  401.3ms
```

- **FW** = sum of DEVICE FW DURATION across all ops (actual compute time)
- **Span** = first device op start cycle → last device op end cycle (wall-clock on device)
- **Tok2Tok** = device span from one token's forward start to next token's forward start

### Averages

| Metric | Greedy | Non-Greedy | Delta |
|---|---|---|---|
| Forward FW sum | 88.3ms | 93.7ms | +5.3ms |
| Forward dev span | 299.2ms | 327.5ms | +28.3ms |
| Sampling FW sum | 0.8ms | 94.1ms | **+93.3ms** |
| Sampling dev span | 0.8ms | 64.9ms | +64.1ms |
| Total FW sum | 89.1ms | 187.7ms | +98.6ms |
| Total dev span | 300.0ms | 392.4ms | +92.4ms |
| Tok-to-tok (device) | 300.0ms | 392.4ms | +92.4ms |
| Tok/s (under Tracy) | 3.3 | 2.5 | -0.8 |

### Key Observations

1. **Device runs continuously with no idle gaps.** Device gaps between phases are ~1 microsecond. All "gaps" visible in the Tracy GUI are host-side.

2. **Sampling adds 93ms of FW compute and 92ms of device span per token.** This is the direct cost of the non-greedy sampling graph.

3. **Forward is nearly identical in both configs.** 88.3ms vs 93.7ms FW — the +5ms is noise or slight overhead from different graph shapes. The model forward itself is unaffected by sampling mode.

4. **FW sum is much less than dev span.** Greedy: 89ms FW in 300ms span. Non-greedy: 188ms FW in 392ms span. The gap (span - FW) is op-to-op dispatch overhead within the device command queue — individual ops finish quickly but there's latency between consecutive ops on device.

5. **Forward span (300ms) dwarfs forward FW sum (88ms).** The model forward pass uses only ~30% of its device span for actual compute. The other 70% is device-side op dispatch overhead. This is a significant inefficiency separate from sampling.

### ASCII Diagram: Token Lifecycle (Non-Greedy, Token 2)

```
DEVICE TIMELINE (383.5ms total, continuous — zero gaps)
├──────────── Forward: 316.2ms span ────────────┤├── Sampling: 67.3ms span ──┤
│  420 ops, FW sum = 93.7ms                     ││  66 ops, FW sum = 94.0ms  │
│  12x SDPA layers, matmul, layernorm,          ││  Sort = 14.4ms            │
│  embedding, KV cache, layout ops              ││  Accum = 24.0ms (1 core!) │
│  (device busy ~30% of span)                   ││  FillPad, Softmax, etc.   │
│                                               ││  (device busy ~140%*)     │
├───────────────────────────────────────────────┤├───────────────────────────┤

* Sampling FW > span because ops overlap on different cores

HOST DISPATCH VIEW (what Tracy GUI shows):
[~100ms dispatch burst] ────── ~280ms Python/vLLM work ────── [next burst]
  (host dispatches all ops)    (scheduler, metadata, .cpu()
                                sync, next token prep)

The host dispatches ops in ~100ms bursts. Device queues them and runs
for ~383ms continuously. Host is idle while device processes the queue.
```

### ASCII Diagram: Token Lifecycle (Greedy, Token 2)

```
DEVICE TIMELINE (316.5ms total, continuous)
├──────────── Forward: 315.7ms span ────────────────────┤├ Samp: 0.8ms ┤
│  382 ops, FW sum = 88.4ms                             ││  ArgMax     │
│  Same model forward as non-greedy                     ││  (trivial)  │
│  (device busy ~28% of span)                           ││             │
├───────────────────────────────────────────────────────┤├─────────────┤
```

## Compiled Graph Identification (from extracted TTNN IR)

17 compiled graphs for non-greedy OPT-125M (`--max-tokens 4`):

| Graph | Ops | Purpose | Key ops |
|---|---|---|---|
| 1 | 2 | KV cache init | `full` |
| 2 | 327 | Prefill backbone | `linear`=48 |
| 3 | 435 | Prefill backbone (longer context) | `linear`=48 |
| 4-7 | 435 each | Decode backbone (4 context variants) | `linear`=48, `sdpa`=12 |
| 8-12 | 23-24 each | Compute logits (5 variants) | `embedding`, `concat` |
| 13 | 3 | Select hidden states | `matmul` |
| 14 | 16 | Structured decoding | `bitwise_and` |
| 15 | 106 | **Non-greedy sampling** | `sort`, `cumsum`, `softmax`, `rand` |
| 16 | 7 | **Greedy sampling** | `argmax` |
| 17 | 37 | Gather logprobs | `sort`, `log`, `exp` |

Each decode token executes: backbone (graph 4-7) → compute logits (graph 8-12) → select hidden states (graph 13) → sampling (graph 15 or 16).

Annotated CSVs with per-token forward/sampling boundaries:
- `perf_debug_e2e/tracy/e2e_non_greedy_inproc_longprefill/reports/.../ops_perf_results_generate_annotated.csv`
- `perf_debug_e2e/tracy/e2e_greedy_inproc_longprefill/reports/.../ops_perf_results_generate_annotated.csv`

## Sampling Graph Top Ops (from ops_perf_results CSV)

The non-greedy sampling graph (graph 15) adds 66 device ops and 94ms FW per token. Ops exclusive to sampling (not in forward):

| Op (CSV name) | TTNN IR op | FW per token | Cores | Notes |
|---|---|---|---|---|
| SortDeviceOperation | `ttnn.sort` | 14.4ms | 110 | Top-k/top-p sort over vocab |
| AccumulationDeviceOperation | `ttnn.cumsum` | 24.0ms | **1** | Cumulative sum for top-p — serialized! |
| FillPadDeviceOperation | `ttnn.full` (x11) | ~14ms | **1** | Tensor initialization — serialized! |
| SoftmaxDeviceOperation | `ttnn.softmax` (x3) | ~7ms | 110 | Probability distribution |
| RandDeviceOperation | `ttnn.rand` | <1ms | 110 | Random number generation |

Ops like Permute, Slice, Typecast, Tilize/Untilize appear in both forward and sampling graphs, but the annotated CSV marks exact row boundaries per token so every op is attributable to either forward or sampling. The IR graphs (graph 15 = sampling, graphs 4-7 = forward backbone) also confirm which ops belong where.

## Tracy Output Files

| File | Description |
|---|---|
| `reports/<ts>/ops_perf_results_<ts>.csv` | Per-op device perf (OP CODE, FW DURATION, KERNEL DURATION, CORE COUNT) |
| `.logs/tracy_profile_log_host.tracy` | Host-side trace, open in Tracy GUI (v0.10). Shows dispatch patterns, not device execution. |
| `.logs/cpp_device_perf_report.csv` | Raw device perf report |
| `.logs/sync_device_info.csv` | Host-device clock sync data (from --sync-host-device) |

---

# Conclusions and Next Steps

## Where time is spent (non-greedy device, per token, under Tracy)

The non-greedy sampling graph adds **+93ms FW / +92ms device span** per token compared to greedy. This is purely the sampling graph cost.

However, the larger issue may be the **device-side op dispatch overhead**: forward FW sum is only 88ms but forward span is 300ms. The device spends ~70% of its time in op-to-op gaps within the command queue. This affects both greedy and non-greedy equally.

## Two independent problems

1. **Sampling graph cost (+93ms FW per token):** Direct compute overhead from Sort, Accumulation (1 core!), FillPad (1 core!), Softmax, and layout ops in the sampling graph. Eliminating this (e.g. CPU sampling) would improve tok/s by ~25% under Tracy.

2. **Device op dispatch overhead (70% of forward span is idle):** The device command queue has high latency between consecutive ops. Forward does 88ms of compute in 300ms of span. This is a runtime/compiler issue affecting all workloads, not specific to sampling.

## Actionable next steps

- [ ] **Accumulation on 1 core (24ms)** — why is cumsum not parallelized? This is the single biggest sampling op.
- [ ] **FillPad on 1 core (14ms)** — 11 `ttnn.full` calls serialized on 1 core. Can these be batched or parallelized?
- [ ] **Sort (14ms)** — already on 110 cores. Check if a top-k alternative would be cheaper than full sort.
- [ ] **Device op dispatch overhead** — 70% of forward span is op-to-op gaps. Investigate whether metal trace mode (`enable_trace=True`) or op fusion can reduce this.
- [ ] **CPU sampling as interim default** — CPU sampling adds only 0.8ms (greedy) to 5ms (non-greedy at 50K vocab). May be the pragmatic choice until device sampling is optimized.

---

# What's in the Sampling Graph

The device sampling path (`cpu_sampling=False`) compiles via `torch.compile(backend="tt")`:
- `integrations/vllm_plugin/vllm_tt/model_runner.py`: `sample_from_logits()` (line ~2131)
- `integrations/vllm_plugin/vllm_tt/sampler.py`: `Sampler` module with XLA-friendly ops
- Operations: temperature scaling, top-k/top-p masking, softmax, multinomial sampling

The CPU path (`cpu_sampling=True`) does all of the above on CPU after pulling logits from device:
- `integrations/vllm_plugin/vllm_tt/model_runner.py`: `sample_from_logits_cpu()` (line ~2152)

## Compilation Cost

Engine init times (profile + KV cache + warmup):
- OPT-125M: ~100s
- Llama-3.1-8B: ~467s

---

# Benchmarking Approach

Two levels of benchmarking, each measuring different things:

**Synthetic perf tests** (`tests/integrations/vllm_plugin/sampling/test_sampling_perf.py`):
- Isolate the sampling graph cost with no model loading (~34s total for full suite, 50 iterations default)
- Use for rapid iteration on sampler changes — modify an op, rerun in seconds
- Results written to `perf_debug/sampling_perf_*.json`

**E2E model benchmarks** (`examples/vllm/opt-125m/chat.py --benchmark`):
- Measure the full decode loop: model forward, logit computation, metadata construction, device-host transfers, vLLM scheduler
- Use to verify that sampling graph improvements land in real tok/s

The gap between synthetic and e2e reveals non-sampling overhead in the pipeline. Once the sampling graph is optimized, e2e tests reveal the next bottleneck.

---

# Resolution: Multi-Core TopK (2026-03-27)

Investigation continued in `perf_debug/ttnn_sampling_integration_plan.md`. Key discoveries:

## Root Cause: Single-Core TopK

Tracy profiling of `ttnn.topk()` revealed CORE_COUNT=1 out of 110 available. The topk kernel has a fully implemented multi-core path (bitonic sort) but it's gated behind:
- Input dimension must be **power of 2**
- Input dimension must be **< 65536** (uint16 index range)
- k must be **<= 64**

Our vocab (128,256) fails the first two conditions. Source: `topk_device_operation.cpp`, `topk_utils.cpp`, `topk_constants.hpp`.

## Fix: Pad to Power-of-2 Chunks

Split 128K vocab into 4 chunks of 32064, pad each to 32768 (largest power of 2 under 65536). Multi-core topk triggers automatically.

| Approach | TopK Cores | Device Time |
|---|---|---|
| topk(128256) direct | 1 | 19.85ms |
| 2× topk(64128) split-in-half | 1 each | 18.76ms |
| **4× topk(32768) padded** | **110 each** | **0.89ms total** |

The `ttnn.sampling()` op (0.03ms on 32 cores) handles the final sampling on the reduced 128-token candidate set. Total pipeline: **0.89ms** — matching greedy argmax.

## Compile-Through Approach

`torch.topk` currently compiles to `ttnn.sort` (not `ttnn.topk`) via tt-mlir. A composite op PR exists to fix this:
- **tt-mlir**: [tenstorrent/tt-mlir#7504](https://github.com/tenstorrent/tt-mlir/pull/7504) — lowers `tenstorrent.topk` composite → `ttnn.topk` (merged)
- **tt-xla**: [tenstorrent/tt-xla#3729](https://github.com/tenstorrent/tt-xla/pull/3729) — wraps `torch.topk` as StableHLO composite (merged, temporarily reverted, re-landing soon)

Once both are available, the fix is rewriting `sampler.py` to use `torch.topk` on padded 32768-wide chunks. No custom ops, no new tt-mlir plumbing.

## Updated Performance Table (Llama 3.1 8B, vocab=128K)

| Path | Batch 1 | Batch 32 | Notes |
|---|---|---|---|
| Current device (66-op graph) | ~147ms | ~147ms | ships today |
| **Proposed device (multi-core topk)** | **~0.89ms** | **~0.89ms** | sampler rewrite + topk lowering |
| Greedy argmax (device) | <1ms | <1ms | target |

## Original Bottleneck Ops — Status

| Op | Time | Cores | Status |
|---|---|---|---|
| Accumulation (cumsum, 1 core) | 24ms | 1 | **Eliminated** — topk replaces sort+cumsum |
| FillPad (ttnn.full ×11, 1 core) | 14ms | 1 | **Eliminated** — not needed with topk |
| Sort (14ms) | 14ms | 110 | **Replaced** by multi-core topk (0.18ms/chunk) |
| Softmax ×3 (7ms) | 7ms | 110 | **Reduced** — runs on ~128 tokens, not 128K |

All original bottleneck ops are either eliminated or reduced to negligible cost by pre-filtering with topk.

## Related Docs

- `perf_debug/ttnn_sampling_integration_plan.md` — full integration plan with multi-core topk
- `perf_debug/cpu_sampling_optimization.md` — CPU fallback optimizations (back pocket)
- `perf_debug/test_ttnn_sampling_direct.py` — verification tests and benchmarks

# Resolved Next Steps

- [x] ~~Investigate 1-core Accumulation and FillPad ops~~ — eliminated by topk pre-filtering
- [x] ~~Sort (14ms)~~ — replaced by multi-core topk (0.18ms/chunk)
- [x] ~~Investigate whether a hybrid approach is viable~~ — device path achieves sub-1ms
- [ ] Rebuild with `TTXLA_TRACY_ZONES=ON` for named PJRT host zones in Tracy
- [ ] Investigate vLLM pipeline overhead (~80-90ms fixed cost per token at batch=1)

---

# Log Files

- `perf_debug/llama3.1_8b_non_greedy_device.log`
- `perf_debug/llama3.1_8b_non_greedy_cpu.log`
- `perf_debug/llama3.1_8b_greedy_device.log`
- `perf_debug/opt125m_non_greedy_device.log`
- `perf_debug/opt125m_non_greedy_cpu.log`
- `perf_debug/opt125m_greedy_device.log`
- `perf_debug/sampling_perf_*.json` — synthetic benchmark JSON outputs
- `perf_debug_e2e/tracy/` — Tracy profiling runs
