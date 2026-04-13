# Metal Trace Investigation for vLLM Models

**Date:** 2026-04-12
**Branch:** `kmabee/vllm_perf_apr12`
**Goal:** Enable `enable_trace` in vLLM plugin for performance improvement

## Background

Metal trace (`enable_trace`) replays compiled programs on device without re-dispatching from host, reducing per-step overhead. It was enabled by default for all benchmark models in commit `71d741794` (with per-model opt-outs for failures). The vLLM plugin never had `enable_trace` wired through `TTConfig`.

Prior attempt in `92d0a81ea` (2026-03-23) added TTConfig plumbing but immediately hit a compiler error.

## Changes Made

Cherry-picked `92d0a81ea` into current branch and resolved merge conflict. The change adds:
- `enable_trace: bool = False` to `TTConfig` in `platform.py`
- `"enable_trace": "true"/"false"` to `get_pjrt_compile_config()`

Created `tests/integrations/vllm_plugin/generative/test_trace_generation.py` with greedy sampling + `cpu_sampling=False` tests for OPT-125m and Llama-3.2-1B.

## Results

| Model | Trace | Sampling | Result | Time |
|-------|-------|----------|--------|------|
| OPT-125m | off (baseline) | non-greedy, cpu | PASS | 88.7s |
| OPT-125m | **on** | greedy, device | **PASS** | 88.9s |
| Llama-3.2-1B | **on** | greedy, device | **FAIL** | 115.6s |

## Root Cause Analysis

### Error
```
error: 'ttnn.capture_or_execute_trace' op All output tensors of trace function must be on device.
%15 = "ttnn.from_device"(%arg6) : (tensor<1x128xsi32, ...dram>) -> tensor<1x128xsi32, ...system_memory> is not on device.
```

### What `%arg6` is

`%arg6` is **position_ids** (`tensor<1x128xi32>`, `ttir.name = "args_1"`). In Llama, position_ids are used for RoPE (rotary position embedding) — they index into the cos/sin cache via a gather op.

### Why OPT works but Llama doesn't

**OPT-125m** uses learned positional embeddings (a weight table lookup). There are no integer position_ids that need host-side reads. The only `from_device` in OPT's prefill graph (`graph_4`) produces `%2` (a `1x1xsi32` scalar), but it's used as an **input** to `capture_or_execute_trace`, not an **output**. Inputs can live on host — the trace `run_and_capture` function writes them to device via `ttnn.write_tensor`.

**Llama-3.2-1B** has position_ids (`1x128xsi32`) flowing through the graph for RoPE. During the TTIR→TTNN lowering + trace insertion pass, position_ids end up needing a `from_device` (to read on host for index computation), and this `from_device` result appears as a **trace output**. The trace verifier rejects this because all trace outputs must remain on-device.

### Graph structure comparison

| | OPT-125m | Llama-3.2-1B |
|---|---|---|
| Extracted TTNN graphs | 15 | 1 (failed during graph 2 compilation) |
| `from_device` in graphs | Yes, but only as trace *inputs* | `from_device(%arg6)` is a trace *output* |
| Position encoding | Learned embeddings (no position_ids) | RoPE (position_ids → gather from cos/sin cache) |
| Failing graph | N/A | `SyncTensorsGraph.3276` (prefill, 1481 TTIR ops) |

### Why non-vLLM benchmark Llama tests work with trace

The benchmark tests (`test_llama_3_2_1b` in `test_llms.py`) use `torch.compile` through `torch_xla`, which generates different graph boundaries than vLLM. The vLLM prefill graph is a single monolithic graph including embedding + all transformer layers + LM head, while `torch.compile` may split subgraphs differently, keeping the position_ids gather outside the traced region.

**This hypothesis needs verification** — running `test_llama_3_2_1b` with debug IR to compare graph structure.

## Extracted IR Artifacts

- `/tmp/opt125m_trace/` — 15 TTNN graphs from OPT with trace (all compiled successfully)
- `/tmp/llama1b_trace/` — 1 TTNN graph from Llama (only init graph; prefill failed)
- `/tmp/llama1b_trace_ttir/` — 2 TTIR graphs from Llama (graph_2 is the 1481-op prefill that fails)
- Debug logs: `test_opt_trace_debug.log` (108k lines), `test_llama_1b_trace_debug.log` (22k lines)

## Benchmark Llama Comparison (2026-04-13)

Ran `test_llama_3_2_1b` benchmark with `--num-layers 2` and trace enabled. **Passed** (PCC=0.997, 506s). Extracted 4 TTNN graphs, all with `capture_or_execute_trace` and **zero `from_device` ops**.

### Key Difference: RoPE Implementation

| | Benchmark (torch.compile) | vLLM (lazy tensor) |
|---|---|---|
| Position_ids shape | `18xsi32` (1D) | `1x128xsi32` (2D) |
| RoPE cos/sin | Pre-computed in `const_eval` functions (host, before trace) | Dynamic gather from `131072x64` cache inside traced graph |
| `from_device` in graph | None | Yes — position_ids read to host for gather indexing |
| Trace result | All outputs on device | `from_device(position_ids)` is a trace output → rejected |

### Root Cause

The benchmark model (HuggingFace transformers via `torch.compile`) uses `const_eval` to pre-compute rotary embeddings from `inv_freq` before the trace runs. The cos/sin values are materialized as device constants — no position_ids needed at trace time.

The vLLM model uses vLLM's own `LlamaForCausalLM` which performs a **dynamic gather** from a pre-built cos/sin cache (`L__self___model_layers_0_self_attn_rotary_emb__forward_method___self___cos_sin_cache`, `131072x64xbf16`) indexed by position_ids. This gather requires position_ids on host, creating the `from_device` that violates the trace constraint.

### Artifacts

- `/tmp/llama1b_benchmark_trace/` — 4 TTNN graphs from benchmark with trace (all pass)
- `test_llama_1b_benchmark_trace_debug.log` — benchmark debug log (6381 lines)

## Benchmark Results (device sampling, greedy, batch=1)

| Model | Trace | TTFT | Decode TPS | Speedup |
|-------|-------|------|------------|---------|
| OPT-125m | off | 18.1ms | 75.3 | — |
| OPT-125m | **on** | 8.2ms | **160.8** | **2.14x** |
| Llama-3.2-1B | off | 41.4ms | 28.0 | — |
| Llama-3.2-1B | **on** | 23.9ms | **49.0** | **1.75x** |
| Llama-3.1-8B | off | 82.6ms | 18.8 | — |
| Llama-3.1-8B | **on** | — | **FAIL** | blocked |

### Interactive Server+Client Validation (OPT-125m)

Using `examples/vllm/OPT-125M/` server+client demo:

| Mode | Decode TPS | TTFT |
|------|------------|------|
| No trace | 73.2-73.3 tok/s | 27ms |
| **Trace** | **157.0 tok/s** | 17ms |

## TTRotaryEmbedding Fix

Added `TTRotaryEmbedding` override in `integrations/vllm_plugin/vllm_tt/overrides.py` that computes cos/sin from `inv_freq` using `torch.outer` + `cos`/`sin` (math ops, all on device) instead of vLLM's `cos_sin_cache.index_select(0, positions)` (gather → `ttir.embedding` → `from_device`).

This unblocks trace for Llama-3.2-1B (**1.75x decode speedup**).

## Llama-3.1-8B Trace Failure (remaining)

8B with `bfp_bf8` + `optimization_level=1` hits a second `from_device` on a different tensor:
```
%22 = "ttnn.from_device"(%arg1) : (tensor<1x1xsi32, ...>) is not on device.
```
This `1x1xsi32` is likely the token embedding gather (input_ids → vocab embedding), not RoPE. Same `ttir.embedding` pattern but for the vocab lookup. Needs further investigation.

## Next Steps

1. Investigate the 8B `1x1xsi32` `from_device` — determine which embedding/gather op it comes from and whether a similar override can fix it.
2. Validate trace for Llama-3.2-1B at batch=32.
3. Once 8B is unblocked, benchmark at production batch sizes.
