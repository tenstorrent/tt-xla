loader_path: third_party.tt_forge_models.causallm_7b_gguf.causal_lm.pytorch.loader
variant_id: CausalLM_7B_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_causallm_7b_q4_k_m_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 22.7819
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "OOM at all optimization levels: opt_level=2 and opt_level=1 fail during warmup with INTERNAL Error code 13 (DRAM OOM); opt_level=0 perf benchmark runs (~489ms/iter decode) but PCC verification OOMs with bank_manager.cpp:462 (allocated 12.6 GB of ~12.7 GB DRAM, only 5 MB free, needs 32 MB for logits tensor at batch_size=32)"

# Benchmark added: test_causallm_7b_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_causallm_7b_q4_k_m_gguf

## Model
- HF name:    TheBloke/CausalLM-7B-GGUF
- Loader:     third_party.tt_forge_models.causallm_7b_gguf.causal_lm.pytorch.loader
- Variant:    CausalLM_7B_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (OOM before measurement)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         ~0:10:00 (opt_level=2), ~0:10:00 (opt_level=1), ~0:22:00 (opt_level=0)
- Hardware:           n150 (wormhole_b0, single-chip assumption on n300)

## Failure summary
All optimization levels fail for the full 7B model at batch_size=32 with bfp_bf8:

- **optimization_level=2**: RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 during perf warmup (SRAM/DRAM OOM during compiled kernel execution)
- **optimization_level=1**: Same RuntimeError: INTERNAL: Error code: 13 during perf warmup
- **optimization_level=0**: Perf benchmark SUCCEEDS (decode ~489ms/iter with 110 steps), but PCC verification OOMs with explicit error:
  - `Out of Memory: Not enough space to allocate 33554432 B DRAM buffer across 12 banks`
  - `allocated: 1066433728 B, free: 5388064 B, largest free block: 2031072 B`
  - Only 5 MB DRAM free when trying to allocate 32 MB logits tensor

Note: num_layers=1 run passes cleanly at opt_level=2 with PCC=0.9997 prefill / 0.9991 decode.
This confirms the architecture is supported; the full model simply exceeds n150 DRAM capacity
at batch_size=32 for the complete 7B parameter set + KV cache + logits.

## Infrastructure fix included
Fixed `tests/benchmark/benchmarks/llm_benchmark.py` line 477: changed `else:` to
`elif hasattr(model_loader, "get_weight_dtype_config_path"):` to gracefully handle loaders
that don't implement `get_weight_dtype_config_path()`. This was a regression in the latest
`[Benchmark] Make sure decode PCC comparison uses same input as golden (#4661)` commit which
reverted an earlier `hasattr` guard (visible in git log of `392e200b9`).

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_causallm_7b_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (DONE_FAIL)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             456440938624
- breakdown.matmul:        456440938624
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  7720931526
- effective_count:        7098601670
- memory_bytes:           8787174164
- memory_gb:              8.18
- effective_memory_bytes: 7542514452
- effective_memory_gb:    7.02
- embedding_count:        622329856
- embedding_memory_bytes: 1244659712

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7819
- top_perf_time_ms:         43.8944
- dram_time_ms:             29.2629
- compute_time_ms_lofi:     1.7830
- compute_time_ms_hifi2:    3.5659
- compute_time_ms_hifi3:    5.3489
- compute_time_ms_hifi4:    7.1319

## Files changed
- tests/benchmark/test_llms.py (added test_causallm_7b_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
