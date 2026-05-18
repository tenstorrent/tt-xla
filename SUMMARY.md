loader_path: third_party.tt_forge_models.afmck_testing_llama_tiny.causal_lm.pytorch.loader
variant_id: testing_llama_tiny
arch: p150
status: DONE_PASS
test_function: test_testing_llama_tiny
samples_per_second: 597.65
ttft_ms: 24.84
prefill_pcc: 0.998722
first_decode_pcc: 0.997942
top_perf_samples_per_sec: 5674.161
pct_of_target: 10.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_testing_llama_tiny

## Test
tests/benchmark/test_llms.py::test_testing_llama_tiny

## Model
- HF name:    afmck/testing-llama-tiny
- Loader:     third_party.tt_forge_models.afmck_testing_llama_tiny.causal_lm.pytorch.loader
- Variant:    testing_llama_tiny

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  597.65
- TTFT (ms):          24.84
- Prefill PCC:        0.998722
- First decode PCC:   0.997942
- Wall clock:         0:00:45
- Hardware:           p150 (Blackhole, single chip; TT_MESH_GRAPH_DESC_PATH required for device init)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_testing_llama_tiny_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 597.65 / 5674.16 = 10.5%

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             47815066880
- breakdown.matmul:        47815066880
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        660
- memory_bytes: 2640

### KV cache
- count:        20971520
- memory_bytes: 41943040
- memory_gb:    0.0390625

### Params
- count:                  53745315
- effective_count:        37361315
- memory_bytes:           72470152
- memory_gb:              0.06749308854341507
- effective_memory_bytes: 39702152
- effective_memory_gb:    0.03697551041841507
- embedding_count:        16384000
- embedding_memory_bytes: 32768000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 5674.1610
- top_perf_time_ms:         0.1762
- dram_time_ms:             0.1175
- compute_time_ms_lofi:     0.0543
- compute_time_ms_hifi2:    0.1087
- compute_time_ms_hifi3:    0.1630
- compute_time_ms_hifi4:    0.2173

## Files changed
- tests/benchmark/test_llms.py (added test_testing_llama_tiny)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added testing_llama_tiny entry)
- SUMMARY.md (this file)

## Notes
- This machine has 4 Blackhole chips (PCI 0xb140) but only /dev/tenstorrent/0 is accessible in the container.
  ARCH_NAME=wormhole_b0 is set as a legacy env var but actual hardware is Blackhole → arch=p150.
  TT_MESH_GRAPH_DESC_PATH must be set (to p150_mesh_graph_descriptor.textproto) for device initialization;
  without it the UMD detects CUSTOM cluster type and aborts.
- llm_benchmark.py was patched: get_weight_dtype_config_path() is not implemented by all loaders;
  added hasattr guard matching the existing pattern in dynamic_torch_model_tester.py.
- The 10.5% of roofline is expected for this micro test model; framework and compilation overhead
  dominates for a model this small.

## tt-forge-models submodule
no change
