loader_path: third_party.tt_forge_models.gritlm.causal_lm.pytorch.loader
variant_id: emb_m7_nodes16_fast
arch: p150
status: DONE_FAIL
test_function: test_gritlm_emb_m7_nodes16_fast
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 44.8551
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler error: failed to legalize operation 'ttir.paged_update_cache' in TTIRToTTNNCommon pipeline during full model run"

# Benchmark added: test_gritlm_emb_m7_nodes16_fast

## Test
tests/benchmark/test_llms.py::test_gritlm_emb_m7_nodes16_fast

## Model
- HF name:    GritLM/emb_m7_nodes16_fast
- Loader:     third_party.tt_forge_models.gritlm.causal_lm.pytorch.loader
- Variant:    ModelVariant.EMB_M7_NODES16_FAST

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (full model failed)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         10:30 (failed before benchmarking)
- Hardware:           p150

## Notes
- Bring-up at num_layers=1, max_output_tokens=3 PASSED:
  - Prefill PCC: 0.999325
  - First decode PCC: 0.999514
  - Sample per second: 0.547 (1 layer only)
- Full model (32 layers) FAILED with compiler error:
  - loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
  - 2026-05-18 23:47:08 ERR| Failed to run TTIRToTTNNCommon pipeline
  - Error triggered after 8 torch.compile recompilations when the paged KV cache becomes "full"
  - RuntimeError: Error code: 13 at tokenizer.batch_decode()
- lm_head.weight is MISSING from the checkpoint (randomly initialized) — this is expected for an embedding model

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gritlm_emb_m7_nodes16_fast_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (full model failed)

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
- total_flops:             455065206912
- breakdown.matmul:        455065206912
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  7241732292
- effective_count:        7110660292
- memory_bytes:           7817470732
- memory_gb:              7.28
- effective_memory_bytes: 7555326732
- effective_memory_gb:    7.04
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.8551
- top_perf_time_ms:         22.2940
- dram_time_ms:             14.8627
- compute_time_ms_lofi:     0.5171
- compute_time_ms_hifi2:    1.0342
- compute_time_ms_hifi3:    1.5514
- compute_time_ms_hifi4:    2.0685

## Files changed
- tests/benchmark/test_llms.py (added test_gritlm_emb_m7_nodes16_fast)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with getattr)
- .github/workflows/perf-bench-matrix.json (added gritlm_emb_m7_nodes16_fast entry)

## tt-forge-models submodule
no change
