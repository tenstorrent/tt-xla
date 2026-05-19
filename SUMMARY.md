loader_path: third_party.tt_forge_models.ettin_decoder.causal_lm.pytorch.loader
variant_id: 17M
arch: p150
status: DONE_FAIL
test_function: test_ettin_decoder_17m
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 17813.3057
pct_of_target: null
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "KeyError: 'sliding_attention' in CPU prefill — transformers modernbert_decoder.py position_embeddings dict missing 'sliding_attention' key for layers 1-6; model uses mixed global/sliding attention but RoPE dict is not initialized for all attention types; passes with --num-layers 1 (global-only layer)"

# Benchmark added: test_ettin_decoder_17m

## Test
tests/benchmark/test_llms.py::test_ettin_decoder_17m

## Model
- HF name:    jhu-clsp/ettin-decoder-17m
- Loader:     third_party.tt_forge_models.ettin_decoder.causal_lm.pytorch.loader
- Variant:    ModelVariant.ETTIN_DECODER_17M (17M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (full model failed before device execution)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

### Failure details
Full model (7 layers) fails in CPU prefill with:
```
KeyError: 'sliding_attention'
  transformers/models/modernbert_decoder/modeling_modernbert_decoder.py:533:
    position_embeddings=position_embeddings[decoder_layer.attention_type],
```
The ettin-decoder-17m model uses mixed attention: layer 0 is `global_attention`,
layers 1-6 are `sliding_attention`. The position_embeddings dict is not initialized
with a `'sliding_attention'` key, causing a KeyError when the full model runs.
With --num-layers 1 (global-only) the test passes cleanly:
  Prefill PCC: 0.999395, First decode PCC: 0.999616, ~564 samples/sec.
This is a model/transformers compatibility bug; not fixable in the benchmark harness.

## Decode roofline (first decode graph, single-chip, from --num-layers 1 run)
Source JSON: tt_xla_ettin_decoder_17m_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: N/A (full model not runnable)

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
- total_flops:             16467091456
- breakdown.matmul:        757110784
- breakdown.linear:        15709980672
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        627
- memory_bytes: 2508

### KV cache
- count:        2097152
- memory_bytes: 4194304
- memory_gb:    0.00390625

### Params
- count:                  26463555
- effective_count:        13569347
- memory_bytes:           40257416
- memory_gb:              0.03749264031648636
- effective_memory_bytes: 14469000
- effective_memory_gb:    0.013475306332111359
- embedding_count:        12894208
- embedding_memory_bytes: 25788416

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 17813.3057
- top_perf_time_ms:         0.0561
- dram_time_ms:             0.0316
- compute_time_ms_lofi:     0.0187
- compute_time_ms_hifi2:    0.0374
- compute_time_ms_hifi3:    0.0561
- compute_time_ms_hifi4:    0.0749

## Files changed
- tests/benchmark/test_llms.py (added test_ettin_decoder_17m)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: use hasattr check before calling get_weight_dtype_config_path, matching runner behavior)

## tt-forge-models submodule
no change
