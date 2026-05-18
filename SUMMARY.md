loader_path: third_party.tt_forge_models.geppetto.causal_lm.pytorch.loader
variant_id: geppetto
arch: p150
status: DONE_PASS
test_function: test_geppetto
samples_per_second: 169.35
ttft_ms: 82.51
prefill_pcc: 0.999483
first_decode_pcc: 0.999621
top_perf_samples_per_sec: 1802.84
pct_of_target: 9.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_geppetto

## Test
tests/benchmark/test_llms.py::test_geppetto

## Model
- HF name:    LorenzoDeMattei/GePpeTto
- Loader:     third_party.tt_forge_models.geppetto.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEPPETTO

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  169.35
- TTFT (ms):          82.51
- Prefill PCC:        0.999483
- First decode PCC:   0.999621
- Wall clock:         0:02:02
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_geppetto_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.4% (169.35 / 1802.84)

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
- total_flops:             158999740416
- breakdown.matmul:        33914880000
- breakdown.linear:        125084860416
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        759
- memory_bytes: 3036

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  131922564
- effective_count:        108096132
- memory_bytes:           162785036
- memory_gb:              0.15160537883639336
- effective_memory_bytes: 115132172
- effective_memory_gb:    0.1072251908481121
- embedding_count:        23826432
- embedding_memory_bytes: 47652864

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1802.84
- top_perf_time_ms:         0.5547
- dram_time_ms:             0.3698
- compute_time_ms_lofi:     0.1807
- compute_time_ms_hifi2:    0.3614
- compute_time_ms_hifi3:    0.5420
- compute_time_ms_hifi4:    0.7227

## Files changed
- tests/benchmark/test_llms.py (added test_geppetto)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
