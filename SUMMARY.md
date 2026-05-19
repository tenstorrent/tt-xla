loader_path: third_party.tt_forge_models.biomedlm.causal_lm.pytorch.loader
variant_id: BioMedLM
arch: p150
status: DONE_PASS
test_function: test_biomedlm
samples_per_second: 5.257329825047927
ttft_ms: 847.993866
prefill_pcc: 0.944720
first_decode_pcc: 0.998990
top_perf_samples_per_sec: 101.4462
pct_of_target: 5.2
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_biomedlm

## Test
tests/benchmark/test_llms.py::test_biomedlm

## Model
- HF name:    stanford-crfm/BioMedLM
- Loader:     third_party.tt_forge_models.biomedlm.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIOMEDLM

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes L1 OOM (out of SRAM) on p150 for this model.
optimization_level=1 (all in DRAM) is stable and passes PCC.

## Measured (full model, defaults)
- Sample per second:  5.257329825047927
- TTFT (ms):          847.993866
- Prefill PCC:        0.944720
- First decode PCC:   0.998990
- Wall clock:         0:22:03
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_biomedlm_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 5.2% (5.26 / 101.45 — gap explained by optimization_level=1, which keeps all tensors in DRAM)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             167161364480
- breakdown.matmul:        6076497920
- breakdown.linear:        161084866560
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        671088640
- memory_bytes: 1342177280
- memory_gb:    1.25

### Params
- count:                  2668221575
- effective_count:        2591626375
- memory_bytes:           2909271576
- memory_gb:              2.7094702944159508
- effective_memory_bytes: 2756081176
- effective_memory_gb:    2.5668006166815758
- embedding_count:        76595200
- embedding_memory_bytes: 153190400

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 101.4462
- top_perf_time_ms:         9.8574
- dram_time_ms:             6.5716
- compute_time_ms_lofi:     0.1607
- compute_time_ms_hifi2:    0.3215
- compute_time_ms_hifi3:    0.4822
- compute_time_ms_hifi4:    0.6429

## Files changed
- tests/benchmark/test_llms.py (added test_biomedlm)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path hasattr guard)
- .github/workflows/perf-bench-matrix.json (added biomedlm entry)

## tt-forge-models submodule
no change
