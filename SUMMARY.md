loader_path: third_party.tt_forge_models.atom_olmo3_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_atom_olmo3_7b_i1_gguf
samples_per_second: 3.3802640461147115
ttft_ms: 1726.407937
prefill_pcc: 0.999143
first_decode_pcc: 0.997115
top_perf_samples_per_sec: 41.5763
pct_of_target: 8.1
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_atom_olmo3_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_atom_olmo3_7b_i1_gguf

## Model
- HF name:    mradermacher/atom-olmo3-7b-i1-GGUF
- Loader:     third_party.tt_forge_models.atom_olmo3_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ATOM_OLMO3_7B_Q4_K_M

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level is set to 0 (instead of default 2) because levels 1 and 2
trigger SDPA fusion which fails with a compiler type mismatch on OLMo2/OLMo3
QK-norm attention: 'ttnn.scaled_dot_product_attention' op Query and result must
have the same element type. This is a tt-mlir compiler bug.

## Measured (full model, defaults)
- Sample per second:  3.3802640461147115
- TTFT (ms):          1726.407937
- Prefill PCC:        0.999143
- First decode PCC:   0.997115
- Wall clock:         0:10:22
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_atom_olmo3_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.1% (3.38 / 41.58)

Note: Low utilization (8.1% of roofline) is expected because optimization_level=0
disables all tensor placement optimizations. Higher optimization levels fail with
a compiler bug (SDPA type mismatch on OLMo2/OLMo3 QK-norm attention).

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
- total_flops:             442899103872
- breakdown.matmul:        442899103872
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
- count:                  7298011334
- effective_count:        6887272646
- memory_bytes:           8139700500
- memory_gb:              7.580686826258898
- effective_memory_bytes: 7318223124
- effective_memory_gb:    6.81562640145421
- embedding_count:        410738688
- embedding_memory_bytes: 821477376

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.5763
- top_perf_time_ms:         24.0521
- dram_time_ms:             16.0348
- compute_time_ms_lofi:     0.4259
- compute_time_ms_hifi2:    0.8517
- compute_time_ms_hifi3:    1.2776
- compute_time_ms_hifi4:    1.7035

## Files changed
- tests/benchmark/test_llms.py (added test_atom_olmo3_7b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed optional get_weight_dtype_config_path)

## tt-forge-models submodule
no change
