loader_path: third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader
variant_id: Simple_V2_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf
samples_per_second: 4.19515704026459
ttft_ms: 1241.081805
prefill_pcc: 0.992996
first_decode_pcc: 0.997859
top_perf_samples_per_sec: 46.04714291796427
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 2
trace_enabled: false
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf

## Test
tests/benchmark/test_llms.py::test_document_validation_qwen2_5_vl_simple_v2_i1_gguf

## Model
- HF name:    hienphantt161/Document-Validation-Qwen2.5-VL-Simple-V2
- Loader:     third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader
- Variant:    Simple_V2_i1_GGUF

Note: This model loads the full Qwen2_5_VLForConditionalGeneration (multimodal vision-language model,
~7.6B params) and extracts the language model component (Qwen2ForCausalLM). The loader
does not properly restrict layers when num_layers is passed, so all 28 transformer layers
are used regardless of the num_layers parameter.

## Test config landed
- optimization_level:        2
- trace_enabled:             false
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.19515704026459
- TTFT (ms):          1241.081805
- Prefill PCC:        0.992996
- First decode PCC:   0.997859
- Wall clock:         0:05:27
- Hardware:           p150 (blackhole)

Note: low samples/sec (9.1% of roofline) is expected with trace_enabled=False.
Trace was disabled because a previous investigation found it hung with trace_enabled=True.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_document_validation_qwen2_5_vl_simple_v2_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 9.1%

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
- total_flops:             454146600960
- breakdown.matmul:        424547463168
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615622790
- effective_count:        7070625414
- memory_bytes:           8602865172
- memory_gb:              8.012042541056871
- effective_memory_bytes: 7512870420
- effective_memory_gb:    6.996905822306871
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.04714291796427
- top_perf_time_ms:         21.716873982421873
- dram_time_ms:             14.47791598828125
- compute_time_ms_lofi:     0.5160756829090909
- compute_time_ms_hifi2:    1.0321513658181818
- compute_time_ms_hifi3:    1.5482270487272745
- compute_time_ms_hifi4:    2.0643027316363636

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
