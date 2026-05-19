loader_path: third_party.tt_forge_models.ken3_5_9b_i1_gguf.causal_lm.pytorch.loader
variant_id: 3_5_9B_I1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_ken3_5_9b_i1_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "GGUF architecture qwen35 not supported by installed transformers: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_ken3_5_9b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_ken3_5_9b_i1_gguf

## Model
- HF name:    mradermacher/Ken3.5-9B-i1-GGUF
- Loader:     third_party.tt_forge_models.ken3_5_9b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.KEN3_5_9B_I1_GGUF (3_5_9B_I1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The model loader attempts to load the Ken3.5-9B GGUF file (Ken3.5-9B.i1-Q4_K_M.gguf) using
AutoTokenizer.from_pretrained with gguf_file kwarg, which in turn calls
transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint. The GGUF file declares
architecture `qwen35`, which is not supported by the installed version of transformers:

  ValueError: GGUF model with architecture qwen35 is not supported yet.

This is a transformers version compatibility issue — the loader cannot be fixed from the
test side. The fix requires either upgrading transformers to a version that supports the
qwen35 GGUF architecture, or updating the loader in tt-forge-models to handle this model
differently (e.g., loading without GGUF). This is out of scope for the benchmark test skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (added test_ken3_5_9b_i1_gguf)
- SUMMARY.md

## tt-forge-models submodule
no change
