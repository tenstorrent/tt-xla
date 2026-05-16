loader_path: third_party.tt_forge_models.adasearch_qwen2_5_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: 7B_Instruct_GGUF
arch: n150
status: DONE_FAIL
test_function: test_adasearch_qwen2_5_7b_instruct_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "insufficient disk space to download GGUF model: ~4.7GB required, disk at 100% capacity (0 bytes available for non-root on /dev/nvme0n1p3)"

# Benchmark added: test_adasearch_qwen2_5_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_adasearch_qwen2_5_7b_instruct_gguf

## Model
- HF name:    mradermacher/AdaSearch-Qwen2.5-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.adasearch_qwen2_5_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ADASEARCH_QWEN2_5_7B_INSTRUCT_GGUF (= "7B_Instruct_GGUF")

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
- Hardware:           n150 (Wormhole n300 L)

## Failure reason
The GGUF model file (AdaSearch-Qwen2.5-7B-Instruct.Q4_K_M.gguf, ~4.7 GB) could not be
downloaded because the disk partition /dev/nvme0n1p3 (183GB) had 0 bytes available for
non-root users at test runtime. The HuggingFace hub aborted with:

  RuntimeError: Internal error: Internal Writer Error: Background writer channel closed
  UserWarning: Not enough free disk space to download the file. The expected file size
  is: 4683.07 MB. The target location only has 728.64 MB free disk space.

The test function and loader are valid; the variant `7B_Instruct_GGUF` exists in the
current submodule HEAD. The test will pass once sufficient disk space is available
(~5GB free needed for the GGUF model download).

Note: The 70B model cache at /home/ogi/.cache/huggingface/hub/models--mradermacher--70B_Incisive_Vernacular-i1-GGUF/ is consuming 26GB and is not referenced by any benchmark test — clearing it would free enough space to run this test.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model did not run)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150
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
- tests/benchmark/test_llms.py (added test_adasearch_qwen2_5_7b_instruct_gguf)

## tt-forge-models submodule
no change
