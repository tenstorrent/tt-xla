loader_path: third_party.tt_forge_models.deeppresenter_9b_gguf.causal_lm.pytorch.loader
variant_id: 9B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_deeppresenter_9b_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported by transformers==5.2.0"

# Benchmark added: test_deeppresenter_9b_gguf

## Test
tests/benchmark/test_llms.py::test_deeppresenter_9b_gguf

## Model
- HF name:    mradermacher/DeepPresenter-9B-GGUF
- Loader:     third_party.tt_forge_models.deeppresenter_9b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPPRESENTER_9B_GGUF ("9B_GGUF")

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
- Hardware:           n150

## Failure details
The test fails at model load time with:

  ValueError: GGUF model with architecture qwen35 is not supported yet.

The model `mradermacher/DeepPresenter-9B-GGUF` uses the `qwen35` GGUF
architecture, which is not supported by `transformers==5.2.0`
(`transformers/modeling_gguf_pytorch_utils.py:478`). This is a loader-level
dependency compatibility issue; the fix belongs in the tt-forge-models repo
(either upgrading the transformers requirement for this loader or waiting for
transformers to add qwen35 GGUF support).

Traceback (abbreviated):
  third_party/tt_forge_models/deeppresenter_9b_gguf/causal_lm/pytorch/loader.py:70
    self.tokenizer = AutoTokenizer.from_pretrained(...)
  venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:478
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not run)
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
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
