loader_path: third_party.tt_forge_models.deepseek_r1_medical_cot.causal_lm.pytorch.loader
variant_id: DeepSeek_R1_Medical_COT
arch: p150
status: DONE_FAIL
test_function: test_deepseek_r1_medical_cot
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
failure_reason: "loader fails with RuntimeError: weight size mismatch - rwibawa/DeepSeek-R1-Medical-COT is based on unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit (4-bit BNB NF4 quantized checkpoint); loader calls AutoModelForCausalLM.from_pretrained without quantization config so NF4 packed weight shapes [8388608,1] mismatch standard BF16 shapes [4096,4096]; fix belongs in tt-forge-models loader"

# Benchmark added: test_deepseek_r1_medical_cot

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_medical_cot

## Model
- HF name:    rwibawa/DeepSeek-R1-Medical-COT
- Loader:     third_party.tt_forge_models.deepseek_r1_medical_cot.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_MEDICAL_COT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed at model load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The model `rwibawa/DeepSeek-R1-Medical-COT` is a fine-tune of
`unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit`, a 4-bit BNB NF4
quantized checkpoint. The loader uses `AutoModelForCausalLM.from_pretrained`
without `load_in_4bit=True` or a `BitsAndBytesConfig`, causing a
`RuntimeError` when transformers detects weight shape mismatches between
the NF4-packed checkpoint (e.g. `[8388608, 1]`) and the standard BF16
model architecture (e.g. `[4096, 4096]`).

This is a loader-level bug in `third_party/tt_forge_models/` and cannot be
fixed from the test side. The fix must be applied in the tt-forge-models repo.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model failed to load)
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
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change (submodule HEAD: 9c78276012)
