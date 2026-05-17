loader_path: third_party.tt_forge_models.bloomz.causal_lm.pytorch.loader
variant_id: 1B1
arch: n150
status: DONE_FAIL
test_function: test_bloomz_1b1
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
failure_reason: "BLOOM ALiBi attention incompatible with benchmark harness StaticCache/cache_position API: RuntimeError in CPU reference forward pass - tensor shape mismatch in alibi.baddbmm() (target sizes [512,15,128] vs actual [512,1,15])"

# Benchmark added: test_bloomz_1b1

## Test
tests/benchmark/test_llms.py::test_bloomz_1b1

## Model
- HF name:    bigscience/bloomz-1b1
- Loader:     third_party.tt_forge_models.bloomz.causal_lm.pytorch.loader
- Variant:    ModelVariant.BLOOMZ_1B1 ("1B1")

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
- Hardware:           n150 (wormhole, n300 board single-chip)

## Failure details

The test failed during the CPU reference forward pass before any TT device execution.
The BLOOM model uses ALiBi (Attention with Linear Biases) positional encoding. When
the benchmark harness calls the model with `cache_position` (to enable the StaticCache
path), BLOOM's ALiBi attention computes biases for only the non-padded tokens (15 tokens
for the sample prompt), while the attention score matrix spans the full static cache
length (128). This causes a shape mismatch in `alibi.baddbmm()`:

  RuntimeError: The expanded size of the tensor (128) must match the existing size (15)
  at non-singleton dimension 2. Target sizes: [512, 15, 128]. Tensor sizes: [512, 1, 15]

Stack: transformers/models/bloom/modeling_bloom.py:268 → alibi.baddbmm(attention_scores)

This is incompatible with the benchmark harness's StaticCache/cache_position API.
Fixing it would require either a BLOOM-specific code path in the harness or changes
to the BLOOM model loader — both are out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
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
- tests/benchmark/test_llms.py (test_bloomz_1b1 added)

## tt-forge-models submodule
no change
