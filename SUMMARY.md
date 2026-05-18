loader_path: third_party.tt_forge_models.insubordinated_plague_parasite_1b_i1_gguf.causal_lm.pytorch.loader
variant_id: INSUBORDINATED_PLAGUE_PARASITE_1B_I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_insubordinated_plague_parasite_1b_i1_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers==5.2.0 Gemma3 model forward pass (CPU golden reference); position_embeddings dict missing sliding_attention key in modeling_gemma3.py:589"

# Benchmark added: test_insubordinated_plague_parasite_1b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_insubordinated_plague_parasite_1b_i1_gguf

## Model
- HF name:    mradermacher/Insubordinated.Plague-Parasite-1B-i1-GGUF
- Loader:     third_party.tt_forge_models.insubordinated_plague_parasite_1b_i1_gguf.causal_lm.pytorch.loader
- Variant:    INSUBORDINATED_PLAGUE_PARASITE_1B_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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
The test failed during CPU golden reference computation with:
  KeyError: 'sliding_attention'
  in transformers==5.2.0 at transformers/models/gemma3/modeling_gemma3.py:589

  The Gemma3 model's forward pass constructs a position_embeddings dict but
  does not include the 'sliding_attention' key, which is required by the
  sliding-attention decoder layers in this 1B Gemma3 variant. This is a
  transformers library compatibility issue with this model's architecture.
  The failure is on CPU (before any TT hardware is involved), so no
  compiler or device changes can address it.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
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
- tests/benchmark/test_llms.py (added test_insubordinated_plague_parasite_1b_i1_gguf)

## tt-forge-models submodule
no change
