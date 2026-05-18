loader_path: third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
variant_id: 1B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_albert_wesker_1b_gguf
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
failure_reason: "KeyError: 'sliding_attention' in Gemma3 forward pass - model/transformers 5.2.0 version incompatibility; cannot fix without editing loader or transformers"

# Benchmark added: test_albert_wesker_1b_gguf

## Test
tests/benchmark/test_llms.py::test_albert_wesker_1b_gguf

## Model
- HF name:    mradermacher/Albert_Wesker-1B-GGUF
- Loader:     third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALBERT_WESKER_1B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails at the CPU golden run (before any TT device compilation) with:

    KeyError: 'sliding_attention'

The traceback shows:
    transformers/models/gemma3/modeling_gemma3.py:589:
        position_embeddings=position_embeddings[decoder_layer.attention_type]

Albert Wesker 1B GGUF is a Gemma3-based model. The Gemma3 forward pass in
transformers==5.2.0 attempts to index `position_embeddings` with
`decoder_layer.attention_type`, which equals `'sliding_attention'`, but that
key is missing from the `position_embeddings` dict. This is a
model/transformers version incompatibility that cannot be fixed in the test
or benchmarking infrastructure. The fix belongs in the loader or in
upgrading/pinning the transformers version in tt-forge-models.

Note: same failure was observed on n150 (branch
odjuricic/ai-benchmark-pipeline-test_albert_wesker_1b_gguf, submodule ~93218a34).
On p150, additional missing dependency `gguf>=0.10.0` was encountered and
resolved by installing gguf==0.19.0; the sliding_attention error then appeared.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model did not compile)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change (submodule at 986e808f12)
