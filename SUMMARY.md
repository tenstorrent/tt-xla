loader_path: third_party.tt_forge_models.falcon_mamba.causal_lm.pytorch.loader
variant_id: Falcon_Mamba_7B
arch: n150
status: DONE_FAIL
test_function: test_falcon_mamba_7b
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
failure_reason: "Mamba SSM architecture incompatible with benchmark harness: FalconMambaConfig has no num_attention_heads; same root cause as test_mamba_2_8b"

# Benchmark added: falcon_mamba_7b

## Test
tests/benchmark/test_llms.py::test_falcon_mamba_7b

## Model
- HF name:    tiiuae/falcon-mamba-7b
- Loader:     third_party.tt_forge_models.falcon_mamba.causal_lm.pytorch.loader
- Variant:    ModelVariant.FALCON_MAMBA_7B

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

## Failure Details

The Falcon-Mamba-7B model uses a Mamba SSM (State Space Model) architecture
rather than a transformer with attention heads. The benchmark harness in
`tests/benchmark/benchmarks/llm_benchmark.py` assumes all models use
a KV cache (StaticCache) and requires `config.num_attention_heads` to
initialize it. FalconMambaConfig has no `num_attention_heads` attribute,
causing:

    AttributeError: 'FalconMambaConfig' object has no attribute 'num_attention_heads'
    (tests/benchmark/llm_utils/decode_utils.py:125, init_static_cache)

Additionally, Falcon-Mamba uses `cache_params` (not `past_key_values`) in its
forward() signature. Supporting Mamba-style models would require a substantial
refactor of the benchmark infrastructure (llm_benchmark.py is 864 lines with
`past_key_values` used at 10+ callsites).

This is the same root cause as the existing commented-out `test_mamba_2_8b`
failure. The fix belongs in the benchmark infrastructure, not in this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py (added test_falcon_mamba_7b with FAILED comment)

## tt-forge-models submodule
no change
