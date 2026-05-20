loader_path: third_party.tt_forge_models.mungert_mirothinker_1_7_mini_gguf.causal_lm.pytorch.loader
variant_id: MiroThinker_1_7_mini_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mungert_mirothinker_1_7_mini_gguf
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
failure_reason: "KeyError: 'blk.1.ffn_down_exps' in transformers/modeling_gguf_pytorch_utils.py - transformers==5.2.0 incompatible with MiroThinker GGUF MoE tensor key mapping"

# Benchmark added: test_mungert_mirothinker_1_7_mini_gguf

## Test
tests/benchmark/test_llms.py::test_mungert_mirothinker_1_7_mini_gguf

## Model
- HF name:    Mungert/MiroThinker-1.7-mini-GGUF
- Loader:     third_party.tt_forge_models.mungert_mirothinker_1_7_mini_gguf.causal_lm.pytorch.loader
- Variant:    MiroThinker_1_7_mini_Q4_K_M_GGUF

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

The test failed at model loading time with:

```
KeyError: 'blk.1.ffn_down_exps'
```

in `venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:143`
inside `_set_moe_expert_tensor` → `tensor_key_mapping[m["name"]]`.

The MiroThinker-1.7-mini GGUF model uses MoE (Mixture of Experts) tensor
keys (e.g. `blk.1.ffn_down_exps`) that are not present in the
`tensor_key_mapping` of `transformers==5.2.0`. This is a library-level
incompatibility — the GGUF loader in transformers 5.2.0 does not know how
to map these MoE expert tensor names to PyTorch model weights. The fix
would require either a newer transformers version that supports this GGUF
MoE variant, or an updated key mapping in the GGUF loading utilities.

This error originates entirely in the transformers library and cannot be
resolved by changes to the test harness or benchmarking infrastructure.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
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
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
