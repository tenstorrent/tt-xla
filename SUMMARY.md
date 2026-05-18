loader_path: third_party.tt_forge_models.granite_4_0_h_350m_base.causal_lm.pytorch.loader
variant_id: 4.0_H_350M_base
arch: p150
status: DONE_FAIL
test_function: test_granite_4_0_h_350m_base
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
failure_reason: "Hybrid Mamba-Transformer model (GraniteMoeHybrid) requires HybridMambaAttentionDynamicCache; incompatible with the StaticCache-based LLM benchmark harness. AttributeError: 'StaticCache' object has no attribute 'has_previous_state' in torch_forward of mamba layer. Same class of failure as test_mamba_2_8b (MambaConfig lacks num_attention_heads)."

# Benchmark added: test_granite_4_0_h_350m_base

## Test
tests/benchmark/test_llms.py::test_granite_4_0_h_350m_base

## Model
- HF name:    ibm-granite/granite-4.0-h-350m-base
- Loader:     third_party.tt_forge_models.granite_4_0_h_350m_base.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_4_0_H_350M_BASE ("4.0_H_350M_base")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before hardware execution)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure details
The model `ibm-granite/granite-4.0-h-350m-base` is a Hybrid Mamba-Transformer
architecture (`GraniteMoeHybridForCausalLM`). Its Mamba layers require a
`HybridMambaAttentionDynamicCache` (with `has_previous_state`, `conv_states`,
`ssm_states` fields) rather than the standard `StaticCache` that the LLM
benchmark harness creates.

The failure occurs on the CPU golden path before any XLA compilation:
```
transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:646: in torch_forward
    and cache_params.has_previous_state
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

This is the same class of incompatibility as the existing `test_mamba_2_8b`
failure (`# FAILED: AttributeError: 'MambaConfig' object has no attribute
'num_attention_heads'`). Supporting Hybrid Mamba models requires a new
static-shape Mamba cache abstraction and changes to the `generate_and_benchmark`
loop — engineering work that belongs in the benchmark infrastructure, not
in the test function itself.

## Decode roofline (first decode graph, single-chip)
N/A — test failed before any compilation or execution

## Files changed
- tests/benchmark/test_llms.py (test_granite_4_0_h_350m_base added, but left in place as DONE_FAIL)

## tt-forge-models submodule
no change
