loader_path: third_party.tt_forge_models.mixtao.causal_lm.pytorch.loader
variant_id: 7Bx2_MoE_v8.1
arch: p150
status: DONE_FAIL
test_function: test_mixtao_7bx2_moe_v8_1
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
failure_reason: "transformers 5.2.0 compatibility: KeyError: '`dense` is not a valid experts implementation registered in the `ExpertsInterface`' — the MoE forward pass requests the dense expert impl which is not registered in this transformers version; loader sets _experts_implementation='eager' but dense is still dispatched at runtime; fix belongs in the loader"

# Benchmark added: test_mixtao_7bx2_moe_v8_1

## Test
tests/benchmark/test_llms.py::test_mixtao_7bx2_moe_v8_1

## Model
- HF name:    mixtao/MixTAO-7Bx2-MoE-v8.1
- Loader:     third_party.tt_forge_models.mixtao.causal_lm.pytorch.loader
- Variant:    ModelVariant.MIXTAO_7BX2_MOE_V8_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before benchmark)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

The test failed during the CPU golden run (before any TT device execution) with:

```
KeyError: '`dense` is not a valid experts implementation registered in the `ExpertsInterface`'
```

Traceback origin: `transformers/integrations/moe.py:317` inside
`ExpertsInterface.get_interface`. The loader (`mixtao/causal_lm/pytorch/loader.py`)
sets `config._experts_implementation = "eager"` and passes
`experts_implementation="eager"` to `from_pretrained`, but during the forward
pass the MoE layer dispatches to the `dense` implementation which is not
registered in transformers 5.2.0's `ExpertsInterface`. This is a compatibility
issue between the loader and the installed transformers version.

Fix required in the loader (tt-forge-models repo), not in tt-xla.

## Decode roofline (first decode graph, single-chip)
N/A — test failed before device execution

## Files changed
- tests/benchmark/test_llms.py (test function added, will be uncommitted on DONE_FAIL)

## tt-forge-models submodule
no change — submodule at 93218a34fc9fc6a671e0e41101da470c80891b2a (2026-05-14 uplift)
