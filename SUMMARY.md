loader_path: third_party.tt_forge_models.bagel_dpo_7b_v0_1_gguf.causal_lm.pytorch.loader
variant_id: Bagel_DPO_7B_v0_1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_bagel_dpo_7b_v0_1_gguf
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
failure_reason: "loader bug: causal_lm/__init__.py imports from .loader which does not exist at submodule HEAD 32ca95eb23; correct loader is at causal_lm/pytorch/loader.py; ModuleNotFoundError: No module named third_party.tt_forge_models.bagel_dpo_7b_v0_1_gguf.causal_lm.loader"

# Benchmark added: bagel_dpo_7b_v0_1_gguf

## Test
tests/benchmark/test_llms.py::test_bagel_dpo_7b_v0_1_gguf

## Model
- HF name:    RichardErkhov/jondurbin_-_bagel-dpo-7b-v0.1-gguf
- Loader:     third_party.tt_forge_models.bagel_dpo_7b_v0_1_gguf.causal_lm.pytorch.loader
- Variant:    Bagel_DPO_7B_v0_1_Q4_K_M_GGUF

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

## Failure Analysis

The test fails at import time due to a broken `causal_lm/__init__.py` in the
`bagel_dpo_7b_v0_1_gguf` loader at submodule HEAD `32ca95eb23`:

```
third_party/tt_forge_models/bagel_dpo_7b_v0_1_gguf/causal_lm/__init__.py:5: in <module>
    from .loader import ModelLoader, ModelVariant
ModuleNotFoundError: No module named 'third_party.tt_forge_models.bagel_dpo_7b_v0_1_gguf.causal_lm.loader'
```

The `__init__.py` tries `from .loader import ModelLoader, ModelVariant`, but no
`loader.py` exists directly in `causal_lm/`. The actual loader is at
`causal_lm/pytorch/loader.py`. Other GGUF loaders in the submodule have no
`__init__.py` in their `causal_lm/` directory at all.

Fix required in `tt-forge-models` repo: either remove the broken
`causal_lm/__init__.py` or change the import to `from .pytorch.loader import ...`.
Editing submodule files is out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not run)
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
- tests/benchmark/test_llms.py (test_bagel_dpo_7b_v0_1_gguf added)

## tt-forge-models submodule
no change — submodule at 32ca95eb23 has a broken causal_lm/__init__.py that blocks import
