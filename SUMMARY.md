loader_path: third_party.tt_forge_models.nemotron_nano_9b_v2_heretic_i1_gguf.causal_lm.pytorch.loader
variant_id: 9B_v2_heretic_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_nemotron_nano_9b_v2_heretic_i1_gguf
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
failure_reason: "compiler graph partitioning failure during warmup forward pass: assert last_output_node is not None in torch.fx.passes.utils.fuser_utils; NemotronH Mamba hybrid model requires NemotronHHybridDynamicCache but benchmark harness uses StaticCache, causing incompatible model outputs and partition error in partition_fx_graph_for_cpu_fallback"

# Benchmark added: test_nemotron_nano_9b_v2_heretic_i1_gguf

## Test
tests/benchmark/test_llms.py::test_nemotron_nano_9b_v2_heretic_i1_gguf

## Model
- HF name:    RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic
- Loader:     third_party.tt_forge_models.nemotron_nano_9b_v2_heretic_i1_gguf.causal_lm.pytorch.loader
- Variant:    NEMOTRON_NANO_9B_V2_HERETIC_I1_GGUF

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
The test failed during compilation of the warmup forward pass with:

    AssertionError: assert last_output_node is not None
    in torch.fx.passes.utils.fuser_utils.insert_subgm
    (called from partition_fx_graph_for_cpu_fallback -> partitioner.fuse_partitions)

Root cause: The NemotronH architecture is a hybrid Mamba+attention model that requires
`NemotronHHybridDynamicCache` to be pre-initialized and passed as `past_key_values`.
The benchmark harness uses `StaticCache` (standard transformers KV cache), which is
incompatible with this architecture. The model warns:
    "NemotronH requires an initialized NemotronHHybridDynamicCache to return a cache.
     None was provided, so no cache will be returned."
Without the proper cache, the model produces graph outputs that cause the TT compiler's
graph partitioner to fail with the assertion.

This is a compiler/model-architecture incompatibility out of scope for this skill.
The fix would require extending the benchmark harness to support
NemotronHHybridDynamicCache initialization, or updating the model loader.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: p150
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py (added test_nemotron_nano_9b_v2_heretic_i1_gguf)

## tt-forge-models submodule
no change
