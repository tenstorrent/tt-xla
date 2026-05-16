loader_path: third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
variant_id: SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_029_shisa_gamma_7b
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
failure_reason: "3-chip P150 Blackhole cluster detected as CUSTOM type; TT_MESH_GRAPH_DESC_PATH not configured for this non-standard topology — device initialization fails (TT_FATAL at tt_cluster.cpp:277) for all tests on this machine"

# Benchmark added: test_029_shisa_gamma_7b

## Test
tests/benchmark/test_llms.py::test_029_shisa_gamma_7b

## Model
- HF name:    mradermacher/029-shisa-gamma-7b-v1-v2new-dpo405b-i1-GGUF
- Loader:     third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF (value: "029_SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device not reachable)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (3-chip Blackhole cluster, CUSTOM topology)

## Infrastructure failure — why the test could not run

The machine hosts three Blackhole P150 chips (PCI vendor 0x1e52, device 0xb140
at 0000:01:00.0, 0000:02:00.0, 0000:03:00.0).  Only one chip is exposed as
`/dev/tenstorrent/0` (MMIO); the other two are reachable via ethernet from the
MMIO chip.  tt-metal's UMD enumerates all three, making `num_chips == 3` in
`get_cluster_type_from_cluster_desc`.  Because the P150 branch only handles
`num_chips ∈ {1, 2, 4, 8}`, the cluster type falls through to `CUSTOM`.

The CUSTOM type unconditionally requires `TT_MESH_GRAPH_DESC_PATH` to be set
to a fabric mesh graph descriptor that encodes the physical cable topology
(see `tt_cluster.cpp:271-278`).  No such variable is set in the environment,
and no pre-shipped descriptor covers a 3-chip P150 topology.  The resulting
`TT_FATAL` fires before any JAX/XLA operation reaches the compiler, so no
test — new or existing — can run on this machine as currently configured.

**To unblock:** an operator needs to set `TT_MESH_GRAPH_DESC_PATH` to a
`.textproto` mesh descriptor that matches the physical ethernet cabling among
the three P150 chips.  Alternatively, disconnecting two chips would reduce the
visible count to 1, giving cluster type `P150` (no descriptor required).

Note: `TT_METAL_VISIBLE_DEVICES=0` was tried but does not help — the cluster
descriptor counts physically-connected chips before the visible-devices filter
is applied.

## Import workaround committed

The loader path `third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf`
starts with a digit, making a plain `from … import` a Python `SyntaxError`.
The test uses `importlib.import_module(…)` to work around this — the only
loader in the test suite that requires this pattern.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test never reached execution
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
- tests/benchmark/test_llms.py (new function test_029_shisa_gamma_7b with importlib import)
- .github/workflows/perf-bench-matrix.json (new entry: 029_shisa_gamma_7b)

## tt-forge-models submodule
no change
