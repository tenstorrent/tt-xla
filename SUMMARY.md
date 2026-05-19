loader_path: third_party.tt_forge_models.chemdfm.causal_lm.pytorch.loader
variant_id: v1.0_13B
arch: p150
status: DONE_FAIL
test_function: test_chemdfm_v1_0_13b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 549.3355
pct_of_target: null
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "MLIR compilation with optimization_level=2 took 5h44m+ without completing for 1-layer 13B model (suspected exponential SRAM optimization due to large LM head 32001x5120 at batch_size=32); process killed with SIGKILL left PCIe device in hung state (Read 0xffffffff over PCIe ID 3); tt-smi --reset and container sysfs reset both failed; requires host-level PCIe bus reset or power cycle. Fixed llm_benchmark.py elif hasattr guard bug (no longer crashes on get_weight_dtype_config_path); test function added and correct."

# Benchmark added: test_chemdfm_v1_0_13b

## Test
tests/benchmark/test_llms.py::test_chemdfm_v1_0_13b

## Model
- HF name:    OpenDFM/ChemDFM-v1.0-13B
- Loader:     third_party.tt_forge_models.chemdfm.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHEMDFM_V1_0_13B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (MLIR compilation timed out; device subsequently hung after kill -9)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         N/A
- Hardware:           p150 (Blackhole)

## Failure timeline
1. Run 1 (optimization_level=2, trace=True, num_layers=1): Failed with AttributeError 'get_weight_dtype_config_path' → fixed via elif hasattr guard in llm_benchmark.py
2. Run 2 (optimization_level=2, trace=True, num_layers=1): Device initialized successfully, MLIR compilation started, ran 5h44m at 102% CPU without completing. Killed with SIGKILL.
3. SIGKILL left PCIe device 3 in hung state (Read 0xffffffff); tt-smi --reset could not re-initialize; sysfs write blocked by container read-only mount.
4. Run 3 (optimization_level=0, num_layers=1): Immediately failed with PCIe hung error on device initialization.
Root cause: optimization_level=2 SRAM placement is pathologically slow for large vocabulary models (32001-token LM head at batch_size=32 creates enormous tensors); requires host power-cycle to clear device hang.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chemdfm_v1_0_13b_perf_metrics_0.json
Note: Captured from 1-layer preliminary run (not full 13B model)
Achieved vs top_perf_samples_per_sec: N/A (full model never ran)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             631065479552
- breakdown.matmul:        631065479552
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        627
- memory_bytes: 2508

### KV cache
- count:        41943040
- memory_bytes: 83886080
- memory_gb:    0.078125

### Params
- count:                  720757955
- effective_count:        518983875
- memory_bytes:           954983496
- memory_gb:              0.8893976882100105
- effective_memory_bytes: 551435336
- effective_memory_gb:    0.5135641768574715
- embedding_count:        201774080
- embedding_memory_bytes: 403548160

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 549.3355
- top_perf_time_ms:         1.8204
- dram_time_ms:             1.1298
- compute_time_ms_lofi:     0.6068
- compute_time_ms_hifi2:    1.2136
- compute_time_ms_hifi3:    1.8204
- compute_time_ms_hifi4:    2.4272

## Files changed
- tests/benchmark/test_llms.py (added test_chemdfm_v1_0_13b)
- tests/benchmark/benchmarks/llm_benchmark.py (added elif hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
