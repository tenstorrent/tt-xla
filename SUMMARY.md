loader_path: third_party.tt_forge_models.bartowski_nvidia_openreasoning_nemotron_7b_gguf.causal_lm.pytorch.loader
variant_id: OpenReasoning_Nemotron_7B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_bartowski_nvidia_openreasoning_nemotron_7b_gguf
samples_per_second: 3.9996167666113847
ttft_ms: 1242.46093
prefill_pcc: 0.944564
first_decode_pcc: 0.895475
top_perf_samples_per_sec: 46.0472
pct_of_target: 8.7
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "first decode PCC=0.895 below required 0.94 at optimization_level=0 (best achievable); higher optimization levels fail prefill PCC due to GGUF Q4_K_M quantization accumulation across 32 layers (best prefill at opt_level=2: 0.906)"

# Benchmark added: test_bartowski_nvidia_openreasoning_nemotron_7b_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_nvidia_openreasoning_nemotron_7b_gguf

## Model
- HF name:    bartowski/nvidia_OpenReasoning-Nemotron-7B-GGUF
- Loader:     third_party.tt_forge_models.bartowski_nvidia_openreasoning_nemotron_7b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BARTOWSKI_NVIDIA_OPENREASONING_NEMOTRON_7B_GGUF

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## PCC tuning summary (full model, 32 layers)
All configurations tested — none achieve both prefill and decode PCC >= 0.94:

| optimization_level | bfp_bf8 | fp32_dest_acc | Prefill PCC | Decode PCC |
|--------------------|---------|---------------|-------------|------------|
| 2                  | yes     | no            | 0.879 FAIL  | (not reached) |
| 1                  | yes     | no            | 0.845 FAIL  | (not reached) |
| 2                  | no      | no            | 0.906 FAIL  | (not reached) |
| 2                  | no      | yes           | 0.906 FAIL  | (not reached) |
| 1                  | no      | no            | 0.836 FAIL  | (not reached) |
| 1                  | no      | yes           | 0.836 FAIL  | (not reached) |
| 0                  | no      | yes           | 0.944 PASS  | 0.895 FAIL |

Best config is optimization_level=0 with fp32_dest_acc_en=True and no bfp_bf8,
which passes prefill but fails first decode PCC. The issue is accumulated GGUF
Q4_K_M quantization error across 32 transformer layers.

## Measured (full model, optimization_level=0)
- Sample per second:  3.9996167666113847
- TTFT (ms):          1242.46093
- Prefill PCC:        0.944564 (PASS)
- First decode PCC:   0.895475 (FAIL)
- Wall clock:         0:04:11
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_nvidia_openreasoning_nemotron_7b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 8.7% (3.999 / 46.05) — low because optimization_level=0

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             452502421632
- breakdown.matmul:        422903283840
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615616707
- effective_count:        7070619331
- memory_bytes:           15231233800
- memory_gb:              14.185191877186298
- effective_memory_bytes: 14141239048
- effective_memory_gb:    13.170055158436298
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5142
- compute_time_ms_hifi2:    1.0284
- compute_time_ms_hifi3:    1.5426
- compute_time_ms_hifi4:    2.0568

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
