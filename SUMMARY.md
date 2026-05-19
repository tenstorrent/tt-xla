loader_path: third_party.tt_forge_models.dolphin_2_1_mistral_7b_gguf.causal_lm.pytorch.loader
variant_id: 2.1_Mistral_7B_GGUF
arch: p150
status: DONE_PASS
test_function: test_dolphin_2_1_mistral_7b_gguf
samples_per_second: 34.9242358865313
ttft_ms: 292.067543
prefill_pcc: 0.999606
first_decode_pcc: 0.979778
top_perf_samples_per_sec: 44.8550
pct_of_target: 77.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Dolphin 2.1 Mistral 7B GGUF — Benchmark Results (p150)

## Model
- **Loader**: `third_party.tt_forge_models.dolphin_2_1_mistral_7b_gguf.causal_lm.pytorch.loader`
- **Variant**: `2.1_Mistral_7B_GGUF`
- **HF model**: `TheBloke/dolphin-2.1-mistral-7B-GGUF` (Q4_K_M, 4.37 GB)
- **Architecture**: Mistral 7B (~7.24B parameters)
- **Hardware**: Blackhole (p150), single chip

## Test Function
```python
tests/benchmark/test_llms.py::test_dolphin_2_1_mistral_7b_gguf
```

## Results (Full Model)

| Metric | Value |
|--------|-------|
| Samples/second | 34.92 |
| TTFT (ms) | 292.07 |
| Prefill PCC | 0.9996 ✓ |
| First Decode PCC | 0.9798 ✓ |
| Roofline (top_perf) | 44.86 samples/sec |
| % of roofline | 77.9% |
| Roofline bound | DRAM |

## Configuration

| Setting | Value |
|---------|-------|
| optimization_level | 2 |
| trace_enabled | true |
| experimental_weight_dtype | bfp_bf8 |
| batch_size | 32 |
| input_sequence_length | 128 |

## Run Summary

- **Bringup** (num_layers=1, max_output_tokens=3): PASSED in 66s
  - Prefill PCC: 0.9993, First decode PCC: 0.9998
  - Throughput: 477.6 samples/sec (1-layer ceiling)
- **Full model** (all layers, max_output_tokens=3): PASSED in 524s (8:44)
  - Prefill PCC: 0.9996, First decode PCC: 0.9798
  - Throughput: 34.92 samples/sec

## Notes

- Model uses GGUF Q4_K_M quantization (4-bit). PCC thresholds pass comfortably at 0.94+.
- 77.9% of DRAM roofline — good utilization for a GGUF quantized model.
- The `experimental_weight_dtype=bfp_bf8` is inherited from `DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE` in conftest.py.
- `optimization_level=2` is the default and passed all PCC checks.
- Trace is enabled (default) and works correctly.
