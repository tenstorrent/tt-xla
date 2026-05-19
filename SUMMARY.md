loader_path: third_party.tt_forge_models.mistral_nemo_instruct_thinking_gguf.causal_lm.pytorch.loader
variant_id: 12B_Thinking_GGUF
arch: p150
status: DONE_PASS
test_function: test_mistral_nemo_instruct_thinking_gguf_12b
samples_per_second: 20.09
ttft_ms: 490.4
prefill_pcc: 0.983401
first_decode_pcc: 0.997257
top_perf_samples_per_sec: 27.7857
pct_of_target: 72.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark: test_mistral_nemo_instruct_thinking_gguf_12b (p150)

## Model

- **Loader**: `third_party.tt_forge_models.mistral_nemo_instruct_thinking_gguf.causal_lm.pytorch.loader`
- **Variant**: `12B_Thinking_GGUF` (`MISTRAL_NEMO_INSTRUCT_THINKING_GGUF`)
- **HF model**: `mradermacher/Mistral-Nemo-Instruct-2407-12B-Thinking-M-Claude-Opus-High-Reasoning-GGUF`
- **Quantization**: Q4_K_M static GGUF (dequantized to bf16)
- **Parameters**: ~12.2B (effective ~11.6B)

## Hardware

- **Arch**: p150 (Blackhole, single chip)
- **DRAM bandwidth**: 512 GB/s

## Test Config (finalized)

- `optimization_level`: 2
- `trace_enabled`: True
- `experimental_weight_dtype`: `"bfp_bf8"`
- `batch_size`: 32
- `num_layers`: (not overridden — full 40-layer model)

## Results

| Metric | Value |
|--------|-------|
| Samples per second | 20.09 |
| TTFT (ms) | 490.4 |
| Prefill PCC | 0.983401 ✅ |
| First decode PCC | 0.997257 ✅ |
| Wall time (full test) | 27:36 |

## Roofline Analysis (first decode graph)

- **Bound**: DRAM
- **Top perf (roofline ceiling)**: 27.79 samples/sec
- **Achieved**: 20.09 samples/sec (**72.3% of target**)
- **Top perf time per step**: 35.99 ms
- **Params**: 12.25B (12.71 GB at bfp_bf8)
- **KV cache**: 0.625 GB

## Notes

- Distinct from the already-benchmarked i1 (imatrix) variant (`12B_Thinking_i1_GGUF`). This is the static Q4_K_M GGUF.
- Full-model accuracy test took ~27 minutes; the GGUF loader dequantizes at load time, producing the same BF16 activations as the i1 variant.
- 72.3% of roofline is reasonable; the ~28% gap reflects trace overhead and memory bandwidth contention across 32-batch decode steps.
