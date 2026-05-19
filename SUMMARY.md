loader_path: third_party.tt_forge_models.gemma_3_12b_it_heretic_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_HERETIC_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_gemma_3_12b_it_heretic_gguf
samples_per_second: 13.409369908269747
ttft_ms: 956.219521
prefill_pcc: 0.992953
first_decode_pcc: 0.987756
top_perf_samples_per_sec: 26.328925778723903
pct_of_target: 50.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gemma_3_12b_it_heretic_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_3_12b_it_heretic_gguf

## Model
- HF name:    mradermacher/gemma-3-12b-it-heretic-GGUF
- Loader:     third_party.tt_forge_models.gemma_3_12b_it_heretic_gguf.causal_lm.pytorch.loader
- Variant:    12B_IT_HERETIC_Q4_K_M_GGUF
- Arch:       p150 (blackhole)

## Configuration
| Parameter                 | Value   |
|---------------------------|---------|
| optimization_level        | 2       |
| trace_enabled             | true    |
| experimental_weight_dtype | bfp_bf8 |
| batch_size                | 32      |

## Results (full model, p150)
| Metric                    | Value               |
|---------------------------|---------------------|
| Prefill PCC               | 0.992953            |
| First decode PCC          | 0.987756            |
| TTFT (ms)                 | 956.219521          |
| Samples per second        | 13.409369908269747  |
| Roofline target (sps)     | 26.328925778723903  |
| % of roofline             | 50.9%               |
| Roofline bound            | DRAM                |

## Notes
- Gemma 3 12B IT Heretic GGUF (Q4_K_M quantization) on p150 (blackhole single-chip).
- Bug fix applied: `setup_model_and_tokenizer` in `llm_benchmark.py` now synchronizes
  per-layer `attention_type`, `layer_type`, `is_sliding`, and `sliding_window` attributes
  after overriding `config.layer_types` to `["full_attention"]`. Without this fix, Gemma 3's
  mixed-attention forward pass KeyErrors on `'sliding_attention'` because the config was
  updated after layer initialization.
- GGUF model loaded with `dtype_override=torch.bfloat16` to align with StaticCache dtype.
- Running at 50.9% of DRAM roofline (26.33 sps target). Both PCC thresholds pass (threshold: 0.94).
