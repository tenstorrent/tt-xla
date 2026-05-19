loader_path: third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader
variant_id: Simple_V2_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf
samples_per_second: 33.407
ttft_ms: 307
prefill_pcc: 0.999228
first_decode_pcc: 0.998096
top_perf_samples_per_sec: 46.0471
pct_of_target: 72.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf (p150)

## Model
- **HuggingFace repo**: hienphantt161/Document-Validation-Qwen2.5-VL-Simple-V2
- **Architecture**: Qwen2ForCausalLM (7.6B params, extracted from Qwen2_5_VLForConditionalGeneration)
- **Variant**: Simple_V2_i1_GGUF
- **Loader**: third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader

## Test Configuration
- **optimization_level**: 2 (DEFAULT_OPTIMIZATION_LEVEL)
- **trace_enabled**: true (DEFAULT_TRACE_ENABLED)
- **experimental_weight_dtype**: bfp_bf8 (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
- **batch_size**: 32 (default)
- **input_sequence_length**: 128 (default)

## Measured Performance (p150 / blackhole)
- **Samples per second**: 33.407
- **TTFT (ms)**: 307
- **Prefill PCC**: 0.999228 (≥ 0.94 required)
- **First decode PCC**: 0.998096 (≥ 0.94 required)

## Roofline Analysis
- **Theoretical max (DRAM-bound)**: 46.0471 samples/sec
- **Achieved**: 33.407 samples/sec = **72.5% of roofline**
- **Bound**: DRAM
- **Model params**: ~7.6B (effective 7.07B at bfp_bf8)
- **DRAM bandwidth**: 512 GB/s

## Notes
- The previous benchmark branch recorded trace_enabled=false with ~4.2 samples/sec (9% of
  roofline). This was based on a mischaracterized "hang" that was actually a cold-kernel
  compilation timeout (57 min) from the first run. Subsequent runs with trace=True complete
  successfully and achieve 72.5% of the DRAM-bound roofline ceiling.
- The loader extracts Qwen2ForCausalLM from the multimodal Qwen2_5_VLForConditionalGeneration
  checkpoint because the GGUF qwen2vl architecture is not yet supported by transformers.
- llm_benchmark.py required a hasattr guard around get_weight_dtype_config_path() since
  this loader does not implement that method (fixed in this branch).
- Submodule must be on branch arch-c-36-tt-xla-dev/nsmith/hf-bringup-47
  (commit 215d1080a2) to see the document_validation_qwen2_5_vl_simple_v2_i1_gguf module.
- Cold compilation (MLIR→hardware binary) takes ~3-4 hours for full trace_enabled=True run;
  subsequent warm-process runs reuse the in-memory JIT cache and are faster.
