loader_path: third_party.tt_forge_models.aidc_llm_laos_4b_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_aidc_llm_laos_4b_gguf
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
failure_reason: "Device (n300, wormhole) repeatedly hung during XLA compilation warmup; compilation failed with RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13 after model loaded and CPU forward pass completed; device entered persistent Ethernet timeout state after each compilation attempt and required hardware-level intervention to recover; unable to determine if error is a real compiler bug or device instability"

# Benchmark added: test_aidc_llm_laos_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aidc_llm_laos_4b_gguf

## Model
- HF name:    mradermacher/aidc-llm-laos-4b-GGUF
- Loader:     third_party.tt_forge_models.aidc_llm_laos_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AIDC_LLM_LAOS_4B_Q4_K_M ("Q4_K_M")
- Architecture: Gemma3Text (gemma3_text model type)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Infrastructure fixes (llm_benchmark.py)
Two bugs were fixed in `tests/benchmark/benchmarks/llm_benchmark.py`:

### 1. Gemma3 layer_types sync fix
The existing code overrode `model.config.layer_types = ["full_attention"] * N` but
did not update the per-layer `attention_type` / `layer_type` attributes that Gemma3
decoder layers store at `__init__` time. When the model ran forward, it built
`position_embeddings` keyed by the modified `config.layer_types` (all `full_attention`),
but then indexed it with the original per-layer `attention_type = "sliding_attention"`,
causing:
    KeyError: 'sliding_attention'
Fix: after overriding `config.layer_types`, iterate `model.modules()` and sync
`attention_type`, `layer_type`, `is_sliding`, and `sliding_window` attributes.

### 2. get_weight_dtype_config_path guard
The fallback path `model_loader.get_weight_dtype_config_path()` was called
unconditionally when `weight_dtype_overrides` was None, but this method is not
defined in the `ForgeModel` base class. Fixed by guarding with
`elif hasattr(model_loader, "get_weight_dtype_config_path")`.

## Bring-up results summary
| Run | Error | Root cause |
|-----|-------|------------|
| 1   | Device hang (wait_for_non_mmio_flush) | Unknown; device crashed during 18-min GGUF download + first compile |
| 2   | RuntimeError: Read 0xffffffff over PCIe (board hung) | Device from run 1 not reset |
| 3   | KeyError: 'sliding_attention' (CPU) | Missing per-layer attribute sync when layer_types overridden |
| 4   | AttributeError: get_weight_dtype_config_path | Missing hasattr guard |
| 5   | RuntimeError: INTERNAL Error code: 13 (XLA compile) | Unknown — device immediately hung after |
| 6+  | Timeout waiting for Ethernet (device init) | Device in unrecoverable state from run 5 |

## Measured (full model, defaults)
- Sample per second:  N/A (device unstable)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (Wormhole, 2-chip)

## Decode roofline (first decode graph, single-chip)
N/A — test did not complete; no perf_metrics JSON generated.

## Files changed
- tests/benchmark/test_llms.py (added test_aidc_llm_laos_4b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (Gemma3 layer_types sync fix + get_weight_dtype_config_path guard)
- SUMMARY.md

## tt-forge-models submodule
no change
