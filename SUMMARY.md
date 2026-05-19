loader_path: third_party.tt_forge_models.gemma3.causal_lm.pytorch.loader
variant_id: 4B_Instruct_bnb_4bit
arch: p150
status: DONE_FAIL
test_function: test_gemma3_4b_instruct_bnb_4bit
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "bitsandbytes 4-bit NF4/FP4 quantization requires CUDA for forward pass; FP4 quantization state not initialized without .cuda() call; p150 benchmark machine has no CUDA"

# Benchmark added: test_gemma3_4b_instruct_bnb_4bit

## Test
tests/benchmark/test_llms.py::test_gemma3_4b_instruct_bnb_4bit

## Model
- HF name:    unsloth/gemma-3-4b-it-unsloth-bnb-4bit
- Loader:     third_party.tt_forge_models.gemma3.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_4B_IT_BNB_4BIT ("4B_Instruct_bnb_4bit")

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

## Failure details

The `4B_Instruct_bnb_4bit` variant uses bitsandbytes (BNB) 4-bit NF4/FP4 quantization.
The loader (`third_party/tt_forge_models/gemma3/causal_lm/pytorch/loader.py`) sets
`device_map="cpu"` and calls `AutoModelForCausalLM.from_pretrained` for this variant.
The pretrained model (`unsloth/gemma-3-4b-it-unsloth-bnb-4bit`) stores weights in
BNB 4-bit quantized format.

Without CUDA, the model loads into memory but bitsandbytes cannot initialize the FP4
quantization state (requires `.cuda()` call to initialize dequantization kernels). Any
forward pass attempt raises:

```
bitsandbytes/nn/modules.py:415: in fix_4bit_weight_quant_state_from_module
    assert module.weight.shape[1] == 1
AssertionError
UserWarning: FP4 quantization state not initialized. Please call .cuda() or
             .to(device) on the LinearFP4 layer first.
```

The p150 (Blackhole) benchmark machine has no NVIDIA GPU / CUDA. Even after installing
`bitsandbytes==0.49.2`, the CPU kernels also failed to load:

```
Failed to load CPU gemm_4bit_forward from kernels-community:
No module named 'kernels'. Please make sure you already pip install kernels >= 0.11.1
```

This is a loader-level issue: the loader needs a CUDA-free code path for the
`4B_Instruct_bnb_4bit` variant (e.g., load non-quantized weights from config or use a
non-BNB variant as a fallback). The fix belongs in `third_party/tt_forge_models` and is
out of scope for this skill.

Two general harness fixes were also made as part of this work:
1. `tests/benchmark/llm_utils/decode_utils.py`: `init_static_cache()` now calls
   `config.get_text_config()` when available, to handle models (like `Gemma3Config`) that
   wrap text-model attributes in a nested sub-config. Fixes:
   `AttributeError: 'Gemma3Config' object has no attribute 'hidden_size'`
2. `tests/benchmark/benchmarks/llm_benchmark.py`: The `get_weight_dtype_config_path()`
   call is now guarded with `hasattr` (matching the pattern in
   `tests/runner/testers/torch/dynamic_torch_model_tester.py`). Fixes:
   `AttributeError: 'ModelLoader' object has no attribute 'get_weight_dtype_config_path'`

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation/execution step

## Files changed
- tests/benchmark/test_llms.py (new test function `test_gemma3_4b_instruct_bnb_4bit`)
- .github/workflows/perf-bench-matrix.json (new matrix entry)
- tests/benchmark/llm_utils/decode_utils.py (general harness fix: nested config support)
- tests/benchmark/benchmarks/llm_benchmark.py (general harness fix: hasattr guard)

## tt-forge-models submodule
no change
