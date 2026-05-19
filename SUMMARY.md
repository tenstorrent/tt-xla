loader_path: third_party.tt_forge_models.llama_3_2_3b_instruct_awq.causal_lm.pytorch.loader
variant_id: 3.2_3B_Instruct_AWQ
arch: p150
status: DONE_FAIL
test_function: test_llama_3_2_3b_instruct_awq
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
failure_reason: "AWQ kernel incompatible with XLA backend: gptqmodel TorchAtenAwqLinear._fused_op_forward raises NotImplementedError when tensor device type is 'xla' (expects 'cpu'); uses torch.ops.aten._weight_int4pack_mm_for_cpu which cannot run through TT XLA compilation pipeline"

# Benchmark added: test_llama_3_2_3b_instruct_awq

## Test
tests/benchmark/test_llms.py::test_llama_3_2_3b_instruct_awq

## Model
- HF name:    AMead10/Llama-3.2-3B-Instruct-AWQ
- Loader:     third_party.tt_forge_models.llama_3_2_3b_instruct_awq.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_2_3B_INSTRUCT_AWQ ("3.2_3B_Instruct_AWQ")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The model uses gptqmodel's `TorchAtenAwqLinear` kernel. Its `_fused_op_forward`
method begins with:

    if x.device.type != "cpu":
        raise NotImplementedError

When running through the TT XLA backend (`torch.compile(model, backend="tt")`),
tensors have device type `"xla"`. This triggers the `NotImplementedError` during
the warmup step (first forward pass that triggers compilation). The kernel uses
`torch.ops.aten._weight_int4pack_mm_for_cpu` which is inherently CPU-only and
cannot be traced or compiled for TT hardware.

Resolution requires either:
- A gptqmodel AWQ kernel implementation aware of XLA/TT device types, OR
- Loading the model with dequantized weights (requires loader changes, out of scope)

The fix is in the tt-forge-models repo (loader or model configuration), not in
tt-xla benchmarking infrastructure.

## Infrastructure fix applied
`tests/benchmark/benchmarks/llm_benchmark.py`: Changed unconditional call to
`model_loader.get_weight_dtype_config_path()` to use `getattr(model_loader,
"get_weight_dtype_config_path", None)` with a None check, so loaders that don't
define this method (pre-quantized AWQ/GGUF models) don't raise AttributeError.
This is a general fix, not model-specific.

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model never compiled successfully)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (added test_llama_3_2_3b_instruct_awq)
- tests/benchmark/benchmarks/llm_benchmark.py (general infra fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added llama_3_2_3b_instruct_awq entry)

## tt-forge-models submodule
no change — submodule HEAD remains 80b67442
