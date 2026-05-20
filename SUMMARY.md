loader_path: third_party.tt_forge_models.falcon_mamba.causal_lm.pytorch.loader
variant_id: Falcon_Mamba_7B
arch: p150
status: DONE_FAIL
test_function: test_falcon_mamba_7b
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
failure_reason: "TT-MLIR ConvertStableHLOToTTIR fails on vhlo.scatter_v2 from FalconMambaCache.update_conv_state: 'ttir.repeat' op Input tensor shape (4,1,1) cannot repeat to output (32,8192,4) using repeat value 32"

# FalconMamba 7B Benchmark — p150 — DONE_FAIL

## Summary

FalconMamba 7B (`tiiuae/falcon-mamba-7b`, variant `Falcon_Mamba_7B`) could not be
compiled on the TT Blackhole (p150) hardware.  Every attempt to compile the model
graph — at any optimization level — failed with the same error in the
`ConvertStableHLOToTTIR` MLIR pass.

## Failure

```
loc("scatter.115"): error: 'ttir.repeat' op
  Input tensor shape (4,1,1) at index 0 does not repeat to output (32,8192,4)
  using repeat value 32.
Failed to convert from SHLO to TTIR module
ValueError: Error code: 13
```

### Root cause

FalconMamba is a Mamba / SSM-style architecture.  Unlike attention-based models
it uses no KV cache; instead it maintains a **convolutional state ring buffer**
(`FalconMambaCache.conv_states`) and a recurrent SSM state
(`FalconMambaCache.ssm_states`).

Every forward pass (both prefill and every decode step) calls
`FalconMambaCache.update_conv_state`, which contains the operation:

```python
conv_state = conv_state.roll(shifts=-1, dims=-1)   # torch.roll
conv_state[:, :, cache_position] = new_conv_state  # indexed scatter
```

The indexed assignment maps to `vhlo.scatter_v2` in StableHLO.  The
TT-MLIR `ConvertStableHLOToTTIR` pass tries to lower this as `ttir.repeat`,
but the update-tensor shape `(4,1,1)` (cache_position indices) cannot be
broadcast to the operand shape `(32,8192,4)` (the full conv state), and
conversion aborts.

### Why no harness workaround is possible

1. The scatter is **structural**: it exists in both prefill and decode, in the
   fast CUDA kernel path and the `slow_forward` fallback, and at every
   `optimization_level`.
2. Removing the cache entirely (`use_cache=False`) breaks autoregressive
   generation — the SSM state that captures context would be discarded after
   every decode step.
3. Editing `FalconMambaCache.update_conv_state` is out of scope (third-party
   loader code).

## Harness infrastructure added (reusable for future SSM models)

Even though FalconMamba itself cannot compile today, the bringup work extended
the benchmark harness with general SSM/Mamba support that will be needed for
any future Mamba-style model:

| File | Change |
|------|--------|
| `tests/benchmark/llm_utils/decode_utils.py` | `init_ssm_cache()`: allocates `FalconMambaCache` / `MambaCache` outside compiled graph (avoids `mark_static_address` trace-forbidden error). `init_static_cache()`: returns `None` for models without `num_attention_heads`. `LLMSamplingWrapper`: SSM cache detection + `cache_params` dispatch. |
| `tests/benchmark/benchmarks/llm_benchmark.py` | `construct_inputs`: SSM-correct `cache_position` (length = `conv_kernel`, not `seq_len`). `transfer_to_device`: moves `conv_states`/`ssm_states`. Warmup reset via `.reset()`. `get_weight_dtype_config_path` guarded with `hasattr`. |
| `tests/benchmark/llm_utils/__init__.py` | Export `init_ssm_cache`. |

## Recommended next steps

- File a TT-MLIR issue for `scatter` / `vhlo.scatter_v2` support in
  `ConvertStableHLOToTTIR`.  FalconMamba can be re-attempted once that op is
  lowerable.
- Alternatively, implement a Mamba-optimised TTNN kernel path that bypasses
  the generic scatter lowering.
