# GPT-OSS 20B Data Parallelism (DP=4, TP=8) Debugging Notes

## Goal
Run GPT-OSS 20B on a 4x8 galaxy mesh with DP=4 and TP=8, achieving good PCC
(Pearson Correlation Coefficient) against CPU golden reference. Currently
running on 1x8 (TP-only) works fine. The 4x8 (DP+TP) variant produces all-zero
TT logits.

## How to Run
```bash
docker exec -u 4123 tt-xla-ird-mvasiljev bash -c \
  "cd ~/tt-xla && source venv/activate && \
   pytest -sv tests/benchmark/test_llms.py \
     -k test_gpt_oss_20b_tp_galaxy_batch_size_64 \
     --num-layers 6 --max-output-tokens 3"
```
Use `--num-layers 6` for faster iteration. Full model has 24 layers.

## Key Files
- `tests/benchmark/benchmarks/llm_benchmark.py` — core benchmark logic, sharding
  setup, KV cache sharding, compile, warmup, metric collection
- `tests/benchmark/test_llms.py` — test definitions, `_gpt_oss_20b_shard_spec_fn`,
  `_batch_parallel_input_sharding_fn`, test entry point
- `third_party/tt_forge_models/gpt_oss/pytorch/loader.py` — ModelLoader, mesh
  config (`(4,8), ("batch","model")` for 32 devices), default shard specs
- `python_package/tt_torch/sharding.py` — `sharding_constraint_hook`,
  `_partition_spec_to_sdy_sharding`
- `venv/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`
  — HuggingFace model implementation, especially `GptOssExperts.forward` (MoE)

## Symptom
When running with DP (batch_size=64, input_sharding_fn=_batch_parallel_input_sharding_fn),
TT output logits are all exactly zero:
```
Warmup step0 logits: shape=[64, 17, 201088], std=0.000000, min=0.0000,
    max=0.0000, sum=0.0000, nonzero=0/218783744
Step0 logits stddev: cpu=3.397495, tt=0.000000
```
This causes PCC computation to fail (denominator is zero).

## Confirmed Working Configuration (Baseline)
- **TP-only (1x8 mesh, batch_size=32, no input_sharding_fn)**:
  - Produces `std=3.485092`, `PCC=0.598124` (with 6 layers)
  - This confirms model weights, compilation, and inference are fine without DP

## Experiments Tried (All Still Produced All-Zero Logits)

### 1. Remove lm_head sharding constraint hook
- **What**: Commented out the `sharding_constraint_hook` on `model.lm_head`
  that applies `(None, None, None)` sharding to all-gather the logits
- **Result**: Still zeros. The computation itself produces zeros, not just a
  gather problem.

### 2. Custom `_gpt_oss_20b_shard_spec_fn` with explicit weight sharding
- **What**: Created explicit shard specs for ALL model weights including:
  - `embed_tokens.weight` → `(None, None)` (replicated)
  - `norm.weight` → `(None,)` (replicated)
  - `lm_head.weight` → `(None, None)` (replicated)
  - Attention QKV → `("model", None)`, biases → `("model",)`
  - Attention O → `(None, "model")`, bias → `(None,)`
  - `sinks` → `(None,)`, layer norms → `(None,)`
  - MoE router → `(None, None)`
  - MoE experts gate_up/down_proj → `("model", None, None)`, biases → `("model", None)`
- **Result**: Still zeros with DP.

### 3. Replicate MoE expert weights (`_gpt_oss_20b_galaxy_shard_spec_fn`)
- **What**: Created a variant shard spec where MoE expert weights are fully
  replicated: `gate_up_proj → (None, None, None)`, `down_proj → (None, None, None)`,
  biases → `(None, None)`. Hypothesis was that the
  `repeat+view+bmm` pattern in `GptOssExperts.forward` confuses SPMD when
  batch dim is sharded.
- **Result**: Still zeros.
- **Note**: `_gpt_oss_20b_galaxy_shard_spec_fn` still exists in test_llms.py
  but is NOT used by the current test.

### 4. Replicate KV cache batch dimension
- **What**: Set `kv_batch_axis = None` (replicated) even when
  `input_sharding_fn` is present, so KV cache is `(None, "model", None, None)`
  instead of `("batch", "model", None, None)`.
- **Result**: Still zeros.
- **Reverted**: KV cache batch axis is currently back to conditional:
  `kv_batch_axis = "batch" if input_sharding_fn is not None else None`

### 5. input_ids pre-forward hook with sharding_constraint
- **What**: Added a `register_forward_pre_hook` on the model to apply
  `torch.ops.tt.sharding_constraint` directly to `kwargs["input_ids"]` with
  `("batch", None)` sharding inside the compiled graph. Hypothesis was that
  `xs.mark_sharding` on dynamic inputs doesn't get captured by `torch.compile`.
- **Result**: Still zeros.
- **Replaced by**: Experiment 6.

### 6. embed_tokens forward hook with sharding_constraint (CURRENT STATE)
- **What**: Added `sharding_constraint_hook(model.model.embed_tokens, mesh,
  ("batch", None, None))` as a forward hook on the embedding layer. This
  constrains the embedding output (not input_ids) to be batch-sharded.
- **Result**: Still zeros. (Confirmed in the last test run.)
- **Status**: This hook is still in the code at `llm_benchmark.py:372-379`.

## Current State of Code Changes

### `llm_benchmark.py` (lines ~362-384)
```python
kv_batch_axis = "batch" if input_sharding_fn is not None else None
for layer in input_args["past_key_values"].layers:
    xs.mark_sharding(layer.keys, mesh, (kv_batch_axis, "model", None, None))
    xs.mark_sharding(layer.values, mesh, (kv_batch_axis, "model", None, None))

if input_sharding_fn is not None:
    input_sharding_fn(mesh, input_args)
    # embed_tokens hook (Experiment 6)
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_hook = sharding_constraint_hook(
            model.model.embed_tokens, mesh, ("batch", None, None)
        )
        model.model.embed_tokens.register_forward_hook(embed_hook)

# lm_head hook (always applied for multichip)
if hasattr(model, "lm_head") and model.lm_head is not None:
    hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
    model.lm_head.register_forward_hook(hook)
```

### `llm_benchmark.py` (lines ~426-432) — diagnostic prints
```python
wl = warmup_logits[0].to(torch.float32).flatten()
print(
    f"Warmup step0 logits: shape={list(warmup_logits[0].shape)}, "
    f"std={wl.std().item():.6f}, min={wl.min().item():.4f}, "
    f"max={wl.max().item():.4f}, sum={wl.sum().item():.4f}, "
    f"nonzero={wl.nonzero().shape[0]}/{wl.numel()}"
)
```

### `llm_benchmark.py` (lines ~448-449) — re-shard after input reconstruction
```python
if is_multichip and input_sharding_fn is not None:
    input_sharding_fn(mesh, input_args)
```

### `test_llms.py` — `_gpt_oss_20b_shard_spec_fn` (lines ~1100-1127)
Explicit shard specs for all weights. See "Experiment 2" above for details.

### `test_llms.py` — test entry (lines ~1225-1246)
```python
test_llm_tp(
    ModelLoader, variant, output_file,
    num_layers=num_layers, request=request,
    max_output_tokens=max_output_tokens,
    batch_size=64,
    input_sharding_fn=_batch_parallel_input_sharding_fn,
    shard_spec_fn=_gpt_oss_20b_shard_spec_fn,
    arch="wormhole_galaxy",
    optimization_level=1,
)
```

## Key Observations / Hypotheses

### The problem is DP-specific
Without DP (1x8, batch_size=32), the model produces meaningful logits. With DP
(4x8, batch_size=64, input_ids sharded on "batch" axis), all logits are zero.

### `xs.mark_sharding` on dynamic inputs may not be captured by `torch.compile`
`_batch_parallel_input_sharding_fn` calls `xs.mark_sharding(input_args["input_ids"],
mesh, ("batch", None))`. This is an *eager* annotation on the input tensor. When
`torch.compile(backend="tt")` traces the model into an FX graph, these annotations
may not be part of the graph. The SPMD partitioner might then assume inputs are
replicated, causing a mismatch between actual data layout and compiler assumptions.

Attempts to fix this via `sharding_constraint_hook` on `embed_tokens` output did
not help — suggesting the issue may be deeper, possibly in how the SPMD partitioner
handles the 2D mesh topology.

### MoE `repeat+view+bmm` pattern
`GptOssExperts.forward` (non-CPU path, lines 128-139 of modeling_gpt_oss.py):
```python
hidden_states = hidden_states.repeat(num_experts, 1)
hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
gate_up = torch.bmm(hidden_states, self.gate_up_proj) + ...
...
next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
next_states = next_states * routing_weights.transpose(0, 1).view(...)
next_states = next_states.sum(dim=0)
```
The `repeat` + `view` re-inserts the batch dimension at a different position.
When inputs are batch-sharded, the SPMD partitioner may not correctly propagate
sharding through `repeat → view → bmm → view → sum`. This is a strong candidate
for the root cause but replicating MoE weights (Experiment 3) didn't fix it,
suggesting the computation pattern itself is the problem, not just the weight
sharding.

## Next Steps to Try

1. **Add sharding constraints inside MoE forward path**: Register hooks on
   `GptOssExperts` or its sublayers to explicitly constrain intermediate tensor
   shardings through the repeat/view/bmm pattern.

2. **Check if the SPMD partitioner generates an all_reduce where it should do
   all_gather (or vice versa)**: Inspect the compiled StableHLO/MLIR to see
   what collective operations are generated for the DP dimension.

3. **Try marking the MoE hidden_states as replicated before the `repeat` call**:
   If the batch-sharded tensor going through `repeat` confuses things, explicitly
   all-gather it before MoE.

4. **Inspect the actual StableHLO graph**: Look at what the SPMD partitioner
   produces for the 4x8 case vs 1x8. Check if collective ops are correct.
   The compiled modules are exported to the `modules/` directory.

5. **Try DP=2, TP=8 on a 2x8 sub-mesh**: Reduce the DP factor to see if even
   minimal DP causes zeros, confirming it's a DP mechanism bug rather than a
   scale issue.

6. **Test with a simpler model (no MoE) on 4x8**: E.g., Llama 70B with DP.
   If that works, the issue is MoE-specific with DP.

7. **Check if `_batch_parallel_input_sharding_fn` needs to shard more inputs**:
   Currently only `input_ids` is sharded. Check if `attention_mask`,
   `position_ids`, or `cache_position` also need batch sharding annotations.

## Environment
- Branch: `mvasiljevic/add_profile_skills`
- Docker container: `tt-xla-ird-mvasiljev`
- Device: Galaxy (4x8 = 32 Wormhole chips)
- Mesh config from loader: `(4, 8), ("batch", "model")` for 32 devices
- Python 3.12, torch 2.7.0, jax 0.7.1
