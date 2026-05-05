# Streaming Design — DeepSeek-V4-Flash on TT

## Goal

Run end-to-end prefill + decode of DeepSeek-V4-Flash (43 blocks, 256
routed experts, BF16 dequantized) on a 32-device TT mesh while keeping
**peak host RAM below ~15 GB** instead of the ~280 GB the all-at-once
loader needs.

The compiled forward graph and the device-resident shards are unchanged
from the existing e2e flow — only the load-and-ship pipeline differs.

## Problem statement

`test_deepseek_v4_full_e2e._build_and_load_full_model` does:

```python
model = mdo.Transformer(args).eval()                        # empty CPU skeleton
sd = weight_loader.load_transformer_state_dict(             # ★ peaks host RAM
    range(args.n_layers), include_mtp=False
)
model.load_state_dict(sd, strict=False)                     # ref the loaded tensors
del sd; gc.collect()
enable_sparse_mlp(model, ...)                               # rewrites MoE on CPU
```

After step ★, ~280 GB sits in host RAM as the full state dict. Even
after `load_state_dict` + `del sd`, CPython holds onto refs through the
model's params until the model is shipped to TT (`model.to(device)`).
The peak overlaps the dequant + load + sparse-MLP rewrite phases.

Dev hosts with 256 GB / 512 GB RAM either OOM hard or spill to swap and
hang.

## Constraints

1. **Forward graph must be the full 43-block transformer** for one
   `torch.compile(model, backend="tt")` call. Per-block compile +
   per-block execute is not on the table — residual + HC connections
   thread state across blocks; per-block execution would require
   manually staging hidden states 43 times per forward step and would
   blow the 4-compile budget the e2e test relies on.
2. **`enable_sparse_mlp` must run on CPU before sharding**, because it
   constructs `StackedExperts` by stacking 256 individual `Expert.weight`
   tensors. That stacking only makes sense on full unsharded weights.
3. **`xs.mark_sharding` must run on the raw `nn.Parameter`** before
   `apply_weight_dtype_overrides` registers parametrizations — see the
   ordering note in `test_deepseek_v4_full_e2e.py`. Streaming has to
   preserve this.
4. **`weight_loader.load_block_state_dict(layer_id)`** is the unit of
   load granularity HF / `safetensors` give us. Each call is one block's
   ~6.5 GB of weights.

## Strategy (selected): per-block load → per-block ship → per-block free

Pseudocode:

```python
# 1. Build empty Transformer skeleton on CPU
model = mdo.Transformer(args).eval()

# 2. Load + ship the small top-level params first (~2 GB)
top_sd = weight_loader.load_top_level_state_dict()
model.load_state_dict(top_sd, strict=False)
del top_sd; gc.collect()

# Move just embed / norm / head / hc_head_* to TT
for sub in ("embed", "norm", "head"):
    setattr(model, sub, getattr(model, sub).to(device))
# (hc_head_fn / base / scale are direct nn.Parameters on Transformer;
#  iterate over them or use `_parameters` dict)

# 3. Stream the 43 blocks
for layer_id in range(args.n_layers):
    block_sd = weight_loader.load_block_state_dict(layer_id)
    # The load_block_state_dict keys are prefixed `layers.{layer_id}.…`
    # Strip the prefix so we can load_state_dict on the block directly:
    stripped = {k.removeprefix(f"layers.{layer_id}."): v for k, v in block_sd.items()}
    model.layers[layer_id].load_state_dict(stripped, strict=False)
    del block_sd, stripped

    # Rewrite this block's ffn (MoE → A2aSparseMLP) ON CPU
    enable_sparse_mlp(model.layers[layer_id], mesh=mesh_shape, ...)

    # Ship this block to TT
    model.layers[layer_id] = model.layers[layer_id].to(device)

    # Mark sharding on this block's params
    apply_block_sharding(model.layers[layer_id], mesh)

    gc.collect()

# 4. apply weight dtype overrides ONCE on the now-fully-resident model
apply_weight_dtype_overrides(model, weight_dtype_overrides)

# 5. compile + run as in the existing e2e test
compiled = torch.compile(model, backend="tt")
…
```

Peak host RAM ≈ **one transient block (~6.5 GB) + Python overhead**.
Previously-shipped blocks have their CPU storage freed because
`block.to(device)` returns new XLA tensors and we drop the CPU refs via
`gc.collect()`.

## Alternatives considered

### A. Build full model, all-at-once load (status quo)

Peak: ~280 GB. Rejected by problem statement.

### B. Lazy `meta` device skeleton + per-param fill

Use `torch.device("meta")` to create the model with no storage, then
fill each param individually. Equivalent end-state but more complex —
`meta` tensors don't support `to_empty(device="xla:0")` cleanly with
SPMD, and `enable_sparse_mlp`'s `StackedExperts` construction requires
real tensor contents to stack. **Rejected** as it doesn't reduce the
peak any further than per-block streaming.

### C. Per-block compile + execute

Compile a single-block forward, run it, save hidden state to host (or
keep on device), discard the block's weights, load next block.

Pros: theoretical minimum device memory.
Cons:
- 43 compiles per forward step (catastrophic compile time)
- Hidden state must be carried across blocks — either materialize on
  host (huge bandwidth cost) or pin on device (defeats the streaming)
- HC `pre`/`post` connections require both `residual` and `comb`/`post`
  to span blocks — would require careful wrapping
- KV cache lives **inside** each block (`Attention.kv_cache` buffer),
  so swapping blocks in/out across decode steps would lose kv state

**Rejected.** Strategy of choice for memory-extreme inference, but
overkill for our 32-device target where each device's share is ~200 MB
per block.

### D. Quantized weight loading

Load weights in their native FP4/FP8 packed form and dequant on device
on demand. Cuts host RAM in half pre-dequant.

Possible follow-up if per-block streaming still doesn't fit. Out of
scope for v1.

## Compile + run (unchanged from e2e)

Once the model is fully loaded to TT and sharded:

- Mirror the prefill+decode loop from
  [`test_deepseek_v4_full_e2e.py`](../tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py).
- 4-compile budget (2 const-eval + 2 main forward = prefill + decode).
- `start_pos` passed as fresh CPU→device tensor each step so dynamo
  treats it as a symbolic graph input (no per-step recompile).

## Why per-block `enable_sparse_mlp` is safe

`enable_sparse_mlp` walks `named_modules()` of whatever you pass in, and
replaces any module that looks like a MoE MLP with `A2aSparseMLP`. The
walk is purely local — it doesn't touch sibling blocks or top-level
attributes. Calling it on a single `Block` instance only rewrites that
block's `.ffn`, leaving the rest of the model untouched.

This lets us stream one block, rewrite its MoE on CPU, ship the
rewritten block to device, free CPU storage, then move on — without ever
holding more than one block's expert weights on host.

## Why this stays compatible with `apply_weight_dtype_overrides`

`apply_weight_dtype_overrides` registers `parametrize` modules on
`nn.Parameter` objects identified by glob. Glob matching against
`layers.*.ffn.mlp.experts.gate_proj` etc. works regardless of whether
the Parameter is on CPU or TT. We call it **once at the end**, after all
blocks are streamed, so the parametrization wraps the device-resident
storage uniformly.

The mark_sharding-before-parametrize ordering required by the e2e test
is preserved: per-block sharding happens in step 3 (during streaming),
parametrize-overrides happens in step 4 (after streaming).

## Testing

Mirror the e2e test's "no PCC, just produce tokens" contract for v1.
Numerical correctness is implicitly validated by the existing
`test_transformer_*` family — streaming changes only the load path, not
the forward graph.

If we need explicit validation of the streamed model:

- `test_streaming_transformer_decode` (mirror of
  `test_transformer_decode` but using `streaming_loader.stream_load_transformer`
  in place of `make_real_transformer + init_transformer_weights`)

## Out of scope (v1)

- KV cache offload to host between decode steps
- Resume-from-partial-load on failure
- Multi-host distributed streaming (single host with 32 devices only)
- Any change to the compile/forward graph

## See also

- [`MEMORY_BUDGET.md`](./MEMORY_BUDGET.md) — exact byte counts
- [`OPEN_QUESTIONS.md`](./OPEN_QUESTIONS.md) — unresolved issues
