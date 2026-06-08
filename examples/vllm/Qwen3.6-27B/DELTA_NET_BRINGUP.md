# Bringup spec: `TTGatedDeltaNetAttention` for Qwen3.6-27B

This is a structured analysis of what would be required to write a TT-compatible
override of `GatedDeltaNetAttention` so that Qwen3.6-27B (and any other
Qwen3-Next-family model) can run under the TT vLLM plugin. It is a **spec**,
not an implementation — the goal is to give a precise picture of what needs to
exist before any code is written.

All file references in this document refer to:
- vLLM (installed): `venv/lib/python3.12/site-packages/vllm/`
- TT plugin (in-tree): `integrations/vllm_plugin/vllm_tt/`

## 1. Scope: what this document covers

A correct override needs three coupled pieces to work end-to-end:

1. **`TTGatedDeltaNetAttention` class** — a drop-in replacement for vLLM's
   `GatedDeltaNetAttention` with an equivalent `forward()` that uses only ops
   the TT compile path can lower.
2. **KV-cache-spec support for `MambaSpec`** — so the recurrent state and conv
   state actually get allocated on the chips.
3. **State plumbing** — `_dummy_run`, `execute_model`, and `bind_kv_cache` need
   to handle Mamba-style state buffers, not just attention KV caches.

Any one of these three in isolation does nothing useful. The aim of this spec
is to scope each one explicitly.

This document does **not** cover:
- M-RoPE for the vision encoder. That is a separate, also-required bringup.
- Vision encoder compile / multimodal input plumbing. Also separate.
- Performance. Once it works, optimizing the chunk decomposition for TT-metal
  is its own follow-up project.

## 2. Where DeltaNet sits in the Qwen3.6-27B model

The architecture is:

```
Qwen3_5ForConditionalGeneration         (outer, multimodal wrapper)
├── language_model: Qwen3_5ForCausalLM
│   ├── model: Qwen3_5Model = Qwen3NextModel
│   │   └── layers[0..63]: Qwen3NextDecoderLayer
│   │       ├── if layer_type == "linear_attention":
│   │       │     linear_attn: GatedDeltaNetAttention  ◄── 48 / 64 layers
│   │       └── elif layer_type == "full_attention":
│   │             self_attn: Qwen3NextAttention        ◄── 16 / 64 layers
│   └── lm_head: ParallelLMHead
└── visual: vision encoder (separate concern)
```

Pattern: `16 × (3 × GatedDeltaNetAttention + 1 × Qwen3NextAttention)`. **48 of 64
decoder layers** are DeltaNet. There is no way to side-step them.

## 3. What `GatedDeltaNetAttention` actually does

File: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`.

The forward has **three parts** (lines 489–565):

### Part 1: input projections (lines 504–534)

Standard `nn.Linear`-style projections produce six tensors from the input hidden
states:

| Tensor | Role | Shape (per token) |
|---|---|---|
| `query` (q) | Read key into the memory matrix | `(num_k_heads, head_k_dim)` |
| `key`   (k) | Write key into the memory matrix | `(num_k_heads, head_k_dim)` |
| `value` (v) | Value to write | `(num_v_heads, head_v_dim)` |
| `z` | Gating signal for the post-norm | `(num_v_heads, head_v_dim)` |
| `b` | Per-head "beta" (write strength) | `(num_v_heads,)` |
| `a` | Per-head "alpha" (decay) | `(num_v_heads,)` |

For Qwen3.5 the projections are `MergedColumnParallelLinear` (qkvz packed)
and a separate `in_proj_ba` for `(b, a)`. These are standard TP layers and
work fine on the TT path.

### Part 2: core attention (lines 539–553) — **this is the part that doesn't compile on TT**

```python
core_attn_out = torch.zeros((num_tokens, num_v_heads/tp, head_v_dim), …)
torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)
```

`gdn_attention_core` is a `direct_register_custom_op` (line 969 of the same
file). Its `fake_impl` is a no-op (line 958-966) — it lies to dynamo, claiming
the op writes nothing observable. That's a deliberate design choice for vLLM:
the actual recurrence math runs out-of-graph in the CUDA path, while dynamo is
told "this op mutates `core_attn_out` in place, just trust me." The real work
happens inside `_forward_core` / `_forward_core_decode_non_spec` (lines 667
and 887), which are pulled out of the compile region by `no_compile_layers`.

When dynamo traces under TT, the downstream `core_attn_out.reshape(-1, …)` ends
up with the wrong leading dimension because the custom op's fake metadata
doesn't propagate `num_tokens` correctly under our `_dummy_run` setup —
**this is the broadcast crash you saw**: `out:(48,128)` vs `z:(786432,128)`.
`out` reflects `num_tokens=1` (decode-shaped), `z` reflects `num_tokens=16384`
(prefill-shaped); they collide at the gated-norm in Part 3.

So this isn't just "we can't run a Triton kernel." It's "the whole custom-op
mechanism that vLLM uses to hide DeltaNet's stateful recurrence from dynamo
is incompatible with how the TT compile path handles `_dummy_run`."

The actual computation inside the custom op (when we re-implement it) is:

#### Part 2a: causal conv1d (state-carrying short-range mixer)

`mixed_qkv` (shape `(num_tokens, qkv_dim/tp)`) flows through a depthwise
causal 1-D convolution with `kernel_size = conv_kernel_size` (4 for
Qwen3.5). The conv carries a **state of shape `(conv_kernel_size-1, qkv_dim/tp)`**
across forward passes — it remembers the last `kernel_size-1` token activations
per request. Different code paths for prefill vs decode:

- **Prefill** (lines 753–768): `causal_conv1d_fn(...)` — does the conv for the
  whole prefill chunk and updates the conv_state at the end.
- **Decode** (lines 769–781): `causal_conv1d_update(...)` — applies the conv
  to one new token using the stored conv_state, and writes the new last
  `kernel_size-1` activations back.

Both of these are Triton kernels under the hood (in `vllm/.../ops/causal_conv1d.py`).
Both need to be replaced with PyTorch-only equivalents in the TT override.

#### Part 2b: gating values g and beta

Computed from `(a, b)` plus learned `A_log` and `dt_bias` (lines 791,
1013-1046):

```
softplus_x = softplus(a + dt_bias)
g    = -exp(A_log) * softplus_x       # decay
beta = sigmoid(b)                     # write strength
```

This is the easy part — pure elementwise, no state. The Triton wrapper
(`fused_gdn_gating_kernel`) just fuses the elementwise ops.

#### Part 2c: the actual delta-rule recurrence — **the heart of DeltaNet**

The recurrent state is a per-head **memory matrix** `S` of shape
`(num_v_heads/tp, head_v_dim, head_k_dim)`. For each token t the update is
the delta-rule:

```
S_t = S_{t-1} · diag(g_t)
       - β_t · (S_{t-1} k_t) k_tᵀ
       + β_t · v_t k_tᵀ
y_t = q_tᵀ · S_t
```

(The `diag(g_t)` decays the state; the `β · v · kᵀ - β · (S·k) · kᵀ` pair is
the "delta" update — it overwrites the part of the memory most aligned with
`k_t`.)

Two different kernels implement this depending on the call shape:

- **Prefill: `chunk_gated_delta_rule`** (lines 832-845). Operates on a chunk of
  consecutive tokens at once; uses a chunkwise-parallel algorithm so the
  recurrence is broken into blocks of size 64 (the `BT` parameter). Inputs:
  `q,k,v: (1, T, H, D)`, `g, beta: (1, T, H)`, `initial_state:
  (1, H, Dv, Dk)`. Outputs the per-token `y` (shape `(1, T, H, Dv)`) and the
  final state (same shape as initial_state).
- **Decode: `fused_recurrent_gated_delta_rule_packed_decode`** (line 921) —
  used when `enable_packed_recurrent_decode=True` and no speculative
  decoding. Single-step update for one token per active sequence; reads/writes
  the ssm_state in place.
- **Decode (fallback): `fused_sigmoid_gating_delta_rule_update`** (line 807,
  852) — same step but combined with the gating computation. Used for
  speculative decode and as the default decode path.

All three are Triton FLA kernels. All three would need pure-PyTorch
equivalents.

### Part 3: post-norm with gating + output projection (lines 558–565)

```
core_attn_out = core_attn_out.reshape(-1, head_v_dim)
z             = z.reshape(-1, head_v_dim)
core_attn_out = self.norm(core_attn_out, z)         # ← RMSNormGated
core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
output[:num_tokens], _ = self.out_proj(core_attn_out)
```

`RMSNormGated` does `rmsnorm(out) * silu(z)` — this is the line our trace
crashed at (`out * F.silu(z)` inside `layernorm.py:585`). Once Part 2 produces
correctly-shaped `core_attn_out`, this line works automatically.

`out_proj` is a standard `RowParallelLinear`.

## 4. The state, in detail

DeltaNet's `kv_cache` is **a tuple of two tensors**, declared by
`get_state_shape()` in `gdn_linear_attn.py:215-224` and computed by
`MambaStateShapeCalculator.gated_delta_net_state_shape` in
`mamba_utils.py:176-200`:

```python
# Per layer, per request:
conv_state_shape    = (conv_kernel_size - 1 + num_spec, conv_dim / tp)
                    # i.e. (3, (2·num_k_heads·head_k_dim + num_v_heads·head_v_dim) / tp)
                    # — the last (kernel-1) inputs to the conv1d, kept around.

temporal_state_shape = (num_v_heads / tp, head_v_dim, head_k_dim)
                    # — the memory matrix S of the recurrence.
```

For Qwen3.6-27B (eyeballing the HF config from the WebFetch earlier):
`num_v_heads=48, head_v_dim=128, head_k_dim=128, num_k_heads=16,
conv_kernel_size=4`. On 8 chips:

- `conv_state`: `(3, (2·16·128 + 48·128)/8)` = `(3, 1280)` bf16 per request,
  per layer. ~7.5 KB.
- `ssm_state` ("temporal" state): `(48/8, 128, 128) = (6, 128, 128)`
  bf16 per request, per layer. ~196 KB.

Across 48 DeltaNet layers and (say) 16 concurrent requests: ~150 MB total.
Manageable. Critical detail: **this is per request, not per token**.

The state lifecycle:
1. Allocated once at startup, sized for `num_blocks` × `block_size` slots,
   indexed via `non_spec_state_indices_tensor` (request → slot mapping).
2. Read at the start of each forward pass through `self.kv_cache`.
3. Written in place by the recurrence (or by `causal_conv1d_*`).
4. Persists across forward passes — that's the whole point. Decode uses the
   state left by prefill.

## 5. The four pieces a TT override must address

### A. `TTGatedDeltaNetAttention` — the math, in pure PyTorch

This is the main code artifact. Roughly the interface:

```python
class TTGatedDeltaNetAttention(nn.Module):
    def __init__(self, layer: GatedDeltaNetAttention):
        super().__init__()
        # Copy parameters, configs, and the projection sub-modules from layer.
        # All projections (in_proj_qkvz, in_proj_ba, out_proj, conv1d) are
        # standard TP layers that already work — keep references to them.
        self.in_proj_qkvz = layer.in_proj_qkvz   # MergedColumnParallelLinear
        self.in_proj_ba   = layer.in_proj_ba     # MergedColumnParallelLinear
        self.out_proj     = layer.out_proj       # RowParallelLinear
        self.conv1d       = layer.conv1d         # ColumnParallelLinear (conv1d weight!)
        self.norm         = layer.norm           # RMSNormGated (override separately if needed)
        self.A_log        = layer.A_log
        self.dt_bias      = layer.dt_bias
        # Shape constants:
        self.num_k_heads  = layer.num_k_heads
        self.num_v_heads  = layer.num_v_heads
        self.head_k_dim   = layer.head_k_dim
        self.head_v_dim   = layer.head_v_dim
        self.tp_size      = layer.tp_size
        # …etc.

    def forward(self, hidden_states, output):
        # Re-implement Part 1 (projections — already TT-friendly, copy logic).
        # Re-implement Part 2 (the custom op — this is where the real work is):
        #   - causal_conv1d_fn / causal_conv1d_update → torch conv1d + manual state update
        #   - fused_gdn_gating → softplus / exp / sigmoid in PyTorch
        #   - chunk_gated_delta_rule → chunkwise PyTorch loop
        #   - fused_recurrent_gated_delta_rule_packed_decode → single-step bmm
        # Re-implement Part 3 (gated norm + out_proj — TT-friendly already).
```

The hard sub-tasks, in order of difficulty:

1. **Causal conv1d + state update** in PyTorch. Easy (~30 lines).
2. **Gating computation** (`g`, `beta`). Easy (~5 lines).
3. **Chunkwise delta-rule prefill**. Medium. The chunkwise algorithm in the FLA
   kernel is published; the naïve version is a `for t in range(T)` loop over
   tokens applying the delta-rule update. A first-pass override should use the
   naive loop, verify numerics against the FLA reference on a CUDA box, and
   *then* (if perf is unacceptable) implement the chunkwise version that the
   FLA kernel does.
4. **Single-step decode update**. Easy once prefill is working (it's the same
   math, T=1).
5. **`RMSNormGated`**. May or may not need a TT override depending on whether
   the existing `forward_native` traces cleanly once shapes are correct. Worth
   trying without an override first; add a `TTRMSNormGated` later if needed.

Output: a new file `integrations/vllm_plugin/vllm_tt/tt_gated_delta_net.py`
containing `TTGatedDeltaNetAttention` plus a `tt_gated_delta_net_module(layer)`
factory.

Then register in `overrides.py`:

```python
from .tt_gated_delta_net import tt_gated_delta_net_module
ISINSTANCE_OVERRIDES = [
    (RotaryEmbedding, tt_rotary_embedding_module),
    (GatedDeltaNetAttention, tt_gated_delta_net_module),   # ← new
]
```

### B. `MambaSpec` in `TTModelRunner.get_kv_cache_spec`

File: `vllm_tt/model_runner.py:747-825`.

The current loop has branches for `Attention` and `MLAAttention` and falls
through (`else: continue`) for everything else. Mamba layers go through the
fall-through, so no spec is ever emitted for them. We need to add:

```python
elif isinstance(attn_module, MambaBase):
    kv_cache_spec[layer_name] = attn_module.get_kv_cache_spec(self.vllm_config)
    # ↑ MambaBase already implements get_kv_cache_spec and returns a MambaSpec.
```

(`MambaBase.get_kv_cache_spec` is implemented in `vllm/.../mamba/abstract.py:43`
— it consults `get_state_shape` / `get_state_dtype` / `mamba_type` and builds
a `MambaSpec`. We just need to call it.)

Pitfalls:
- `vllm_config.cache_config.mamba_block_size` and `mamba_page_size_padded` need
  to be set for this to work. Look at vLLM's `Platform.check_and_update_config`
  hooks — TPU's platform sets these; TT's may need to too.
- After this change, `TTModelRunner.initialize_kv_cache` (around line 2128) hits
  `raise NotImplementedError` on the `MambaSpec` branch — see (C) below.

### C. Allocate & bind Mamba state in `initialize_kv_cache`

File: `vllm_tt/model_runner.py:2120-2169`.

The current code only knows how to allocate `AttentionSpec` caches (one
`k_cache` and one `v_cache` per attention layer, both 4-D). For a `MambaSpec`
we need to allocate the **tuple** of `(conv_state, ssm_state)` per Mamba layer
and store both in `kv_caches[layer_name]`.

Rough sketch of what to add inside the per-`kv_cache_spec` loop:

```python
elif isinstance(kv_cache_spec, MambaSpec):
    shapes = kv_cache_spec.shapes        # tuple of state shapes from get_state_shape
    dtypes = kv_cache_spec.dtypes        # tuple of dtypes
    state_tensors = []
    for shape, dtype in zip(shapes, dtypes):
        # Allocate (num_blocks, *shape) — the leading dim indexes into the
        # request-slot table (non_spec_state_indices_tensor).
        full_shape = (num_blocks, *shape)
        t = torch.zeros(full_shape, dtype=dtype).to(self.device)
        state_tensors.append(t)
    kv_caches[layer_name] = state_tensors   # list of 2 tensors for GDN
```

Then `bind_kv_cache` will plumb these into `layer.kv_cache` so the layer's
forward can read/write them (line 2165).

Pitfalls:
- Sharding for the Mamba state across the TP mesh. The `xs.mark_sharding` call
  at line 2176 assumes 4-D attention caches; the conv_state is 2-D and the
  ssm_state is 3-D. Mamba state shapes already factor in `tp_size` (see
  `gated_delta_net_state_shape`), so the actual stored tensor per chip is
  already smaller. Probably want to leave Mamba state unsharded (or sharded
  along the head axis if it lines up with the mesh). Pick safe-and-replicated
  first.
- `bind_kv_cache` expects a `list[torch.Tensor]`; passing a list of two tensors
  per layer is the same pattern attention uses. Should be fine.

### D. The `attn_metadata` story

The CUDA path's `_forward_core` reads `attn_metadata: GDNAttentionMetadata`
(from `forward_context`) for fields like `num_prefills`, `num_decodes`,
`non_spec_state_indices_tensor`, `has_initial_state`, `non_spec_query_start_loc`,
etc. — the metadata that tells the recurrence "which token belongs to which
request, where in the state table is its slot."

Right now the TT plugin only constructs `TTMetadata` (attention metadata) in
`metadata.py`. For Mamba layers we need to either:

1. Construct `GDNAttentionMetadata` alongside it, or
2. Have the TT override read a subset of fields we already populate.

A first-pass override can probably get away with **option 2** for a single
request at a time:
- `num_prefills = 1, num_decodes = 0` (during prefill) or vice versa
- `non_spec_state_indices_tensor = torch.tensor([0])` (always slot 0 for a single
  request)
- `non_spec_query_start_loc = torch.tensor([0, num_tokens])`
- `has_initial_state = torch.tensor([False])` for first forward, `True` after
- `num_actual_tokens = num_tokens`

This works for `max_num_seqs=1` (our current config). Scaling to many concurrent
requests later means actually constructing `GDNAttentionMetadata` in
`_prepare_inputs`.

## 6. Why we crashed exactly where we did — the precise mechanism

This is worth keeping in mind because the fix needs to address it:

1. `model.compile(backend="tt")` ran dynamo on the full forward.
2. Dynamo reached `torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a,
   core_attn_out, prefix)` and looked up the **fake** implementation —
   `gdn_attention_core_fake` in `gdn_linear_attn.py:958-966`, which is a no-op
   that returns nothing.
3. The custom op is marked `mutates_args=["core_attn_out"]`, so dynamo
   conservatively assumes `core_attn_out` is written in place but **keeps its
   original FakeTensor shape**.
4. That FakeTensor shape was `(num_tokens=1, num_v_heads/tp=48, head_v_dim=128)`
   because the dummy run sent in 1 token. But `z` (which never touched the
   custom op) was already `(num_tokens=16384, num_v_heads/tp=48, head_v_dim=128)`
   from earlier in the same trace.
5. The reshape-and-multiply at the gated-norm sees `(48, 128)` vs `(786432, 128)`
   and fails.

The override sidesteps this entirely because **there is no custom op**: the
recurrence is a sequence of plain matmul / elementwise / RMSNorm ops, all of
which have proper meta implementations, so dynamo's shape propagation is
correct end-to-end.

## 7. Order to do the work in

```
1. Add MambaSpec handling in get_kv_cache_spec       (B above, ~15 lines)
2. Add MambaSpec branch in initialize_kv_cache       (C above, ~20 lines)
3. Stub TTGatedDeltaNetAttention with naive PyTorch  (A above, ~200 lines)
   — register it in overrides.py
4. Run on a tiny prompt (num_tokens small, 1 layer)
   — verify numerical equivalence against the FLA reference on CUDA
5. Once correct, scale up: full model, prefill+decode, multiple requests
6. Performance: replace naive loop with chunkwise recurrence if needed
```

Steps 1+2 are mechanical and small. Step 3 is the bulk of the work. Steps 4-6
are validation + perf.

## 8. Open questions / known risks

- **Numerical equivalence.** The chunkwise algorithm has subtle ordering of
  matmul + decay that a naive token-by-token loop doesn't replicate exactly in
  bf16 due to accumulation order. Step 4 needs a tolerance, not bit-exact match.
- **`RMSNormGated.forward_native` may itself not trace cleanly** under TT after
  shapes are correct (it has `if residual is not None` branches and uses
  `.data` in places — same family of issues as the original `TTRMSNorm`
  motivation). A separate `TTRMSNormGated` override may be needed.
- **Sharding of the recurrent state.** The state is naturally per-head, and
  `num_v_heads` (48) may not divide cleanly into the mesh's "batch" axis.
  Safe-and-replicated to start; optimize later.
- **Speculative decoding.** The CUDA path has spec_decode-specific branches
  (`mixed_qkv_spec` / `mixed_qkv_non_spec`). For first bringup we should refuse
  speculative decode in `TTPlatform.check_and_update_config` (it's already
  asserted off on line 277 of platform.py, but worth confirming).
- **M-RoPE and the vision encoder are still blockers** for full Qwen3.6-27B
  service. Even with DeltaNet working, text-only requests are the realistic
  scope of the first milestone.

## 9. Files touched, summary

| File | Purpose |
|---|---|
| (new) `integrations/vllm_plugin/vllm_tt/tt_gated_delta_net.py` | The override class. ~200-300 lines. |
| `integrations/vllm_plugin/vllm_tt/overrides.py` | Register the override in `ISINSTANCE_OVERRIDES`. |
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Add `MambaBase` branch in `get_kv_cache_spec` (line ~822). Add `MambaSpec` branch in `initialize_kv_cache` (line ~2159). |
| (maybe) `integrations/vllm_plugin/vllm_tt/platform.py` | Set `mamba_block_size` / `mamba_page_size_padded` in `check_and_update_config`. |
| (maybe) `integrations/vllm_plugin/vllm_tt/metadata.py` | Construct minimal `GDNAttentionMetadata`. |

## 10. Estimated effort

- **B + C** (KV cache + state allocation plumbing): **~half a day** for someone
  who knows the plugin.
- **A** (write `TTGatedDeltaNetAttention` from scratch, naïve loop, numerics
  verified on CUDA reference): **2–4 days**.
- **Integration + first end-to-end run + debugging shape mismatches and state
  lifecycle issues**: **2–4 days**.
- **Total to first working text-only chat on Qwen3.6-27B (still no image
  support, performance unoptimized)**: **roughly 1–2 weeks of focused work**
  by one person.

Vision encoder + M-RoPE adds significantly more on top of this.
