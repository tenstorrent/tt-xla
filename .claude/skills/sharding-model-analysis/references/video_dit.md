Here we walk through how we shard the **Wan 2.2 14B DiT** (`WanTransformer3DModel`) - the dual-expert transformer that runs every denoising step in the Wan 2.2 A14B pipeline. This is the DiT counterpart to the [VAE sharding discussion](https://github.com/tenstorrent/tt-xla/discussions/4664). Implementation lives in `shared.py::shard_dit_specs` and `apply_dit_sp_activation_sharding`.

---

## DiT structure - what we're sharding

```
WanTransformer3DModel  (~14B params per expert, 2 experts in A14B)
├── rope                  WanRotaryPosEmbed  (buffers, no params)
├── patch_embedding       Conv3d(C_in → D, k=(1,2,2), s=(1,2,2))      ← entry, replicated
├── condition_embedder    WanTimeTextImageEmbedding
│   ├── time_embedder     Linear + SiLU + Linear  (→ D)               ← Megatron pair
│   ├── time_proj         Linear(D → 6·D)                             ← replicated
│   └── text_embedder     Linear + GELU + Linear  (→ D)               ← Megatron pair
├── blocks[0..N-1]        WanTransformerBlock × 40                    ← the bulk
│   each block:
│     ├── norm1, norm3    FP32LayerNorm  (non-affine)
│     ├── attn1           WanAttention (self, with RoPE)              ← Megatron pair
│     ├── attn2           WanAttention (cross, K/V from prompt)       ← Megatron pair
│     ├── norm2           FP32LayerNorm  (affine)
│     ├── ffn             FeedForward (D → ffn_dim → D, gelu-approximate) ← Megatron pair
│     └── scale_shift_table  Parameter (1, 6, D)                      ← replicated
├── norm_out              FP32LayerNorm  (non-affine)
├── proj_out              Linear(D → out_channels · prod(patch_size)) ← exit, replicated
└── scale_shift_table     Parameter (1, 2, D)                         ← replicated
```

The model has `num_heads = 40` attention heads and `num_layers = 40` blocks. Per forward, the activation enters the block stack flat as `(B, L, D)` where `L = F'·H'·W'` is the number of patchified video tokens.

---

## Two sharding axes

Following tt-metal's terminology:

| Role | What it shards | Size on (2, 4) mesh | Size on (8, 4) mesh |
|---|---|:---:|:---:|
| **TP** (tensor-parallel) | hidden dim `D` and FFN inner dim - head-aligned | **4** | **4** |
| **SP** (sequence-parallel) | flat token dim `L` of activations | **2** | **8** |

We fix TP=4 on every production mesh because `num_heads = 40 = 10 × 4` divides cleanly and each device ends up with a whole number of attention heads. SP scales with whatever's left over after pinning TP.

`shard_dit_specs` handles **TP via weight annotations**. `apply_dit_sp_activation_sharding` handles **SP via `sharding_constraint` ops on intermediate activations** - since SP shards a dim (L) that has no analog in any parameter.


---

## Why Megatron is right for the DiT

A DiT block is textbook [Megatron-LM](https://arxiv.org/abs/1909.08053): two column→row matmul pairs (attention QKV+O, FFN up+down) with a non-linearity between. Same template the VAE residual block uses, just at transformer scale:

- **Column-parallel** (`to_q/k/v`, `ff1`): split the output dim. The output is head-aligned (40 heads split 4 ways → 10 heads per device), so each device naturally owns a contiguous head group. No CCL.
- **Row-parallel** (`to_out`, `ff2`): split the input dim. Consumes the partitioned activation directly; the matmul output is a partial sum that needs **one all-reduce** to materialize.

Three such pairs per block give us 3 all-reduces per block at minimum (`attn1.to_out`, `attn2.to_out`, `ffn.ff2`). No `all_gather` between the column and row sides - the activation stays sharded across the non-linearity. The norms inside attention (`norm_q/k`) reduce over the full hidden dim, which is sharded, so each one needs a small stats AR.

---

## Block-level sharding strategy

Inside the block, the activation flips between two sharding states depending on whether we're inside or outside a Megatron pair:

```
STATE A - between Megatron pairs   :  TP-replicated D, SP-sharded L
   per-device shape: (B, L/sp, D)

STATE B - inside a Megatron pair   :  TP-sharded D, SP-sharded L
   per-device shape: (B, L/sp, D/tp)
```

The transitions are baked into the matmul shapes:
- **STATE A → STATE B** at every column-parallel projection (`to_q/k/v`, `ff1`) - pure local matmul, no CCL.
- **STATE B → STATE A** at every row-parallel projection (`to_out`, `ff2`) - emits a partial sum, AR materializes the full D.

The block's entry and exit both sit in STATE A, so the 40 blocks chain transparently.

---

## One block, end to end

```
                  spatial : STATE A
                      │
                      ▼
              ┌── norm1 ─────────────────────────────────────────────┐
              │  non-affine LN + adaLN (scale_msa, shift_msa)        │
              │  no params, fully local                              │ 0 CCL
              └──────────────────────────────────────────────────────┘
                      │
              ┌── attn1 (self-attention) ────────────────────────────┐
              │  to_q/k/v   col-parallel  STATE A → STATE B          │
              │  norm_q/k   RMSNorm across full D, γ TP-sharded      │ 2 small stats AR
              │  head split + RoPE apply (local)                     │
              │  AG of K, V along SP axis on L                       │ 2 large SP-AG
              │  SDPA       per-device on local Q × full K, V        │
              │  to_out     row-parallel  STATE B → STATE A          │ 1 large TP-AR
              └──────────────────────────────────────────────────────┘
                      │
                      + gate_msa * attn1_out  (STATE A + STATE A)
                      ▼
              ┌── norm2 ─────────────────────────────────────────────┐
              │  affine LN, γ/β replicated, fully local              │ 0 CCL
              └──────────────────────────────────────────────────────┘
                      │
              ┌── attn2 (cross-attention to prompt) ─────────────────┐
              │  to_q       col-parallel on spatial                  │
              │  to_k/v     col-parallel on prompt (already SP-rep)  │ no K/V AG needed
              │  norm_q/k   stats AR                                 │ 2 small stats AR
              │  SDPA       per-device, no ring                      │
              │  to_out     row-parallel + AR  STATE B → STATE A     │ 1 large TP-AR
              └──────────────────────────────────────────────────────┘
                      │
                      + attn2_out  (STATE A + STATE A)
                      ▼
              ┌── norm3 + adaLN ─────────────────────────────────────┐ 0 CCL
              └──────────────────────────────────────────────────────┘
                      │
              ┌── ffn ───────────────────────────────────────────────┐
              │  net[0].proj  col-parallel  STATE A → inner-sharded  │
              │  GELU         local on inner-sharded                 │
              │  net[2]       row-parallel + AR                      │ 1 large TP-AR
              └──────────────────────────────────────────────────────┘
                      │
                      + c_gate_msa * ffn_out  (STATE A + STATE A)
                      ▼
                  spatial : STATE A
```

Per-block CCL bill at the largest mesh we care about (Galaxy 8×4, sp=8, tp=4):

| Op | Count | Size |
|---|:---:|:---:|
| `attn1` K and V SP-AGs | 2 | big |
| `attn1.to_out` AR | 1 | big |
| `attn2.to_out` AR | 1 | big |
| `ffn.net[2]` AR | 1 | big |
| norm_q/k stats ARs (× 2 attns) | 4 | small |
| **Total** | **5 big + 4 small** | |

On the 1×4 mesh, SP collapses to a no-op so the two K/V SP-AGs disappear - that block costs just **3 big + 4 small** CCLs.

---

## Why cross-attention is "free" on the SP axis

In `attn1` (self-attention), K and V are derived from `spatial` so they share its SP-sharded L axis. Each device only sees its own slice of L tokens, but SDPA needs every Q token to attend against the full K, V - so we **all-gather K and V along SP** before the dot-product.

`attn2` (cross-attention) is different. K and V come from the prompt, which has its own L_prompt axis that's unrelated to the video's L. The prompt is replicated on the SP axis to start with, so `attn2`'s K and V are already SP-replicated and no SP-AG is needed. That's why cross-attention costs half the CCL of self-attention.

---

## Sequence parallelism - sharding the L axis of activations

SP shards the activation's L axis across `sp_factor` devices. Since no weight has an L axis, we can't express SP via `shard_dit_specs` - we annotate intermediate activations directly with `sharding_constraint` ops.

The dataflow at runtime:

```
input (5D, replicated on all axes)
       ↓ patch_embedding (replicated weight) → 5D (B, D, F', H', W') replicated
       ↓ flatten(2).transpose(1, 2)          → 3D (B, L, D) replicated
       ↓  ★ sharding_constraint "replicated"   ★ sharding_constraint "L on SP"   ← introduce SP here
       ↓
       (B, L/sp, D)  ← STATE A
       ↓ block 0
       ↓ ...
       ↓ block 39 → (B, L/sp, D) ← STATE A
       ↓ proj_out (replicated weight)        → (B, L/sp, 64)
       ↓  ★ sharding_constraint "replicated"  ← AG L back to full before unpatchify
       ↓ reshape → permute → flatten ×3
       output (replicated)
```

The interesting part is the **double constraint at the block-stack entry** (and an analogous one inside `rope.forward`). It's there to work around a tt-mlir lowering issue with reshape:

```
What we want:                            What happens with a single constraint:

    5D replicated                            5D replicated
        ↓                                       ↓
    reshape (5D → 3D)                       reshape (5D → 3D)
        ↓                                       ↓
    sharding_constraint("L on SP")          sharding_constraint("L on SP")
        ↓                                       ↑ ← Shardy back-propagates
    block stack ...                              │   the shard INTO the
                                                 │   reshape, partitions one
                                                 │   of (F', H', W'), and
                                                 │   tt-mlir fails to update
                                                 │   the reshape's static
                                                 │   result type to per-device
                                                 │   shape → "number of elements
                                                 │   doesn't match" error
```

The fix is two constraints back to back, both downstream of the reshape:

```
    reshape (5D → 3D)
        ↓
    sharding_constraint(replicated)   ← terminates back-propagation here
        ↓
    sharding_constraint(L on SP)      ← shard requested between two non-reshape ops
        ↓                                  → Shardy inserts a clean scatter
    block stack ...                         (reshape is never partitioned)
```

We apply this pattern in two places: at the block-stack entry (after `flatten+transpose`) and on the rope output (rope's body ends in a similar reshape). The exit at `proj_out` only needs a **single** constraint because the direction is reversed there - we're going sharded → replicated *upstream* of the unpatchify reshape, which doesn't trigger the bug; the all-gather lands cleanly before the reshape.


---

## Full DiT layout - what's sharded where

```
WanTransformer3DModel
├── rope                  buffers, replicated; output marked SP-on-L via forward hook
├── patch_embedding       weight replicated, output stays replicated
├── condition_embedder
│   ├── time_embedder     Megatron col→row pair (TP on D)
│   ├── time_proj         replicated  (lone col-par would cost an AG every block)
│   └── text_embedder     Megatron col→row pair (TP on D)
├── blocks[0..39]
│   ├── norm1, norm3      no params (non-affine LN)
│   ├── norm2.weight/bias replicated  (matches STATE A)
│   ├── attn1, attn2
│   │   ├── to_q/k/v      ("tp", None)        col-parallel on D
│   │   ├── norm_q/k      ("tp",)             gamma matches sharded act
│   │   └── to_out[0]     (None, "tp")        row-parallel + AR → STATE A
│   ├── ffn
│   │   ├── net[0].proj   ("tp", None)        col-parallel on inner
│   │   └── net[2]        (None, "tp")        row-parallel + AR
│   └── scale_shift_table replicated
├── norm_out              no params
├── proj_out              replicated  (one explicit AG hook to force replicated output)
└── scale_shift_table     replicated  (final-layer adaLN)
```
