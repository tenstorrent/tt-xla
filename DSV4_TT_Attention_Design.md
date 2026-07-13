# Design: DeepSeek V4 Attention on the `tt` Platform (tt-xla `vllm_plugin`)

**Status:** Draft for review
**Scope:** Register out-of-tree (OOT) attention backends + supporting infrastructure in
`integrations/vllm_plugin/vllm_tt` so DeepSeek V4 (DSV4) attention runs on Tenstorrent
hardware through the `tt` PJRT/`torch_xla` path.
**Author:** (design proposal)

---

## 0. TL;DR

DSV4 attention is not "one more MLA variant." Relative to the DeepSeek V3-style MLA that
`TTMLAAttentionBackendImpl` already handles, DSV4 adds four things that the current plugin
has no representation for:

1. **A lightning-indexer sparse branch** — per query token attends only a top-k subset of
   compressed KV slots (`compress_ratio == 4`, "CSA"), or a contiguous compressed prefix
   (`compress_ratio == 128`, "C128A" / HCA).
2. **A sliding-window (SWA) branch** stored in a *separate* per-token cache, run in parallel
   with the compressed branch and **merged via online softmax**.
3. **Multiple co-resident KV caches per layer** (compressed latent, SWA per-token, indexer
   k-cache) with heterogeneous specs.
4. **Attention sinks** folded into the softmax denominator.

The good news, established by tracing tt-metal, is that **most of the required kernels already
exist in ttnn**:

- `ttnn::transformer::paged_flash_multi_latent_attention_decode` already accepts
  `attention_sink` and `sliding_window_size`
  (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp`).
- `ttnn::transformer::sparse_sdpa` is explicitly documented as *"Sparse MLA prefill (DeepSeek
  DSA)"*, taking a `[1,1,S,TOPK] uint32` index tensor with `0xFFFFFFFF` sentinel masking, and
  even supports block-cyclic SP-sharded caches for chunked prefill
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/sparse_sdpa.hpp`).
- SDPA prefill supports `sliding_window_size`, `attention_sink`, and `chunk_start_idx`.
- `ttnn::topk` exists for computing indexer top-k.

The work is therefore concentrated in three layers, in decreasing order of how much already
exists:

- **tt-metal (ttnn):** mostly present; the main gap is a **sparse/indexed *decode*** op
  (`sparse_sdpa` is prefill-only — no `cur_pos_tensor`) and, if we choose kernel-side merge,
  LSE outputs from the MLA decode op.
- **tt-mlir:** plumb new `stablehlo.custom_call` targets → TTIR → TTNN, and unblock the two
  parameters the runtime executor currently hardcodes (`slidingWindowSize = std::nullopt`).
- **tt-xla `vllm_plugin`:** the bulk of *this* document — new custom ops, a DSV4 MLA impl, a
  multi-cache-group KV spec, per-group attention metadata, and the OOT registration wiring.

> **Verification note.** All file/line references below were read directly from the four repos
> at their current `main`. Two things I did **not** fully verify and flag as assumptions:
> (a) whether the pinned `vllm==0.20.2` (`requirements-vllm-plugin.txt:1`) already ships the
> `vllm.model_executor.models.deepseek_v4` module and its `DeepseekV4Attention` /
> `DeepseekV4SWACache` / compressor / indexer classes — the design assumes it does, since
> `TTMLAAttentionBackendImpl.__init__` already accepts an `indexer=` kwarg
> (`attention_mla.py`), which is a strong signal the indexer-aware MLA plumbing is present;
> (b) the exact numeric behavior of `sparse_sdpa`'s sentinel/gather path on multi-chip meshes.
> Both should be pinned down before implementation starts.

---

## 1. Current state of the plugin (what we build on)

### 1.1 Registration surface

`vllm_tt/__init__.py` is the entry point. It does three distinct kinds of registration:

- **Attention backends** via `register_backend(...)` against `AttentionBackendEnum.CUSTOM`
  (dense) and `AttentionBackendEnum.FLASH_ATTN_MLA` (MLA).
- **Layer symbol substitution** in `register_oot_layers()` — monkeypatches
  `vllm...attention.Attention → TTAttention`, and imports modules whose side effect is
  `@register_oot` registration (`attention_mla`, `TTFusedMoE`).
- **`register()`** returns the platform class path `"vllm_tt.platform.TTPlatform"`.

The MLA backend itself is chosen in `TTPlatform.get_attn_backend_cls`
(`platform.py`):

```python
if attn_selector_config.use_sparse:
    raise NotImplementedError("Sparse Attention is not supported on TT devices.")
if attn_selector_config.use_mla:
    return AttentionBackendEnum.FLASH_ATTN_MLA.get_path()
```

**This is the first hard blocker: DSV4 sets `use_sparse=True`, so it hits the `raise` before
it ever reaches the MLA branch.**

### 1.2 Existing MLA path

`vllm_tt/attention_impls/attention_mla.py`:

- `TTMLAAttentionBackend` — declares `get_kv_cache_shape` as a **single** dense latent cache
  `(num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim)`, `num_kv_heads == 1`.
- `TTMLAAttentionBackendImpl(MLAAttentionImpl)` — does Q-absorption (`q_nope @ W_UK_T`),
  concatenates latent Q/K, and dispatches:
  - **prefill** → `torch.ops.tt.flash_mla_prefill` + `torch.ops.tt.paged_fill_cache`
  - **decode** → `torch.ops.tt.paged_update_cache` + `torch.ops.tt.paged_flash_mla_decode`
  - It **rejects** `kv_cache_dtype != "auto"` (no fp8 KV) and accepts but ignores `indexer=`.
- `TTMLAAttention(MLAAttention)` — overrides `forward` to call `impl.forward(...)` directly.
- `TTMultiHeadLatentAttentionWrapper` — `@MultiHeadLatentAttentionWrapper.register_oot`; swaps
  `MLAAttention → TTMLAAttention` during construction and reshapes 3D↔2D I/O.

### 1.3 Custom-op mechanism

`python_package/tt_torch/custom_ops.py` defines each `torch.ops.tt.*` op with a uniform
three-path pattern:

- `device.type == "xla"` → `stablehlo_custom_call.stablehlo_custom_call(inputs, "tt.<name>",
  [out_shape], [out_dtype], frontend_attributes={...})` — this is what tt-mlir consumes.
- `device.type == "cpu"` → an eager reference (usually `F.scaled_dot_product_attention` /
  index_put), used for tracing and CPU-only tests.
- `@<op>.register_fake` → shape/dtype meta for Dynamo tracing.

Notably, `paged_flash_mla_decode` **already declares** optional `attention_sink` and
`cur_pos_tensor` operands and forwards `has_attention_sink` as a frontend attribute — with an
in-file `TODO(@hshah)` that the CPU path doesn't model the sink yet.

### 1.4 Compiler lowering pipeline

For every `tt.<name>` there is:

1. A dedicated `OpConversionPattern<stablehlo::CustomCallOp>` in
   `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` keyed on
   `funcName == "tt.<name>"` (see the dispatch table: `tt.flash_mla_prefill`,
   `tt.paged_flash_mla_decode`, `tt.paged_fill_cache`, `tt.paged_update_cache`,
   `tt.sparse_matmul`, `tt.all_to_all_dispatch`, …).
2. A TTIR op (`include/ttmlir/Dialect/TTIR/IR/TTIROps.td`) and TTNN op
   (`TTNNOps.td`).
3. A runtime executor under `tt-mlir/runtime/lib/ttnn/operations/transformer/` that calls the
   real ttnn op — e.g. `paged_flash_multi_latent_attention_decode.cpp` currently builds an
   `SDPAProgramConfig`, forwards `attentionSink`, and **hardcodes
   `slidingWindowSize = std::nullopt`.**

### 1.5 KV cache spec + attention metadata

- `TTModelRunner.get_kv_cache_spec()` (`model_runner.py:1021`) emits **one** `KVCacheSpec` per
  layer. For `MLAAttention` it always builds a single `MLAAttentionSpec(num_kv_heads=1,
  head_size=attn_module.head_size, ...)`. It has a `SlidingWindowSpec` branch, but only for
  classic (non-MLA) `Attention`.
- The attention metadata is a single `TTMetadata` dataclass (`attention.py:177`) carrying
  `cache_position`, `page_table`, `fill_page_table`, `attn_mask`, `is_causal`,
  `chunk_start_idx`. Crucially, `_prepare_inputs` builds **one** `TTMetadata` and fans it out to
  every layer: `per_layer_attn_metadata = dict.fromkeys(self._attention_layer_names,
  attn_metadata)` (`model_runner.py:1521`, and again at 2323/2723).

**This single-shared-metadata design is the second structural blocker.** DSV4's SWA and
compressed branches have *different* sequence lengths (`kv_len` vs `kv_len // compress_ratio`),
different block sizes, and different page tables. They cannot share one `TTMetadata`.

---

## 2. What DSV4 requires (recap grounded in the model)

From the upstream GPU/TPU implementations (`vllm/models/deepseek_v4/`,
`tpu-inference/.../experimental/deepseek_v4/`), each DSV4 attention layer is typed by
`compress_ratio`:

| Layer type | `compress_ratio` | Compressed branch | SWA branch |
|---|---|---|---|
| SWA-only | `<= 1` | none | full window, normalized alone |
| CSA / C4A | `4` | lightning-indexer **top-k** slots | window, merged |
| HCA / C128A | `128` | contiguous compressed **prefix** `(pos+1)//ratio` | window, merged |

Per token the flow is: one latent (NoPE + RoPE) is (a) written raw into the SWA per-token cache
and (b) compressed `ratio:1` into the compressed cache. Attention runs both branches and
merges them with an online-softmax stitch (rescale each branch by `exp(m_branch − m_global)`,
sum, normalize once). Attention sinks enter as an extra per-head logit.

We must reproduce this on TT using ttnn primitives, respecting XLA's functional (return-not-
mutate) cache-update model and DYNAMO_TRACE_ONCE static shapes.

---

## 3. Proposed architecture

### 3.1 Component overview

```
vllm_tt/
  __init__.py                         # + register DSV4 backend, import dsv4 module
  platform.py                         # + allow use_sparse for DSV4; DSV4 cache/scheduler cfg
  attention_impls/
    attention_mla.py                  # unchanged (V3-style MLA)
    attention_dsv4.py        [NEW]    # TTDSV4* backend, impl, layer, OOT wrappers
  metadata.py / attention.py          # + TTDSV4Metadata (per-branch page tables/seqlens)
  model_runner.py                     # + multi-group KV spec, per-group metadata build
python_package/tt_torch/
  custom_ops.py                       # + tt.sparse_mla_prefill, tt.paged_sparse_mla_decode,
                                      #   tt.mla_swa_decode, tt.lightning_indexer_topk,
                                      #   tt.kv_compress (as needed)
```

Compiler-side (separate PRs, tracked as dependencies):

```
tt-mlir/
  lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp   # + new custom-call patterns
  include/ttmlir/Dialect/{TTIR,TTNN}/IR/*.td                   # + new op defs
  runtime/lib/ttnn/operations/transformer/*.cpp               # + executors; unblock SWA size
tt-metal/
  ttnn/.../sdpa_decode/*                                       # + sparse/indexed decode op
```

### 3.2 Design decision: cache layout — overlay vs. separate tensors

The GPU packs all DSV4 caches into one byte buffer via `block_stride`/`offset` and reads each
layer as a typed view. The TPU backend overlays them onto one `uint8` page pool with equal
per-page byte footprints. **Both rely on primitives TT does not have** (GPU: byte-level type
punning; TPU: raw `uint8` reinterpret inside a Pallas kernel).

**Recommendation: do NOT emulate the overlay initially.** Allocate the compressed cache and the
SWA cache as *separate* `KVCacheTensor`s / cache groups. Rationale:

- TT tensors are strongly typed; the `load_bkv` NaN-masking hack the TPU backend needs exists
  precisely because it reinterprets bf16/fp8 bytes in a shared pool. Separate typed caches make
  that class of bug impossible.
- vLLM's hybrid KV-cache allocator already supports multiple groups per model; we lean on it
  rather than reimplementing page-size equalization.
- It costs some HBM (no page sharing between SWA and compressed), which is acceptable for a
  first correct implementation. Overlay can be a later memory optimization once the
  page-ownership semantics are proven (and only if HBM pressure demands it).

This means DSV4 becomes vLLM's first **hybrid / multi-group** model on TT — see §3.5.

### 3.3 New custom ops (`custom_ops.py`)

All follow the existing xla/cpu/fake three-path pattern. Frontend attributes stringify scalars
and `has_*` flags exactly as `paged_flash_mla_decode` does.

**(a) `tt::mla_swa_decode`** — thin wrapper we may not even need as a new op: the SWA decode
branch maps directly onto the *existing* `tt.paged_flash_mla_decode` once tt-mlir stops
hardcoding `slidingWindowSize`. Preferred plan: **add a `sliding_window` int arg to
`paged_flash_mla_decode`** (default 0 = disabled) and forward it as a frontend attribute,
rather than a brand-new op. This reuses the sink support already there.

**(b) `tt::sparse_mla_prefill`** — wraps `ttnn::transformer::sparse_sdpa`.

```
def sparse_mla_prefill(query, kv, indices, head_dim_v, scale=None) -> Tensor
  # query:   [1, H, S, K_DIM]      (K_DIM = kv_lora_rank + qk_rope_head_dim, e.g. 576)
  # kv:      [1, 1, T, K_DIM]      compressed latent cache, contiguous per user
  # indices: [1, 1, S, TOPK] int32/uint32, 0xFFFFFFFF sentinel tail = masked
  # -> [1, H, S, head_dim_v]
```

CPU reference: gather `kv[indices]`, mask sentinels to `-inf`, run `F.sdpa`. This is the
lightning-indexer / CSA prefill branch. `sparse_sdpa` already documents the exact
`indices`/sentinel contract, so the tt-mlir pattern is mostly attribute plumbing.

**(c) `tt::paged_sparse_mla_decode`** — the genuinely missing kernel (see §4). Signature mirrors
`paged_flash_mla_decode` but adds a per-user `indices` operand (top-k compressed page slots)
and drops the causal-prefix assumption:

```
def paged_sparse_mla_decode(query, key, head_dim_v, page_table, indices,
                            cur_pos_tensor, attention_sink=None, scale=None) -> Tensor
  # query:   [1, num_users, H, K_DIM]
  # key:     paged compressed cache [num_blocks, 1, block_size, K_DIM]
  # indices: [num_users, TOPK] int32 compressed-slot ids (sentinel = -1)
```

**(d) `tt::lightning_indexer_topk`** — optional. The indexer is a small
`ReplicatedLinear` + RoPE + score matmul + `topk`. It can be expressed as plain
torch/StableHLO ops (matmul + `torch.ops.tt` topk if one exists, else `ttnn::topk` via a new
`tt.topk` custom call). Prefer **composition in StableHLO** over a monolithic custom op so the
compiler can fuse/shard it; only fall back to a custom op if the top-k selection needs a fused
kernel for performance.

**(e) `tt::kv_compress`** — the `ratio:1` compression is a strided
`ReplicatedLinear`/matmul over grouped tokens; express as StableHLO matmul + reshape, not a new
op. The write into the compressed cache reuses `tt.paged_fill_cache` /
`tt.paged_update_cache`.

> Net new *kernels* (C++ in tt-metal): **one** — the paged sparse MLA decode. Everything else
> is reuse (existing ttnn ops) or StableHLO composition.

### 3.4 `attention_dsv4.py` — the impl

Modeled on `attention_mla.py` but branch-aware. Key classes:

- **`TTDSV4AttentionBackend(AttentionBackend)`** — but note DSV4 needs *two* cache shapes.
  Because a vLLM `AttentionBackend.get_kv_cache_shape` returns one shape, we expose the
  compressed-cache shape here and register the SWA cache as a **separate backend/spec** (see
  §3.5). Alternatively, follow upstream's pattern where `DeepseekV4SWACache` is its own
  `AttentionLayerBase` with its own backend — this is the cleaner mirror and what we recommend.

- **`TTDSV4MLAAttentionBackendImpl(MLAAttentionImpl)`** — `__init__` captures
  `compress_ratio`, `sliding_window`, `indexer`, `kv_lora_rank`, `qk_rope_head_dim`,
  `v_head_dim`, and the attention-sink parameter. `forward` dispatches by
  `(_infer_is_prefill, compress_ratio)`:

  ```
  forward():
    q_nope, q_pe = q
    build latent q_lat = [q_nope @ W_UK_T ; q_pe]
    write branches:
      - SWA cache  <- tt.paged_fill_cache / paged_update_cache (per-token latent)
      - compressed <- kv_compress(...) then paged_fill_cache / paged_update_cache
    if compress_ratio <= 1:            # SWA-only
        out = swa_branch(...); normalize alone
    else:
        if is_prefill:
            idx = indexer_topk(...) if ratio==4 else prefix_indices(pos, ratio)
            comp = tt.sparse_mla_prefill(q_lat, compressed_cache_gathered, idx, ...)
            swa  = tt.flash_mla_prefill(q_lat, swa_k, sliding_window=..., sink=...)
            out  = merge(comp, swa)          # see §3.6
        else:  # decode
            idx  = indexer_topk_decode(...) if ratio==4 else prefix_indices
            comp = tt.paged_sparse_mla_decode(q_lat, compressed_cache, idx, sink=...)
            swa  = tt.paged_flash_mla_decode(q_lat, swa_cache, sliding_window=..., sink=...)
            out  = merge(comp, swa)
    out = out @ W_UV                    # expand latent -> v_head_dim (existing pattern)
  ```

- **`TTDSV4Attention` / OOT wrappers** — mirror `TTMLAAttention` +
  `TTMultiHeadLatentAttentionWrapper`. Because DSV4 uses distinct upstream classes
  (`DeepseekV4Attention`, `DeepseekV4SWACache`, the compressor, the indexer), we intercept them
  the same way the existing code swaps `MLAAttention`: monkeypatch inside the OOT wrapper's
  `__init__`, or register `@<UpstreamClass>.register_oot` where upstream exposes such a hook.
  (Upstream GPU dispatches DSV4 by faking `is_rocm=True`; we instead select via
  `get_attn_backend_cls` + class substitution — do **not** replicate the ROCm hack.)

### 3.5 Multi-group KV cache (`model_runner.py`)

Extend `get_kv_cache_spec()` so a DSV4 layer emits **two** specs under distinct layer keys:

```python
elif isinstance(attn_module, MLAAttention) and is_dsv4(attn_module):
    # compressed latent branch (skip for SWA-only layers)
    if attn_module.compress_ratio > 1:
        kv_cache_spec[compressed_key] = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=attn_module.head_size,          # kv_lora_rank + qk_rope
            dtype=self.kv_cache_spec_dtype,
            cache_dtype_str=cache_dtype_str,
            # storage_block_size = block_size // compress_ratio  (see note)
        )
    # SWA per-token branch
    kv_cache_spec[swa_key] = SlidingWindowSpec(
        block_size=block_size,                        # or a smaller SWA block
        num_kv_heads=1,
        head_size=attn_module.head_size,
        dtype=self.kv_cache_spec_dtype,
        sliding_window=attn_module.sliding_window,
    )
```

Two consequences to handle:

1. **`storage_block_size != block_size`.** The compressed cache stores one slot per
   `compress_ratio` tokens. vLLM's `MLAAttentionSpec` supports a `storage_block_size` /
   `compress_ratio` distinction (used by the GPU path). Verify `vllm==0.20.2` exposes it; if
   not, we bump the vLLM pin or carry a small shim spec.
2. **Per-group page tables.** The single-shared `dict.fromkeys(...)` metadata fan-out
   (`model_runner.py:1521/2323/2723`) must become **per-group**. Introduce a
   `TTDSV4Metadata` with `swa_page_table` / `swa_cache_position` /
   `compressed_page_table` / `compressed_seq_lens` / `topk_indices`, and build the
   per-layer dict from the group each layer belongs to (vLLM already groups layers; use
   `KVCacheGroupSpec.layer_names`).

Also update `AscendScheduler` (the TT default scheduler, set in
`platform.check_and_update_config`) only if it makes MLA-specific block-accounting assumptions;
the SWA branch's block reclamation outside the window is handled by vLLM's `SlidingWindowSpec`
manager, same as GPU.

### 3.6 The two-branch merge — the key open decision

The MLA decode op returns a **single `Tensor`, no LSE** (`sdpa_decode.hpp`). So we cannot
naively do the GPU/TPU online-softmax stitch across two separate kernel calls unless one side
exposes running stats. Three options, in order of preference:

**Option A — single combined sparse call (mirror GPU prefill).** Gather SWA + compressed KV
into one contiguous latent workspace, build one combined index list (window indices ∪ top-k
slot indices) per query, and call **one** `sparse_sdpa` / `paged_sparse_mla_decode`. One
softmax, one kernel, no merge math. This is exactly what the GPU prefill path does
(`dequantize_and_gather_k_cache` + `combine_topk_swa_indices` + one `flash_mla_sparse_fwd`).
- Pro: numerically clean, no LSE needed, best matches an existing, proven design.
- Con: needs a gather step (extra HBM traffic) and, for decode, the missing sparse decode op
  must accept a combined index list spanning two logical caches — feasible if both branches are
  gathered into one buffer first.

**Option B — expose LSE and merge in StableHLO.** Add an optional `return_lse` to the MLA
decode / sparse ops; do the `exp(m−m_global)` rescale-and-sum as plain StableHLO ops in the
Python impl. Most flexible, keeps caches separate, but requires kernel changes on **both**
branches and careful numerics.

**Option C — `joint_scaled_dot_product_attention`.** ttnn already has a `joint_*` SDPA that
returns `(out, lse)` and concatenates a "joint" sequence. Investigate whether its joint-sequence
semantics can express "compressed slots + window tokens as one logical K/V." If so, it may give
Option A's single-softmax property without a manual gather.

**Recommendation:** start with **Option A for prefill** (directly reuses `sparse_sdpa`, which
was built for exactly this), and **Option A for decode** gated on the new
`paged_sparse_mla_decode` accepting a combined index list. Fall back to Option B only if the
gather cost is prohibitive. Prototype both branches' correctness against the CPU reference paths
before committing to the merge strategy.

### 3.7 Platform gating (`platform.py`)

- Replace the blanket `use_sparse → raise` in `get_attn_backend_cls` with a DSV4-aware branch:
  return the new DSV4 backend path when the model is DSV4 (detect via
  `model_config.hf_config` architecture / presence of `compress_ratios`), keep raising for
  other sparse models.
- In `check_and_update_config`, DSV4 needs the same MLA guards already present (force
  `enable_chunked_prefill=False` / prefix caching off for the first version — chunked prefill
  with sparse gather is a follow-up, even though `sparse_sdpa` *does* support block-cyclic
  chunked caches). Set `cache_config.block_size` consistent with `compress_ratio` divisibility
  (block_size % compress_ratio == 0, and the existing `% 8 == 0` page-table alignment).
- Wire `additional_config` knobs: `dsv4_topk` (indexer k), and reuse
  `experimental_kv_cache_dtype` — but note the current MLA impl rejects non-`auto` KV dtype;
  DSV4's fp8 latent format is a **later** optimization, so v1 uses bf16 latent caches.

---

## 4. tt-metal / tt-mlir dependencies (separate PRs)

| Need | Status in ttnn | Action |
|---|---|---|
| SWA decode + sink | **Exists** (`paged_flash_multi_latent_attention_decode`, `sliding_window_size`, `attention_sink`) | Unblock in tt-mlir runtime executor (stop hardcoding `slidingWindowSize=std::nullopt`); add `sliding_window` frontend attr to `tt.paged_flash_mla_decode` |
| Sparse MLA prefill | **Exists** (`sparse_sdpa`, "DeepSeek DSA") | New `tt.sparse_mla_prefill` custom call + TTIR/TTNN op + executor |
| SWA prefill + sink + chunk | **Exists** (SDPA prefill args) | Reuse via `tt.flash_mla_prefill` extended with `sliding_window`/`sink` attrs |
| Sparse MLA **decode** | **Missing** (`sparse_sdpa` has no `cur_pos_tensor`) | **New ttnn kernel** (paged, indexed, single-token) — the one real kernel deliverable |
| top-k | `ttnn::topk` exists | Expose as `tt.topk` custom call, or compose |
| LSE outputs (only if Option B) | Not on MLA decode; `joint_*` returns lse | Add `return_lse` if we choose Option B |

For each new `tt.<name>`: add the `OpConversionPattern` in `StableHLOToTTIRPatterns.cpp`
(clone the `tt.paged_flash_mla_decode` pattern at line ~8697), the TTIR/TTNN op tds, and the
runtime executor under `runtime/lib/ttnn/operations/transformer/`.

---

## 5. Phased delivery plan

**Phase 0 — unblock + SWA-only (smallest end-to-end slice).**
- `platform.py`: DSV4-aware `get_attn_backend_cls`.
- tt-mlir: forward `sliding_window` through `tt.paged_flash_mla_decode` executor.
- Plugin: `attention_dsv4.py` handling **only** `compress_ratio <= 1` layers (SWA-only,
  single normalized branch, sink). Multi-group KV spec for SWA. Validate a DSV4 config with all
  layers forced SWA-only (or a reduced-layer debug model via `num_hidden_layers`).

**Phase 1 — HCA / C128A (contiguous prefix, no indexer).**
- Add compressed cache group + `kv_compress` (StableHLO).
- Prefix indices `(pos+1)//ratio` — no top-k needed.
- Merge via Option A prefill; add `tt.sparse_mla_prefill` (reuses `sparse_sdpa`).
- Decode still needs the merge; if `paged_sparse_mla_decode` isn't ready, gate C128A decode on
  a temporary dense-prefix fallback (attend the whole compressed prefix via the existing paged
  MLA decode with a causal mask) to unblock correctness before the sparse decode kernel lands.

**Phase 2 — CSA / C4A (lightning indexer top-k).**
- Indexer (`ReplicatedLinear` + RoPE + score matmul + `topk`) as StableHLO/composed op.
- `paged_sparse_mla_decode` ttnn kernel (the main C++ deliverable) + full decode merge.

**Phase 3 — optimizations.**
- fp8 latent cache format (mirror the 576/640-byte DS-MLA layout) once bf16 is correct.
- Cache overlay (page sharing) if HBM-bound.
- Chunked prefill via `sparse_sdpa`'s block-cyclic SP path.

---

## 6. Testing strategy

- **CPU reference parity.** Every new `tt.*` op ships a `device.type == "cpu"` reference (as all
  existing ops do). Add op-by-op tests under `tests/` comparing xla vs cpu for
  `sparse_mla_prefill`, `paged_sparse_mla_decode`, and the SWA-with-sink path, following the
  existing `tests/op_by_op` structure.
- **Numerical golden vs vLLM GPU/CPU.** Run the same DSV4 config through stock vLLM CPU (the
  `deepseek_v4` reference) and compare logits per layer type (SWA-only, C4A, C128A) at small
  shapes. The merge (§3.6) is the highest-risk numeric area — test it in isolation with a
  hand-built two-branch case.
- **Reduced-layer bring-up.** Use `TTConfig.num_hidden_layers` to compile a 2–4 layer DSV4 for
  fast iteration; force per-layer `compress_ratio` to exercise each branch.
- **Shape/recompile discipline.** DYNAMO_TRACE_ONCE means every dynamic top-k length or
  variable window must be padded to a compiled bucket. Assert static shapes for `indices`
  (`[users, TOPK]`, sentinel-padded) exactly as the SWA/prefill token ladders are padded today.

---

## 7. Risks & open questions

1. **`vllm==0.20.2` DSV4 availability** (§0 note). If the pin predates `deepseek_v4`, we either
   bump the pin or register the model OOT. Must confirm first — it gates everything.
2. **The merge / no-LSE constraint** (§3.6). Biggest technical unknown. De-risk by prototyping
   Option A end-to-end early.
3. **`paged_sparse_mla_decode` kernel effort.** The single real new kernel; scope it with the
   tt-metal team before committing Phase 2 dates.
4. **`storage_block_size` support in the pinned vLLM allocator** (§3.5). Determines whether the
   compressed cache can pack `block_size/ratio` slots per page cleanly.
5. **Per-group metadata refactor blast radius.** Changing the `dict.fromkeys` fan-out touches
   three call sites and the dummy/profile-run paths (`model_runner.py:2310`, `2711`). Keep the
   single-group path intact for non-DSV4 models to avoid regressions.
6. **Sink numerics.** `paged_flash_mla_decode`'s CPU reference does *not* model the sink yet
   (`TODO(@hshah)`); fix that reference as part of Phase 0 so parity tests are meaningful.

---

## 8. Summary of concrete changes

**tt-xla `vllm_plugin`:**
- `platform.py` — DSV4 branch in `get_attn_backend_cls`; DSV4 config guards.
- `__init__.py` — register DSV4 backend; import `attention_dsv4`.
- `attention_impls/attention_dsv4.py` **[new]** — backend, `MLAAttentionImpl`, layer, OOT
  wrappers; branch dispatch + merge.
- `attention.py` / `metadata.py` — `TTDSV4Metadata` (per-branch page tables/seqlens/topk).
- `model_runner.py` — multi-group `get_kv_cache_spec`; per-group metadata build (replace
  `dict.fromkeys` fan-out).

**tt-xla `custom_ops.py`:**
- Extend `tt.paged_flash_mla_decode` / `tt.flash_mla_prefill` with `sliding_window` (+ ensure
  sink on both, fix CPU sink ref).
- New: `tt.sparse_mla_prefill`, `tt.paged_sparse_mla_decode`, (optional) `tt.topk`.

**tt-mlir:**
- New `OpConversionPattern`s + TTIR/TTNN op defs + runtime executors for the new ops.
- Unblock `slidingWindowSize` in `paged_flash_multi_latent_attention_decode.cpp`.

**tt-metal:**
- One new kernel: paged sparse (indexed, single-token) MLA decode.

The load-bearing insight is that ttnn already carries DSV4-shaped primitives (`sparse_sdpa`
"DeepSeek DSA", SWA + sink in the paged MLA decode). The plugin work is mostly *wiring and
representation* — multi-group caches, per-branch metadata, and the two-branch merge — not new
math, with a single genuinely new kernel (sparse decode) as the critical-path dependency.
