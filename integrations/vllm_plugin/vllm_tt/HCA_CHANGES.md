# DeepSeek v4 High-Compression Attention (HCA) on TT — Change Log & Limitations

This document tracks the high-level changes made to bring DeepSeek v4 attention
(specifically **HCA = High-Compression Attention**) to Tenstorrent hardware via
the vLLM plugin, plus the current known limitations. **Keep this file updated
whenever the HCA implementation changes.**

## Context

- **HCA = High-Compression Attention** = DeepSeek v4's `compress_ratio == 128`
  (C128A) attention layers: sliding-window (local) attention **+** a 128:1
  compressed global KV pool, **no** sparse indexer.
- DeepSeek v4 layer types, keyed by `config.compress_ratios[layer_id]`:
  - `1`   → sliding-window-only (SWA), no compression.
  - `4`   → C4A: SWA + learned sparse indexer (top-k) + 4:1 compression.
  - `128` → **C128A (HCA)**: SWA + 128:1 compression, no indexer.
- Reference: vLLM 0.20.1 `vllm/model_executor/layers/deepseek_v4_attention.py`,
  `deepseek_compressor.py`, `v1/attention/backends/mla/sparse_swa.py`. We mirror
  vLLM's structure, **not** the `tt_forge_models` implementation.

## Files

- `attention_hca.py` — HCA backend, impl, and layer wiring (all here).
- `HCA_CHANGES.md` — this file.
- `vllm_tt/__init__.py` — `register_hca_oot_layer()`.
- `setup.py` — `tt_hca` general-plugins entry point.

## Components implemented

### 1. Backend + impl (`TTHCAAttentionBackend`, `TTHCAAttentionBackendImpl`)
- Latent attention via TT custom ops:
  - prefill → `torch.ops.tt.flash_mla_prefill` + `torch.ops.tt.paged_fill_cache`
  - decode  → `torch.ops.tt.paged_flash_mla_decode` + `torch.ops.tt.paged_update_cache`
- Non-absorbed MLA-from-latent: the query's nope part already lives in the
  compressed latent (no `W_UK_T`/`W_UV`); V is read from the latent K
  (`value=None`).
- KV-cache layout per the v4 fusion: a single latent head of width `head_dim`
  with rope embedded in the trailing `qk_rope_head_dim` dims →
  `[num_blocks, 1, block_size, head_dim]`. (`kv_lora_rank == v_head_dim ==
  head_dim` in vLLM v4.)
- Attention sink (`attn_sink`) forwarded to the decode op.

### 2. Layer wiring
- `TTHCAAttention(MLAAttention)` — inner cache-owning sub-layer. Subclasses
  `MLAAttention` so the TT model runner allocates + binds a single latent cache
  with no model_runner changes. Custom `__init__` (no `kv_b_proj`,
  no `get_attn_backend`); `forward(q, kv, output)` → impl.
- `TTDeepseekV4MLAAttention(DeepseekV4MultiHeadLatentAttentionWrapper)` —
  OOT-registered outer wrapper (via `PluggableLayer.register_oot`, dispatched by
  class name). Clean bf16 forward reusing the model's projection modules:
  `fused_wqa_wkv → split → q_norm/kv_norm → wq_b → per-head q-norm →
  RoPE(q,kv) → HCA attention → inverse-RoPE(o) → grouped wo_a → wo_b`.
  Reuses vLLM's `rotary_emb.forward_native` (trailing-dim GPT-J rope, with
  `inverse=True` for the output).
- Registration: `register_hca_oot_layer()` in `__init__.py`, `tt_hca` entry
  point in `setup.py`.

### 3. Dual cache (C128A) — LIVE (runner-paged, bf16)
C128A layers now run the real high-compression attention end-to-end. **No model
runner changes were needed** — the trick is that all HCA caches are registered as
identical-spec `MLAAttention` sub-layers, so the existing runner allocates/binds
them and they share group 0's block table.

- **`TTHCAPagedCache(MLAAttention)`** — a bare single-latent paged cache used for
  the compressed pool + the two compressor state caches. Same `MLAAttentionSpec`
  as the window cache ⇒ same KV-cache group ⇒ **shared block table**, so the one
  `TTMetadata.page_table` / `cache_position` addresses every cache.
- **`TTHCAAttention`** (window cache owner) also owns `compressed`, `cstate_kv`,
  `cstate_score` when `compress_ratio == 128`, and exposes `forward_dual`:
  - **prefill** — exact full attention over the window cache (`flash_mla_prefill`),
    then `_prefill_populate_compression` writes per-token compressor states and
    the compressed prompt windows (paged_fill) for later decode.
  - **decode** — writes the new token to the window + state caches; pools the
    current (possibly partial) window from the state caches (`_masked_gated_pool`),
    RMSNorm+RoPE, and writes it to compressed slot `pos//ratio`; then the joint
    softmax over (windowed window cache ∪ compressed pool) + sink
    (`forward_dual_decode`). Compressed length / boundaries are derived from
    `cache_position` (`(pos+1)//ratio`), so **generated tokens are compressed
    incrementally** with no per-step runner metadata.
- **`TTHCACompressor`** — `project()` (per-token kv/score), `norm_rope()`, and
  `compress()` (prefill window pooling). Built by the wrapper at
  `compress_ratio == 128` (weights load under `<wrapper>.compressor.*`).
- **`hca_dual_attention`** — the single joint softmax over `[compressed ∪ window]`
  with the per-head sink (extra softmax column); MLA-from-latent (V = leading
  `v_head_dim` of K); per-key padding masks.
- Dispatch: the wrapper calls `forward_dual` when `mla_attn.dual` (C128A), else
  the single-cache `forward`.

## Current limitations

- **Memory not yet optimised.** The compressed + state caches are full-size (same
  spec as the window cache, to share the block table), and the window cache is a
  full cache attended with a window mask (not a bounded sliding-window cache). So
  HCA layers use ~4× a normal layer's KV memory. Right-sizing (bounded
  sliding-window window cache + smaller compressed/state groups) needs multi-group
  block-table support in the TT runner — the runner reads `block_table[0]` and
  broadcasts one `TTMetadata` today.
- **C128A only.** The C4A overlapping compressor (`coff=2`) and its top-k sparse
  indexer are not implemented (`TTHCACompressor` asserts `compress_ratio != 4`);
  C1/C4 layers use the single-cache path.
- **bf16 only.** vLLM v4 uses FP8 (`fp8_ds_mla`, packed `head_bytes`) for the
  caches and FP8 einsums in o_proj; the TT path is bf16. `wo_a.weight.view(...)`
  assumes unquantized weights — a real FP8 checkpoint needs dequant.
- **Prefill assumes start-at-0** per user (no prefix cache / chunked-prefill
  continuation) for the compressor APE/window alignment (token `i` in a window
  gets `ape[i]`). Mid-sequence prefill needs position-aware ape indexing.
- **`platform.get_attn_backend_cls`** raises on `use_sparse`; if vLLM core flags
  v4 as sparse during setup this may surface independently of the HCA path.
- **Validation:** all logic verified on CPU only (no real weights, no TT
  hardware). Numerics unvalidated on device.
- **`setup.py` pins `vllm==0.19.1`** but the code targets the installed 0.20.1
  (v4 isn't in 0.19.1). Pin should be bumped.

## Validation status

| Area | How verified |
|------|--------------|
| Backend/impl prefill+decode + cache writes | CPU numeric test (`flash_mla_prefill`/`paged_flash_mla_decode` CPU refs) |
| OOT dispatch (wrapper → TT class) | `op_registry_oot` + `PluggableLayer.__new__` |
| Outer wrapper forward (2D/3D shapes, rope call contract, o_proj, dual dispatch) | CPU mock test |
| Gated compression (`_hca_gated_compress`) | CPU test vs naive per-window softmax-pool reference |
| Joint dual attention (`hca_dual_attention`) | CPU test vs naive reference (sink + masks); sink shown to change output |
| `forward_dual_decode` | CPU test vs gather+joint reference; window mask keeps exactly `window_size` keys |
| **Runner-paged dual cache end-to-end** | CPU test: prefill compresses N windows correctly; decode pools partial window into the right slot each step; completed decode window == from-scratch batch compression; `num_compressed` accounting |
| On-device numerics | NOT verified |
