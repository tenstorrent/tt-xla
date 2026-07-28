# MLA Prefill Code Flow (vllm_plugin)

Full code flow for an MLA (Multi-head Latent Attention) prefill through the
`vllm_plugin`, traced from registration through to the device custom calls.

## Overview

There are three phases: **(A) registration/setup**, **(B) model construction +
weight prep**, and **(C) per-step prefill execution**. The key trick is that the
plugin replaces vLLM's stock MLA layer with TT subclasses via an out-of-tree
(OOT) registry, so model code instantiating `MultiHeadLatentAttentionWrapper`
transparently gets the TT version that routes through a single unified
`forward()` calling two custom ops: `tt.flash_mla_prefill` and
`tt.paged_fill_cache`.

All paths below are relative to the repo root. `vllm_tt/` =
`integrations/vllm_plugin/vllm_tt/`.

---

## Phase A — Registration & backend selection

**1. Plugin entry point** — `vllm_tt/__init__.py:26`
vLLM discovers the plugin and calls `register()`, which returns the platform
class path `"vllm_tt.platform.TTPlatform"`. At module import time it also
registers two backends as *class-path strings only* (no real import yet, to
dodge a partially-initialized `vllm.config`):
- `CUSTOM` → `TTAttentionBackend` (non-MLA)
- `FLASH_ATTN_MLA` → `TTMLAAttentionBackend` (`vllm_tt/__init__.py:20`)

**2. Deferred MLA wiring** — `vllm_tt/platform.py:245` `TTPlatform.check_and_update_config`
Called during `vllm.LLM(...)` → `create_engine_config`, by which point
`vllm.config` is fully loaded. It does `from . import attention_mla`
(`vllm_tt/platform.py:257`), whose import has two side effects:
- Registers the `FLASH_ATTN_MLA` backend again (`vllm_tt/attention_mla.py:436`).
- `@MultiHeadLatentAttentionWrapper.register_oot` on
  `TTMultiHeadLatentAttentionWrapper` (`vllm_tt/attention_mla.py:381`) installs
  the TT wrapper into vLLM's OOT layer registry under the key
  `MultiHeadLatentAttentionWrapper`.

**3. Backend selection** — `vllm_tt/platform.py:160` `get_attn_backend_cls`
When `attn_selector_config.use_mla` is true, returns
`AttentionBackendEnum.FLASH_ATTN_MLA.get_path()` → resolves to
`TTMLAAttentionBackend`, whose `get_impl_cls()` is `TTMLAAttentionBackendImpl`
and whose `get_kv_cache_shape()` is the single concatenated latent layout
`(num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim)`
(`vllm_tt/attention_mla.py:66`).

---

## Phase B — Model construction & weight prep

**4. Layer swap at construction** — `vllm_tt/attention_mla.py:381`
When model code (e.g. DeepSeek) does
`self.mla_attn = MultiHeadLatentAttentionWrapper(...)`,
`PluggableLayer.__new__` consults the OOT registry and instead constructs
`TTMultiHeadLatentAttentionWrapper`.

**5. Nested layer swap** — `vllm_tt/attention_mla.py:393`
`TTMultiHeadLatentAttentionWrapper.__init__` temporarily rebinds the
module-level name `vllm.model_executor.layers.mla.MLAAttention =
TTMLAAttention`, runs the upstream `super().__init__()` (which builds
`self.mla_attn = MLAAttention(...)`), then restores the name in a `finally`. Net
effect: `self.mla_attn` is a `TTMLAAttention` instance, and its `.impl` is
`TTMLAAttentionBackendImpl` — without reimplementing the wrapper's `__init__` or
double-registering the layer.

**6. Weight absorption** — upstream `mla_attention.py` `process_weights_after_loading`
Splits `kv_b_proj` into up-projection matrices and stores them as **plain tensor
attributes** (not `nn.Parameter`/buffers):
- `layer.W_UK_T` : `[num_heads, qk_nope_head_dim, kv_lora_rank]`
- `layer.W_UV`  : `[num_heads, kv_lora_rank, v_head_dim]`

Because they're plain attributes, `model.to('xla')` doesn't move them — which is
why the impl does explicit `.to(device=...)` later
(`vllm_tt/attention_mla.py:237,266`).

---

## Phase C — Per-step prefill execution

**7. Metadata build** — `vllm_tt/model_runner.py:889` `_prepare_inputs`
Builds the per-layer `TTMetadata` (`vllm_tt/model_runner.py:1077`):
- `cache_position = seq_lens - 1` (`vllm_tt/model_runner.py:1032`)
- `page_table` from the input batch block tables
- `fill_page_table` — `page_table` with each user's prefix blocks rolled to the
  end so `paged_fill_cache` writes suffix blocks instead of clobbering shared
  prefix blocks (`vllm_tt/model_runner.py:1043-1056`)
- `is_causal=True`

`TTMetadata` itself is defined at `vllm_tt/attention.py:170`. The same object is
fanned out to every attention layer:
`per_layer_attn_metadata = {layer_name: attn_metadata ...}`
(`vllm_tt/model_runner.py:1107`).

**8. Forward under context** — `vllm_tt/model_runner.py:1342`
In `sample_tokens`, the model runs inside
`set_forward_context(attn_metadata, vllm_config, num_tokens=...)`, then
`self.model(input_ids=..., positions=self.position_ids, inputs_embeds=...)`
(`vllm_tt/model_runner.py:1352`).

**9. TT wrapper forward** — `vllm_tt/attention_mla.py:398` `TTMultiHeadLatentAttentionWrapper.forward`
- TT model runner passes 3D `[users, S, H]` hidden_states + 2D `[users, S]`
  positions. This flattens them to vLLM's standard 2D `[tokens, H]` / 1D
  `[tokens]` (`vllm_tt/attention_mla.py:419-423`) — necessary because the
  upstream `k_pe.unsqueeze(1)` only behaves correctly for 2D input (would
  otherwise broadcast wrong against the DeepSeek rope cos/sin).
- Calls `super().forward()` = upstream `MultiHeadLatentAttentionWrapper.forward`,
  which does the **MLA preprocess**:
  - `fused_qkv_a_proj` → split into `q_c` and `kv_lora`; `q_a_layernorm`;
    `q_b_proj` → `q` (or the `q_proj`/`kv_a_proj_with_mqa` path when
    `q_lora_rank is None`)
  - split `kv_lora` → `kv_c`, `k_pe`; `kv_a_layernorm(kv_c)` → `kv_c_normed`
  - `q.view(-1, num_heads, qk_head_dim)`; `k_pe.unsqueeze(1)`
  - apply `rotary_emb` to `q[..., qk_nope_head_dim:]` and `k_pe`
  - calls `self.mla_attn(q, kv_c_normed, k_pe,
    output_shape=(tokens, num_heads*v_head_dim))` → **`TTMLAAttention.forward`**
- Reshapes the 2D output back to 3D `[users, S, ...]`
  (`vllm_tt/attention_mla.py:428`).

**10. TTMLAAttention layer** — `vllm_tt/attention_mla.py:327`
- Splits `q` → `(q_nope, q_pe)` along the last dim
  (`vllm_tt/attention_mla.py:341`)
- Pulls `attn_metadata` out of the forward context
  (`vllm_tt/attention_mla.py:344`) and `kv_cache = self.kv_cache`
- Allocates the `output` buffer and calls
  `self.impl.forward(q=(q_nope,q_pe), kv_c_normed, k_pe, kv_cache,
  attn_metadata, layer=self, output=output)` (`vllm_tt/attention_mla.py:353`)

**11. The unified impl forward** — `vllm_tt/attention_mla.py:171`
`TTMLAAttentionBackendImpl.forward` — this is where the real work happens:

| Step | Code | What it does |
|---|---|---|
| Prefill guard | `:196` | `_infer_is_prefill` (heuristic: >1 token/user); decode path raises `NotImplementedError` |
| Reshape | `:221-224` | inputs → `[users, S, ...]` (`q_nope`→`[b,S,N,P]`, `q_pe`→`[b,S,N,R]`, `kv_c`→`[b,S,L]`, `k_pe`→`[b,S,1,R]`) |
| **Q absorption** | `:234` | `einsum("bsnp,npl->bsnl", q_nope, W_UK_T)` (in fp32) → `q_nope_lat [b,S,N,L]` |
| Build latent Q/K | `:241-242` | `q_lat = cat([q_nope_lat, q_pe])` `[b,S,N,L+R]`; `k_lat = cat([kv_c, k_pe])` `[b,S,1,L+R]` |
| Transpose | `:249-250` | → `q_for_kernel [b,N,S,L+R]`, `k_for_kernel [b,1,S,L+R]` |
| **Attention** | `:251` | `torch.ops.tt.flash_mla_prefill(query, key, head_dim_v=L, value=None, is_causal, scale)` → `out_lat [b,N,S,L]` |
| **W_UV proj** | `:263` | `einsum("bnsl,nlv->bnsv", out_lat, W_UV)` → `out [b,N,S,V]` |
| Reshape | `:272` | → `[tokens, N*V]` (vLLM's output contract) |
| **Persist KV** | `:279-294` | per-user loop calling `torch.ops.tt.paged_fill_cache(kv_cache, k_lat_for_fill[b], fill_page_table, batch_idx)`; skipped when `attn_metadata is None` or `kv_cache.numel()==0` (profile-run sentinels) |
| Write output | `:297-300` | `output.copy_(out)` |

Notes:
- V is *not* passed — the latent K and V share the compressed representation, so
  the kernel takes V as the leading `head_dim_v=L` features of K.
- Attention runs **before** the cache write so a kernel failure leaves the cache
  untouched (`vllm_tt/attention_mla.py:244-248`).
- The einsums upcast to fp32 then cast back to the activation dtype.

---

## Custom ops → device

**12. `tt.flash_mla_prefill`** — `python_package/tt_torch/custom_ops.py:336`
After shape asserts (notably `query.shape[2] % 32 == 0`, tile alignment), on
`xla` it emits `stablehlo_custom_call("tt.flash_mla_prefill", ...)` with frontend
attributes `head_dim_v`, `is_causal`, `has_value`, `scale`
(`custom_ops.py:447`). On `cpu` it falls back to
`F.scaled_dot_product_attention` with `value = key[..., :head_dim_v]` and
`enable_gqa=True` (`custom_ops.py:455`). The fake/meta impl returns zeros of
shape `[b, nqh, s, head_dim_v]` (`custom_ops.py:475`).

**13. `tt.paged_fill_cache`** — `python_package/tt_torch/custom_ops.py:766`
On `xla`, emits `stablehlo_custom_call("tt.paged_fill_cache", [cache.shape],
[cache.dtype])`; on `cpu` it block-scatters `fill_value` into `cache` rows
selected by `page_table[batch_idx]` (`custom_ops.py:784-839`).

**14. Lowering** — These StableHLO custom calls flow through the PJRT plugin →
TT-MLIR compiler, which lowers `tt.flash_mla_prefill` to
`ttnn.transformer.flash_mla_prefill` and the cache write to the ttnn
paged-fill-cache op on Tenstorrent hardware.

---

## Open items in the current branch

- **`paged_fill_cache` is defined twice** — `custom_ops.py:766` and again at
  `custom_ops.py:857`, with identical bodies. The second
  `@torch.library.custom_op("tt::paged_fill_cache", ...)` re-registers/shadows
  the first. Harmless if identical, but redundant and a likely merge artifact
  worth removing.
- **Decode path is stubbed** — the unified forward raises
  `NotImplementedError` for the decode branch; this flow covers prefill only.
