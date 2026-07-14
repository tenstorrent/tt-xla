# DeepSeek-V4 attention on `tt` — what shipped, and concrete next steps

This is the companion to `DSV4_TT_Attention_Design.md`. It records what was
implemented and validated for the **SWA-only first milestone**, and gives
ready-to-apply diffs for the pieces that need a `tt-mlir` rebuild or are
deferred to later milestones.

---

## 1. What shipped (SWA-only, validated)

**Definition of done met:** single DeepSeek-V4 sliding-window attention layers run
end-to-end through the tt-xla vLLM plugin, validated by a focused test suite —
with the SWA-only slice green first, on real hardware.

Files:

| File | Change |
|---|---|
| `python_package/tt_torch/custom_ops.py` | `tt.paged_flash_mla_decode`: added `sliding_window` arg (emits `sliding_window_size` frontend attr on xla); **fixed the CPU reference to model the attention-sink fold** (`out *= sigmoid(lse - sink)`, the `TODO(@hshah)`) and the sliding-window lower-bound mask. Backward compatible (defaults `None` → identical to before; the existing MLA path is untouched). |
| `integrations/vllm_plugin/vllm_tt/attention_impls/attention_dsv4.py` | **[new]** `TTDeepseekV4AttentionBackend` + `TTDeepseekV4AttentionBackendImpl` (window/sink-aware MLA impl, SWA-only), **plus `TTDeepseekV4MLAWrapper`** — the OOT layer replacement that constructs the CUDA-only upstream wrapper on tt via the monkeypatch recipe (§3.2). |
| `integrations/vllm_plugin/vllm_tt/__init__.py` | registers the DSV4 backend against `AttentionBackendEnum.FLASHMLA_SPARSE` (§3.1). |
| `tests/.../oot_backends/test_dsv4_swa_attention_impl.py` | **[new]** 6 focused impl tests (11 params), green on `single_device`. |
| `tests/.../oot_backends/test_dsv4_backend_registration.py` | **[new]** 3 backend-registration tests. |
| `tests/.../oot_backends/test_dsv4_wrapper_construction.py` | **[new]** 3 tests: OOT wrapper is registered, **constructs on tt** via the monkeypatch recipe, and `forward` is (currently) `NotImplementedError`. |

**Test results (`pytest -m "push or single_device"`, wormhole_b0): 17 passed**
(11 impl + 3 registration + 3 construction).

| Mechanism | Test | Reference | Runs on HW today? |
|---|---|---|---|
| SWA prefill (window + sink) | `test_dsv4_swa_prefill` | CPU-impl path | ✅ yes (PCC ≥ 0.99) |
| Sink fold math | `test_dsv4_attention_sink_math` | analytic fp32 + strong-sink→0 limit | ✅ (CPU) |
| Window boundary | `test_dsv4_sliding_window_boundary` | structural (perturbation) | ✅ yes (cpu + tt) |
| Attention sink (decode) | `test_dsv4_attention_sink_decode` | CPU-impl path (native ttnn sink) | ✅ yes (PCC ≥ 0.99) |
| Windowed decode | `test_dsv4_swa_decode_window` | analytic fp32 | ⚠️ CPU only (see §2) |
| Prefill→decode + cache | `test_dsv4_prefill_then_decode` | CPU-impl path | ✅ yes |

**How each mechanism maps to kernels (validated on HW):**

* **Sliding-window prefill** → `tt.flash_mla_prefill` with a banded windowed-causal
  additive `attn_mask` (`is_causal=False`). `flash_mla_prefill` accepts an arbitrary
  mask, so this needs no kernel/compiler change.
* **Attention sink (prefill)** → folded in the impl from the (pre-sink) log-sum-exp
  of the windowed logits: `out *= sigmoid(lse - sink)`. Plain ops; lowers to
  StableHLO; runs on HW.
* **Attention sink (decode)** → native `attention_sink` operand on
  `tt.paged_flash_mla_decode` (the ttnn kernel folds it). **The sink tensor must be
  bf16** — the kernel asserts `attention_sink.dtype()==BFLOAT16`; the impl converts.
* **Sliding-window decode** → native `sliding_window` arg → `sliding_window_size`
  frontend attr. This is the one piece gated on a tt-mlir change (§2): the MLA
  *decode* kernel is causal-only (`TT_FATAL: Multi-latent attention decode only
  tested for causal!`), so the prefill mask trick is unavailable, and the runtime
  executor currently hardcodes `slidingWindowSize=std::nullopt`.

---

## 2. tt-mlir: thread `sliding_window_size` through `paged_flash_mla_decode` (HW) — ✅ DONE

**Status: applied, rebuilt, and verified on the tt device.** The full 9-file
diff (7 runtime/flatbuffer path + EmitC/EmitPy) is recorded verbatim in
`tt_mlir_changes.md` (against tt-mlir `260d4c49`). `cmake --build build --target
install` recompiled + reinstalled `libTTMLIRCompiler.so` / `libTTMLIRRuntime.so`
into `third_party/tt-mlir/install/lib` — which the pre-built `pjrt_plugin_tt.so`
dynamically links, so no plugin relink was needed. Verified: `paged_flash_mla_decode`
with an active window now matches the CPU windowed reference on the tt device
(PCC 0.9997) and differs from full-causal (0.55), i.e. the window is genuinely
applied. `test_dsv4_swa_decode_window` now asserts this on hardware.

The ttnn kernel `paged_flash_multi_latent_attention_decode` already accepted
`sliding_window_size`; the change just threaded the frontend attribute through
the compiler + runtime, copying the fully-plumbed sibling
`PagedScaledDotProductAttentionDecodeOp`. The diffs below (kept for reference)
copy that idiom. **To re-apply on a clean tt-mlir branch, use `tt_mlir_changes.md`.**

### 2.1 `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (~8747)

Read the attr right after the `scale` block in
`StableHLOToTTIRPagedFlashMLADecodeOpConversionPattern`:

```cpp
    // (after scaleAttr is built, before parseHasFlag lambda)
    auto slidingWindowSizeStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("sliding_window_size");
    IntegerAttr slidingWindowSizeAttr = nullptr;
    if (slidingWindowSizeStringAttr) {
      uint32_t slidingWindowSize;
      if (!llvm::to_integer(slidingWindowSizeStringAttr.getValue(),
                            slidingWindowSize)) {
        return rewriter.notifyMatchFailure(
            srcOp, "sliding_window_size attribute string must be convertible "
                   "to a non-negative integer.");
      }
      slidingWindowSizeAttr = rewriter.getUI32IntegerAttr(slidingWindowSize);
    }
```

and pass it to the builder (~8810). The TTIR op builder arg order must match the
new `.td` (place `sliding_window_size` after `scale`):

```cpp
    rewriter.replaceOpWithNewOp<
        mlir::tt::ttir::PagedFlashMultiLatentAttentionDecodeOp>(
        srcOp, outputType, query, key, value, headDimVAttr, pageTable,
        isCausalAttr, attentionMask, curPosTensor, attentionSink, scaleAttr,
        slidingWindowSizeAttr);   // <-- added
```

### 2.2 `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` (~6205, `TTIR_PagedFlashMultiLatentAttentionDecodeOp`)

```tablegen
                       Optional<AnyRankedTensor>:$attention_sink,
                       OptionalAttr<F32Attr>:$scale,
                       OptionalAttr<UI32Attr>:$sliding_window_size);  // <-- added
```

(No verifier change required; `sliding_window_size` is an independent optional
attribute. Mirror the doc string from `PagedScaledDotProductAttentionDecodeOp`.)

### 2.3 `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (~3237, PagedFlashMLADecode pattern)

```cpp
    rewriter.replaceOpWithNewOp<ttnn::PagedFlashMultiLatentAttentionDecodeOp>(
        op, this->getTypeConverter()->convertType(op.getType()),
        adaptor.getQuery(), adaptor.getKey(), adaptor.getValue(),
        static_cast<uint32_t>(adaptor.getHeadDimV()), adaptor.getPageTable(),
        adaptor.getIsCausal(), adaptor.getAttentionMask(),
        adaptor.getCurPosTensor(), adaptor.getAttentionSink(),
        adaptor.getScaleAttr(),
        adaptor.getSlidingWindowSizeAttr());   // <-- added
```

### 2.4 `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` (~3925, `TTNN_PagedFlashMultiLatentAttentionDecodeOp`)

```tablegen
                         Optional<AnyRankedTensor>:$attention_sink,
                         OptionalAttr<F32Attr>:$scale,
                         OptionalAttr<UI32Attr>:$sliding_window_size);  // <-- added
```

### 2.5 `include/ttmlir/Target/TTNN/operations/transformer.fbs` (table `PagedFlashMultiLatentAttentionDecodeOp`, ~148)

```fbs
  scale: float = null;
  sliding_window_size: uint32 = null;   // <-- added (before out/memcfg)
  out: tt.target.ttnn.TensorRef;
  memcfg: tt.target.ttnn.MemoryConfig;
```

### 2.6 `lib/Target/TTNN/TTNNToFlatbuffer.cpp` (~3384, `createOp(... PagedFlashMultiLatentAttentionDecodeOp)`)

Add near the other field conversions:

```cpp
  auto slidingWindowSize = toFlatbuffer(cache, op.getSlidingWindowSize());
```

and pass it into the `Create...` call (~3425), matching the new `.fbs` field order:

```cpp
  return ::tt::target::ttnn::CreatePagedFlashMultiLatentAttentionDecodeOp(
      *cache.fbb, query, key, value, headDimV, pageTable, isCausal,
      attentionMask, curPosTensor, attentionSink, scale,
      slidingWindowSize,   // <-- added
      out, memoryConfig);
```

### 2.7 `runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp` (line 50)

```cpp
  std::optional<float> scale = op->scale();
  std::optional<uint32_t> slidingWindowSize =
      op->sliding_window_size() ? std::optional<uint32_t>(op->sliding_window_size())
                                : std::nullopt;   // <-- was: = std::nullopt;
```

(The ttnn call at line ~59 already passes `slidingWindowSize` positionally — no
further change. Also check `TTNNWorkaroundsPass` / `TTNNOpModelInterface` if they
reconstruct the op; the SDPA-decode sibling shows the pattern.)

Done: `test_dsv4_swa_decode_window` now asserts the TT-device path with an active
window (PCC 0.9997 vs cpu-windowed; the full suite passes stably across repeated
runs).

### 2.8 Known limitation — the plugin's compiled-program cache ignores this attr

The tt PJRT executable cache keys on tensor shapes/dtypes but **not** on the
`paged_flash_mla_decode` custom-call frontend attributes (`sliding_window_size`,
and by extension the sink flag). Two decode calls with identical shapes but
different `sliding_window` in one process collide — the first-compiled program is
reused, silently dropping the second's window. Proven directly: same shape,
no-window decode first → then `window=16` gives PCC 0.9998 vs cpu-**full-causal**,
0.55 vs cpu-windowed.

**Real models are unaffected** — every SWA layer shares one `config.sliding_window`,
so no same-shape/different-window variation ever occurs. This only surfaces in
tests that sweep windows over a fixed shape; the decode-window test therefore uses
`cur_pos=80` (2 cache blocks), a shape no no-window decode compiles. If per-shape
multi-window support in a single process is ever needed, the executable cache key
must be extended to include the custom-call frontend attributes.

---

## 3. Full-model wiring (deferred; out of scope for this milestone)

The upstream `DeepseekV4MultiHeadLatentAttentionWrapper` is **CUDA-only** — its
`__init__` asserts `get_device_capability() is not None`, and it uses
`torch.cuda.Event` / fp8 einsum / FlashMLA sparse kernels. It cannot be
constructed on `tt`, so a full-model forward is blocked upstream, not by the
plugin. Running a full DSV4 model on TT therefore first requires a TT-native
replacement layer. The pieces, as concrete changes:

### 3.1 `platform.py` — gate the `use_sparse` rejection for DSV4 — ✅ DONE

`get_attn_backend_cls` used to raise unconditionally on `use_sparse`; DSV4 sets
`use_sparse=True`. Since `AttentionSelectorConfig` carries no model architecture,
DSV4 can't be told apart from DeepSeek-V3.2 (also sparse-MLA) from the selector
args, so the gate keys off `get_current_vllm_config().model_config.hf_config.`
`architectures` (`"DeepseekV4ForCausalLM"`). Implemented as `platform.py`:

```python
if attn_selector_config.use_sparse:
    if cls._is_deepseek_v4():   # architectures check, best-effort, guarded
        return "vllm_tt.attention_impls.attention_dsv4.TTDeepseekV4AttentionBackend"
    raise NotImplementedError("Sparse Attention is not supported on TT devices.")
```

`_current_model_architectures()` reads the current vLLM config (empty list if
unavailable → falls back to the old raise, so **non-DSV4 sparse models are
unchanged**). The backend is also registered in `__init__.py` via
`register_backend(AttentionBackendEnum.FLASHMLA_SPARSE, "...TTDeepseekV4AttentionBackend")`.
Covered by `test_dsv4_engine_wiring.py::test_gating_*` (DSV4→backend, other
sparse→raise, MLA path unchanged) and `test_dsv4_backend_registration.py`.

### 3.2 A TT-native DSV4 layer (replaces the CUDA wrapper)

**Prototype done (construction):** `TTDeepseekV4MLAWrapper` in `attention_dsv4.py`,
registered via `@DeepseekV4MultiHeadLatentAttentionWrapper.register_oot` (the same
`PluggableLayer.register_oot` hook the V3 `TTMultiHeadLatentAttentionWrapper` uses;
`PluggableLayer.__new__` dispatches on the base `cls.__name__`, so our `__init__`
runs). Rather than rewriting the CUDA-bound base ctor, it **runs `super().__init__()`
under temporary monkeypatches** (the tpu-inference technique), so the base still
builds all the device-agnostic submodules:

* `torch.cuda.Event` / `torch.Event` → no-op (GPU stream-overlap events);
* `current_platform.get_device_capability` → a dummy `.major` (defeats the
  `assert cap is not None` CUDA guard);
* `current_platform.device_type` → `"cpu"` (buffer allocs land on CPU; unused);
* `cache_config.cache_dtype` → `"fp8_ds_mla"` for the duration (satisfies
  `DeepseekV4MLAAttention`'s fp8 assert), then restored (tt stays bf16).

Verified on the tt platform by `tests/.../oot_backends/test_dsv4_wrapper_construction.py`
(construction succeeds; `mla_attn` + `swa_cache_layer` are built; `cache_dtype`
is restored). Driver findings for a standalone construction: also set the current
config via `set_current_vllm_config(...)` (a `SimpleNamespace` works — no type
check), `compilation_config.custom_ops = ["none"]` (native CustomOp dispatch, no
CUDA), and stub `get_tensor_model_parallel_world_size → 1` (no distributed init).

**Forward — implemented + validated on tt (prefill / fresh-sequence path).**
`TTDeepseekV4MLAWrapper.forward` reimplements the upstream fused-CUDA forward in
**bf16, no quantization**: `fused_wqa_wkv → split → q_norm/kv_norm → wq_b →
per-head Q RMSNorm → decoupled RoPE → windowed + attention-sink attention →
inverse-RoPE → grouped o-proj`. Validated on the tt device (bf16 tt-vs-cpu PCC
~0.9998; fp32 vs a pure-torch reference ~1.0; `test_dsv4_wrapper_forward.py`).
The base ctor's fp8 activation quantizer (`QuantFP8`, `self._wo_a_act_quant`) is
**dropped in `__init__`** and `mla_attn.kv_cache_dtype` reset to `auto` — weights
are assumed dequantized to bf16 at load time.

Two findings that shaped it:

1. **DSV4 is *direct* MLA, not V3 absorption.** The SWA path is
   `q = wq_b(qr).view(-1, N, head_dim)` and `kv` is the `[T, head_dim]` latent
   (nope+rope); attention is Q·K over the full `head_dim` with `head_dim_v ==
   head_dim` (V = the *full* latent, un-roped in the o-proj). There is **no
   `W_UK_T`/`W_UV`** — so `TTDeepseekV4AttentionBackendImpl` (which is V3-absorption
   shaped) does not fit DSV4; the wrapper does direct MLA via the ops instead.
2. **V == qk padding workaround.** ttnn `flash_mla_prefill` requires
   `head_dim_v < qk`. DSV4 needs `head_dim_v == qk`, so we zero-pad the qk head
   dim by one tile (`_MLA_V_PAD`): `V = key[..., :head_dim_v]` stays the real
   latent and the padded dims contribute 0 to scores. Validated on tt (~0.9998).

> A `MeshBuffer` error seen during bring-up turned out to be a **test-harness
> artifact** — building the `attn_sink` on the xla device with an in-place
> slice-assign (`torch.full(..., device=xla); sink[:N] = cpu_tensor`) left a
> corrupt pending xla op that poisoned the device. Building the sink on cpu and
> `.to(dev)`-ing it fixed it; wrapper construction + forward run cleanly on tt.
> (Lesson: avoid xla in-place slice-assignment during layer setup.)

**Paged SWA KV cache — done + validated (attention level).** `forward` is now
cache-aware: it reads `self.swa_cache_layer.kv_cache` + a `TTMetadata`
(page_table / cache_position / fill_page_table) from the forward context and
dispatches **paged prefill** (windowed self-attend + write latents via
`tt.paged_fill_cache`) vs **paged decode** (write the new token via
`tt.paged_update_cache`, then a windowed paged read via
`tt.paged_flash_mla_decode` with sink + `sliding_window`). Cache layout
`(num_blocks, 1, block_size, head_dim)` bf16. Validated on tt by
`test_dsv4_wrapper_forward.py::test_dsv4_paged_cache_prefill_decode`: prefill
writes the roped latents (verified against the gathered cache), decode reads the
windowed history (fp32 vs a pure-torch reference; bf16 tt-vs-cpu). Notes:
`paged_flash_mla_decode` allows `head_dim_v == qk` (no padding needed, unlike
prefill); the decode `sliding_window` now applies on HW (§2 landed). This
round-trip test uses a window ≥ history (tt-causal == cpu-windowed), so it does
not itself exercise an active window — that is covered by
`test_dsv4_swa_decode_window` (§2.8 explains the per-shape cache caveat).

**Remaining (forward):**
* the **RoPE convention** (GPT-J interleave via `cos_sin_cache`) matches the
  reading of DSV4's fused `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`;
  cross-check on GPU (that fused op is CUDA-only, not runnable here).
* **chunked prefill with history** (prefill that also reads a prior cache) — the
  current paged prefill attends only the current chunk; multi-chunk needs the
  cached-history read path.

### 3.3 `model_runner.py` — feed the cache + metadata to the DSV4 layer

The layer now *consumes* a paged SWA cache + `TTMetadata` (see "Paged SWA KV
cache" above). The model_runner now *supplies* them for **SWA-only** DSV4,
reusing the existing **single-group** MLA path (all SWA layers share one cache
spec type). ✅ **DONE:**

* **`get_kv_cache_spec`** — added a gated branch (`model_runner.py`). The DSV4
  wrapper owns a separate `DeepseekV4SWACache` layer (an `AttentionLayerBase`
  that is neither `Attention` nor `MLAAttention`, so it used to fall through to
  `continue` → no cache). The branch emits a **bf16 `MLAAttentionSpec`**
  (`num_kv_heads=1`, `head_size=head_dim`, `block_size=64`) for it. It
  deliberately does **not** call the module's own `get_kv_cache_spec`, which
  returns the upstream uint8 / 584-B `SlidingWindowMLASpec` (`alignment=576`,
  `model_version="deepseek_v4"`); the TT impl reads a bf16
  `(num_blocks, 1, block_size, head_dim)` cache. The SWA-only
  `DeepseekV4MLAAttention` layer holds no cache of its own (skipped); a
  compressed (C4A/C128A, `compress_ratio > 1`) layer raises `NotImplementedError`
  (a second cache group is not supported yet). Gated on the DSV4 layer classes
  (lazy import → `(None, None)` on vLLM builds without DSV4), so **non-DSV4
  models are byte-for-byte unchanged**.
* **Metadata fan-out** — no change needed. `_attention_layer_names` is
  `get_layers_from_vllm_config(vllm_config, AttentionLayerBase).keys()`, which
  already includes `DeepseekV4SWACache`, so the single `TTMetadata` built in
  `_prepare_inputs` is already fanned out (`dict.fromkeys`) to
  `swa_cache_layer.prefix` — exactly the key the wrapper looks up. For SWA-only
  one metadata object per step is all the layer needs.
* **`initialize_kv_cache`** — no change needed. The single-group `MLAAttentionSpec`
  path allocates `(num_blocks, 1, block_size, head_size)` bf16 via
  `TTMLAAttentionBackend.get_kv_cache_shape` and binds it to the SWA cache
  layer's `.kv_cache` by prefix — the exact tensor the impl reads. `block_size=64`
  matches the DSV4 backend's `get_page_size`.

Covered by `test_dsv4_engine_wiring.py::test_kv_cache_spec_*`.

The `>1 group` lift + per-branch metadata below is only needed for **C4A/C128A**
(compressed branch = a second cache group). Validating end-to-end needs a
**loadable DSV4 model** — platform gating (§3.1, done) + the MoE (§3.4) + a real
checkpoint. When adding the compressed branch, **gate every change so
single-group (non-DSV4) models stay byte-for-byte unchanged:**

* `initialize_kv_cache:3328` hard-raises on `len(kv_cache_groups) > 1`. DSV4 (SWA
  cache + compressed cache) needs ≥2 groups. Lift the guard; make the downstream
  single-group assumptions per-group (`input_batch.block_table[0]` indexed at
  1352/1361/3366; single `self.block_size` / `max_num_blocks_per_req`; the
  persistent device-buffer pools at 614–667).
* `get_kv_cache_spec:1021` builds one `MLAAttentionSpec` per MLA layer. A DSV4 SWA
  layer needs a `SlidingWindowSpec`-flavored latent cache; C4A/C128A layers add a
  second compressed `MLAAttentionSpec(compress_ratio=...)` under a distinct key.
* The single-`TTMetadata` `dict.fromkeys(self._attention_layer_names, attn_metadata)`
  fan-out (1521 / 2322 / 2722) must become per-group: build N `TTMetadata` with
  per-branch page_table / cache_position / slot-mapping and assign by
  `KVCacheGroupSpec.layer_names`. Add `sliding_window` / `attn_sink` to the
  DSV4-branch metadata.

### 3.4 MoE — can the existing TT MoE backend be reused? (mostly ✅)

`DeepseekV4MoE.__init__` branches on `kernel_config.moe_backend`:

* **`"deep_gemm_mega_moe"` → `DeepseekV4MegaMoEExperts`** — CUDA / fp4 / DeepGEMM
  only (`raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA")`,
  `... requires SM100 GPUs`). **Not usable on TT — must be avoided** (do not pass
  `--kernel-config moe_backend=deep_gemm_mega_moe`; the default backend selection
  must not resolve to it on TT).
* **otherwise → `self.experts = FusedMoE(...)`** — the *standard* vLLM `FusedMoE`
  layer, which the plugin's existing **`TTFusedMoE`** (OOT-registered for
  `FusedMoE`, `layers/fused_moe.py`) already intercepts and runs via
  `tt_dense_experts_forward` / `tt_experts_forward`. **So the expert-dispatch
  backend is reused as-is.**

The gap was **routing**. DSV4 uses `scoring_func="sqrtsoftplus"` +
`e_score_correction_bias` (noaux_tc) + `routed_scaling_factor`, and for early
layers hash routing (`gate.tid2eid`). vLLM's own sqrtsoftplus router is a
**CUDA-only custom op** (`ops.topk_hash_softplus_sqrt` → `_moe_C.topk_softplus_sqrt`)
with **no device-agnostic fallback** (`grouped_topk`/`select_experts` only handle
`softmax`/`sigmoid`), and `TTFusedMoE.forward_native` previously did plain
softmax+top-k — wrong for DSV4. ✅ **Fixed:** `TTFusedMoE._route_native` /
`_route_sqrtsoftplus` now reproduce `scores = sqrt(softplus(logits))` → noaux_tc
bias-select (weight by *unbiased* scores) → renormalize → `× routed_scaling_factor`,
gated on `scoring_func == "sqrtsoftplus"` so all other models keep the exact
softmax+top-k path. **Hash routing (early layers, `tid2eid`) is now implemented**
too: `input_ids` is threaded `apply_monolithic → forward_native → _route_sqrtsoftplus`,
and when `hash_indices_table` is set the expert ids come from `tid2eid[input_ids]`
(weights are still the unbiased sqrtsoftplus scores, computed via one-hot × sum —
SPMD-safe, not `.gather`, per the reference). Covered by
`test_dsv4_engine_wiring.py::test_moe_*` and validated **exactly** against the
device-agnostic ground-truth `modified_model.Gate` (hash + noaux_tc) in
`test_dsv4_moe_reference_parity.py` — which resolves the old "validate against a
GPU reference" caveat without a GPU. The real layer-0 MoE (256 fp4 experts →
bf16 + `tid2eid`) runs with hash routing in
`test_dsv4_flash_layer0_e2e.py::test_dsv4_flash_layer0_moe_hash_routing_runs`, and
the new routing runs on the TT device in `..._hash_routing_on_device`.

> ⚠️ **Remaining MoE note:** the full 256-expert layer-0 MoE (~12.9 GB bf16)
> exceeds a single Wormhole's DRAM, so the *on-device* full MoE uses the
> multi-device-sharded sparse-MLP path that `test_deepseek_v4_e2e_streaming`
> already exercises (layer 0 included). The reimplemented routing here matches
> that path's reference numerics.

So a full DSV4 engine run still needs: §3.1 (done) + §3.3 (done) + §3.4 (done for
non-hash) + a **loadable bf16 DSV4 checkpoint** (§3.5). No MoE *stub* is required
after all — the real (non-Mega) MoE path runs on TT.

### 3.5 Weights — bf16 checkpoint (dequant offline) — ✅ tooling done

DSV4-Flash ships fp8 (e4m3, block-128) linears + fp4 (MXFP4, block-32) experts,
and vLLM's DSV4 quant path (`DeepseekV4FP8Config`) is CUDA-oriented — TT does no
fp8 matmul. Rather than a fp8-aware quant method, weights are **dequantized to
bf16 offline** and `quantization_config` is stripped, so vLLM loads an
*unquantized* model → `UnquantizedLinearMethod` / `UnquantizedFusedMoEMethod`
(→ `TTFusedMoE`) + the bf16 MLA/SWA path. This is the same idea as
`deepseek_v3_2_exp/build_weight_cache.py`, but **name-preserving** (vLLM's own
`load_weights` does the module mapping) and it **drops the scale tensors**.

`tests/torch/models/deepseek_v4/build_vllm_bf16_checkpoint.py` — streams the
checkpoint shard-by-shard (bounded memory), dequantizes each `.weight`/scale
pair (fp8 or fp4, dispatched by scale shape; the same math as the validated
`weight_loader.py`), passes other tensors through as bf16, rewrites `config.json`
(`quantization_config` removed, `torch_dtype=bfloat16`), and copies tokenizer/aux
files. Handles `.scale` / `.weight_scale_inv` / `.weight_scale` suffixes and an
optional `--n-layers` smoke subset. Validated end-to-end on a synthetic fp8+fp4
checkpoint by `test_build_vllm_bf16_checkpoint.py` (3 tests: dequant + name/scale
handling, config rewrite + aux copy, layer filter).

`tests/integrations/vllm_plugin/generative/test_dsv4_generation.py` — the
ready-to-run vLLM E2E, gated on `DSV4_BF16_CHECKPOINT` pointing at a converted
dir (skipped in CI). Run:

```
python tests/torch/models/deepseek_v4/build_vllm_bf16_checkpoint.py \
    --repo deepseek-ai/DeepSeek-V4-Flash --dst /path/to/dsv4-bf16
DSV4_BF16_CHECKPOINT=/path/to/dsv4-bf16 pytest -svv \
    tests/integrations/vllm_plugin/generative/test_dsv4_generation.py
```

**Open items for the first real run:** (a) confirm vLLM's DSV4 `load_weights`
consumes the same top-level `model.safetensors.index.json` the converter
preserves (else it needs HF-Transformers-format weights); (b) the fp8/fp4 dequant
+ sqrtsoftplus routing are byte-exact vs a GPU reference; (c) disk — a fully
dequantized V4-Flash is large (convert per-`--n-layers` for smoke, full for a
coherent run). An **already-working torch-path E2E** (`test_streaming_dsv4_flash`,
`weight_loader.py` + `modified_model`) validates the DSV4 dequant + attention +
MoE numerics on hardware independently of vLLM — the fastest confidence check.

> The torch-path `modified_model` MoE is also a device-agnostic reference for
> **validating `TTFusedMoE._route_sqrtsoftplus`** (§3.4 caveat 1) with no GPU.

### 3.6 First-layer E2E on hardware — ✅ VALIDATED (real weights)

`tests/torch/models/deepseek_v4/test_dsv4_flash_layer0_e2e.py` runs
**DeepSeek-V4-Flash layer 0 (SWA-only) on the TT device with real dequantized
weights** and matches CPU at **PCC 0.9998**. Confirmed from the real config that
layer 0 is SWA-only: `compress_ratios = [0, 0, 4, 128, 4, 128, …]` (layers 0–1
SWA-only, then C4A/C128A alternate).

This uses the torch `modified_model` + `weight_loader` path (not vLLM), because:
(a) the DSV4-Flash checkpoint ships in DeepSeek **native** tensor naming
(`layers.0.attn.wq_a`, `embed`/`head`/`norm`, ~160 GB / 46 shards), which vLLM's
HF-format `DeepseekV4ForCausalLM` loader does not consume — so the §3.5 converter
would additionally need a native→HF name remap for the vLLM engine path; and
(b) layer 0 is a **hash-MoE** layer (`num_hash_layers=3`, `gate.tid2eid` present)
— now supported by `TTFusedMoE` (§3.4) and demonstrated running (real weights on
CPU; the reimplemented routing on the TT device). The single-layer attention run
needs only shard 2 (~3.5 GB). Gotcha: run under `torch.no_grad()`, not
`torch.inference_mode()` (inference tensors trip torch_xla on the `freqs_cis`
slice: "Cannot set version_counter for inference tensor").

---

## 4. Later attention milestones (compressed branch, indexer, merge)

Each has its own test stub target in
`tests/.../oot_backends/test_dsv4_swa_attention_impl.py` naming, and a TODO in
`attention_dsv4.py`.

1. **C128A (`compress_ratio == 128`)** — a contiguous compressed-prefix branch of
   length `(pos + 1) // ratio` plus the window branch. The compressed branch is
   *dense* MLA attention over the prefix — reuse `tt.flash_mla_prefill` /
   `tt.paged_flash_mla_decode`, **no sparse kernel needed**. Add a second KV cache
   group for the compressed latent (`MLAAttentionSpec` with `compress_ratio=128`,
   `storage_block_size = block_size // 128`). Then §5 merge.

2. **C4A (`compress_ratio == 4`)** — the lightning-indexer top-k branch.
   **Blocker: the pinned tt-metal ships NO sparse SDPA kernel** — `sparse_sdpa`
   does not exist (the design doc's claim is false at this pin; there is no
   `sparse`* transformer op at all). The indexer top-k selection can be composed
   (`ReplicatedLinear` + RoPE + score matmul + a `topk`), but the sparse gather +
   attention over top-k slots needs either **a new tt-metal kernel** or a
   gather-then-dense fallback (gather the top-k latent slots into a dense
   workspace, then `tt.flash_mla_prefill`). Scope the kernel with tt-metal before
   committing.

3. **Two-branch online-softmax merge** — combine the window-branch output with the
   compressed-branch output. Upstream does it inside a single fused CUDA kernel
   (`flash_mla_with_kvcache` with `extra_k_cache`); TT has no LSE-returning MLA op,
   so either (a) gather both index sets into one workspace and run a single
   attention call (mirrors the GPU *prefill* path — `combine_topk_swa_indices` +
   one `flash_mla_sparse_fwd`), or (b) add an optional `return_lse` to the MLA
   ops and do the `exp(m − m_global)` rescale-and-sum in StableHLO. The sink fold
   already computes an LSE-based factor in the impl — the same machinery extends to
   an inter-branch merge. **Highest-risk numeric piece; prototype against the CPU
   reference in isolation first** (a `test_dsv4_two_branch_merge` with a hand-built
   two-branch case).

4. **fp8 latent cache** (`fp8_ds_mla`, 584 B/token layout), **cache overlay** (page
   sharing between SWA and compressed caches), and **chunked prefill** — all
   optimizations, only after bf16 correctness. The current impl rejects
   `kv_cache_dtype != "auto"` deliberately.

---

## 5. tt-metal notes

* No changes were required for the shipped milestone — the SWA path reuses existing
  ttnn ops (`flash_mla_prefill`, `paged_flash_multi_latent_attention_decode`,
  `paged_fill_cache`, `paged_update_cache`), all already JIT-compiled from source.
* **Attention sink must be bf16** (`sdpa_decode_device_operation.cpp:434`).
* **Prefill sink in-kernel (optional future):** `flash_mla_prefill` has no
  `attention_sink` arg. The impl folds the sink post-hoc via LSE, which is exact,
  so no kernel change is needed. If an in-kernel fold is desired for perf, add
  `attention_sink` to `ttnn::transformer::flash_mla_prefill` (the tt-mlir side
  lowers it to a `ttcore.composite` whose decomposition already does an explicit
  softmax, so the sink can be added at the decomposition level too).
* **No sparse SDPA kernel exists** (see §4.2) — the critical-path dependency for
  the C4A indexer branch.
