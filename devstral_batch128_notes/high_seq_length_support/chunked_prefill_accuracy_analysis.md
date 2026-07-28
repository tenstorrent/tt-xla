# Chunked-prefill accuracy error — cause attribution vs `paged_fill_cache` / the tile bug

> **Read-only code + numerics investigation.** Nothing modified, nothing run on hardware.
> Line numbers are the current working tree. Question: is the *expected chunked-prefill accuracy
> error* caused by `paged_fill_cache` / the tile-vs-rowmajor page_table bug we fixed, or is it an
> inherent numerical property of chunked prefill?

## TL;DR verdict

The expected chunked-prefill accuracy error is **dominated by bfp8 (block-float8) quantization of the
KV cache**, exposed through one specific asymmetry: on the **cached-prefix chunk path**, the attention
inputs — *both the prefix and the current chunk* — are read back out of the bfp8 cache, whereas a
single full-sequence prefill attends over the **fresh, never-round-tripped bf16** K/V.

- It is a property of the **bfp8 cache dtype that `paged_fill_cache` writes into** — physically the
  quantization is introduced at the fill (device bf16→bfp8 pack) and compounded at the chunked-SDPA
  read. It is **NOT a defect in the fill logic**, and it is **NOT related to the tile-vs-rowmajor
  page_table bug** we fixed.
- The tile bug was a *wrong-block-index correctness catastrophe* (hang on Blackhole / silent KV
  corruption on Wormhole) — gross, not a small bounded PCC delta. With the row-major fix in place,
  `paged_fill_cache` is **lossless modulo the storage dtype** (it is a pure scatter, no arithmetic).
- The chunking *math* itself is **exact** (the op does one masked softmax over the full
  `[0, chunk_start+chunk_len)` range, not a lossy cross-chunk online-softmax merge), so the only
  residual beyond bfp8 is negligible bf16 accumulation-order rounding.

---

## 1. Where chunked prefill loses precision vs a single full-sequence prefill

### The mechanism (the centerpiece)

In `TTAttentionBackendImpl.forward` the cache write happens **before** attention is computed
(`attention.py:309-328`):

1. `_handle_paged_attention` writes the current chunk's K/V into the paged cache via
   `paged_fill_cache` (`attention.py:500-511`). The on-device cache is **bfp8** (see §KV-dtype
   below), so this write **quantizes** the chunk's K/V.
2. `_compute_full_attention` then branches on `chunked_prefix` (`attention.py:545-560`):
   - **Cached-prefix chunk** (`chunk_start_idx is not None`): calls
     `torch.ops.tt.chunked_scaled_dot_product_attention(query, kv_cache[0], kv_cache[1], page_table,
     chunk_start_idx, …)` — the K and V operands are **the paged cache tensors themselves**
     (`attention.py:551-558`). So this chunk attends over K/V read entirely from the **bfp8 cache**:
     the prefix (written on earlier chunks) *and* the current chunk (just written in step 1).
   - **Standard / non-chunked prefill** (`attention.py:575-579`): attends over
     `inputs.key/inputs.value` — the **fresh bf16** projections, **never round-tripped through the
     cache**.

That asymmetry *is* the expected error: the last chunk of a chunked prefill produces the first
generated token from attention over **bfp8-quantized** prefix K/V, whereas a full-sequence prefill
produces it from **bf16** K/V.

### Ranking the three candidate error sources

**(a) bfp8 KV-cache quantization — DOMINANT.** The KV cache is stored as `bfp_bf8`
(`test_prefill.py:180,271`; `model_runner.py:357-360`). The staged host buffer is bf16 and is
**converted to bfp8 on device** (`model_runner.py:3427-3433`, comment "converted on device"; and
`initialize_kv_cache` allocates the staged buffer at `self.kv_cache_dtype` = bf16). bfp8_b shares one
exponent per block, so small-magnitude elements lose mantissa bits relative to bf16. Every K/V value
the cached-prefix chunk attends over has been through this pack/unpack.

> **Magnitude — directional, verify by experiment; do not assert a number.** bfp8 is normally the
> dominant term and is usually small, but the optimistic textbook read is *not* safe here: per H23
> (`decisions.md:423`, tt-mlir #8140) the on-device bf16→bfp8 **device packer is numerically
> inaccurate** — bad enough that #8140 deliberately moved *weight* packing to the host packer
> (+48% TOP1 regression on gpt-oss-120B with the host packer avoided). The **KV path has no host
> packing option inside the forward loop** — it necessarily eats the device packer. So the true PCC
> hit could be **worse than the textbook bfp8 estimate**. Treat the magnitude as *what the experiment
> in §3 measures*, not a recalled constant.

**(b) The chunked cached-prefix attention math — NOT an approximation.** The op's CPU reference
(`custom_ops.py:1376-1429`, explicitly "Doubles as the equivalence oracle") gathers the **full**
prefix+chunk K/V from the cache and runs a **single** softmax over `[0, chunk_start+chunk_len)` with a
causal+offset mask (`custom_ops.py:1414-1428`). There is **no** cross-chunk online-softmax rescaling
at the Python/op-contract level — each chunk's query sees the entire accumulated prefix in one
softmax. So the chunking scheme reconstructs full attention **exactly**, modulo (a) the dtype of the
K/V it reads and (c) below. (The ttnn kernel uses flash-attention-style block accumulation internally,
which is exact up to floating-point ordering — that is (c), not an algorithmic approximation.)

**(c) Accumulation order / bf16 rounding — negligible.** Flash-attention block ordering and bf16
reduction differ slightly from a monolithic softmax. Real but orders of magnitude below (a).

**Conclusion for §1:** the inherent/"expected" error is **(a) bfp8 KV quantization**, surfaced by the
cached-prefix path reading K/V from the bfp8 cache; (b) contributes ~nothing; (c) is noise.

---

## 2. Is it related to `paged_fill_cache` / the tile bug?

**Not to the tile bug, and not to a fill-logic defect — only to the cache dtype the fill writes into.**

- **The tile bug was a correctness catastrophe, not a bounded error.** Pre-fix, the `page_table`
  operand reached the kernel in TILE layout instead of RowMajor, so `paged_fill_cache` read **wrong
  block indices** → hang (Blackhole) / silent KV corruption (Wormhole)
  (`paged_fill_cache_deepdive.md` context recap; `decisions.md:170,187`). That is gross garbage, not a
  small PCC delta. It is categorically different from the expected bfp8 loss.

- **With the row-major fix in place, `paged_fill_cache` is lossless modulo dtype.** The CPU reference
  (`custom_ops.py:1037-1092`) is a **pure scatter** — index-assignment of `fill_value` into `cache`,
  no arithmetic. The only precision change is the bf16→bfp8 storage conversion, which is a **property
  of the cache dtype**, not of the fill op. So the fill introduces **no additional error beyond the
  bfp8 quantization of the stored K/V**.

- **Could a residual fill issue cause a distinct correctness error?** The residual concerns —
  `fill_page_table` prefix-roll (`attention.py:503`, `model_runner.py` roll region), `batch_idx %
  local_batch` rebase (`attention.py:497-499`), or a wrong page_table — all, if buggy, write to the
  **wrong blocks / wrong user**, producing **gross corruption**, easily told apart from uniform bfp8
  noise.

  **The one residual that could masquerade as a *bounded* error is the partial-final-block write**
  (`custom_ops.py:1084-1090`, `part_of_final_block_to_fill` when a chunk length is not a multiple of
  `block_size`). If the device kernel mishandled the partial tail block, the error would be
  **localized to the token positions at chunk boundaries** — spatially structured — whereas the
  expected bfp8 loss is **uniform across all cached positions**. That signature is how you tell them
  apart (see §3). Note the design largely avoids this: `_chunked_sdpa_active` requires
  `max_num_blocks_per_req % 8 == 0` (`model_runner.py:432-449`) and chunks are sized in block
  multiples, so well-formed runs should have no partial tail block.

**Conclusion for §2:** "expected dtype loss" = the bfp8 KV quantization the fill writes into
(inherent). "A fill_cache bug" = wrong-index corruption (the tile bug, now fixed) or a
partial-block-write defect (bounded, chunk-boundary-localized) — none of which is the expected error.

---

## 3. How to measure/attribute it — the decisive PCC experiment

### Primary experiment — bf16 vs bfp8 KV, everything else fixed, on a long prompt

Toggle **only** the KV cache dtype and hold the chunking config constant. The lever already exists:
`experimental_kv_cache_dtype` selects bfp8 vs the bf16 fall-through (`model_runner.py:357-366`, the
`else` branch keeps `self.kv_cache_dtype` = bf16).

- **Arm A:** `experimental_kv_cache_dtype="bfp_bf8"` (current).
- **Arm B:** `experimental_kv_cache_dtype` unset → bf16 KV.
- **Both arms:** chunked prefill ON, and a **prompt longer than the chunk budget** (128) so chunks
  2..N actually fire the cached-prefix path (`test_prefill.py:51-53`; the multi-chunk path is only
  exercised for prompts > chunk budget). Compare prefill-stage logits / first-token PCC against a
  bf16 full-sequence (chunked-off) reference.

**Attribution:** if Arm B (bf16 KV) recovers PCC to ~full-sequence and Arm A does not, the error is
**bfp8 quantization** — the expected/inherent term. This is the single decisive test.

### Supporting experiments to isolate the other candidates

1. **Chunked vs non-chunked at the *same* KV dtype** (isolates the chunking math, candidate b). Same
   bfp8 KV, chunked-on vs chunked-off, long prompt. If PCC matches, the chunking math is exact and (b)
   is ruled out — consistent with the CPU-oracle read of `custom_ops.py:1376-1429`.
2. **Single-chunk vs multi-chunk** (isolates the cached-prefix read). A prompt that fits in one chunk
   never sets `chunk_start_idx`, so the chunked op never fires; a longer prompt exercises chunks
   2..N. Error appearing only in the multi-chunk case confirms the cached-prefix bfp8 read is the
   locus.
3. **Error-signature check** (separates bfp8 from a partial-block-write bug, §2): plot per-token error.
   **Uniform** across cached positions → bfp8. **Localized at chunk-boundary token positions** →
   suspect the partial-final-block write (`custom_ops.py:1084-1090`).

### Short-prompt corollary — the attribution key

For test prompts **< 128 tokens** the multi-chunk cached-prefix path **is never invoked at runtime**:
`chunk_start_idx` stays `None` (`model_runner.py:1461-1475`, set only when `num_computed > 0` and
`_chunked_sdpa_active`), so `chunked_scaled_dot_product_attention` is never called and prefill takes
the standard fresh-bf16 SDPA path. Consequently, for a short prompt **chunked-on and chunked-off are
bit-identical in prefill**. Any accuracy error observed there is therefore:

- **NOT** from multi-chunk attention (that code didn't run), and
- attributable to the **decode-side** bfp8 KV read (present in *both* chunked and non-chunked runs
  because they share the same bfp8 cache), i.e. common-mode, not chunk-specific.

This is why a short-prompt "chunked prefill accuracy" complaint cannot be the multi-chunk path — and
why the primary experiment must use a **long** prompt.

---

## 4. Verdict

The expected chunked-prefill accuracy error is **dominated by bfp8 (block-float8) quantization of the
KV cache**, surfaced by the cached-prefix chunk attending over K/V read back from the bfp8 cache
(`attention.py:551-558`) — both the prefix and the current chunk — versus a full-sequence prefill
attending over fresh bf16 K/V (`attention.py:575-579`). The magnitude is directionally
bfp8-dominant and, because the KV write is forced through the numerically weaker **device** bfp8
packer (H23, `decisions.md:423`, tt-mlir #8140), potentially larger than the textbook bfp8 estimate —
the §3 experiment is what pins the number.

It is a property of the **bfp8 cache dtype that `paged_fill_cache` writes into**, physically introduced
at the fill (device pack) and compounded at the chunked-SDPA read. It is:

- **NOT related to the tile-vs-rowmajor page_table bug** — that was a wrong-block-index correctness
  catastrophe (hang / silent corruption), now fixed; the fix makes the fill read/write correct blocks
  but changes nothing about the dtype.
- **NOT a fill-logic defect** — with the fix in place `paged_fill_cache` is a pure scatter, lossless
  modulo the storage dtype. (The one fill defect that *could* look like a bounded error — a
  partial-final-block write — is distinguishable by its chunk-boundary-localized signature and is
  avoided by the block-aligned chunking gate.)
- **The chunking math is exact** — a single masked softmax over the full range, not a lossy online
  merge — so it adds no algorithmic approximation beyond bf16 accumulation noise.

**Recommended confirming experiment:** rerun the chunked-on Devstral DP+TP prefill on a prompt longer
than the chunk budget with `experimental_kv_cache_dtype` bf16 vs `bfp_bf8`, all else fixed, and
compare first-token PCC against a bf16 full-sequence reference. bf16 KV recovering PCC ⇒ the error is
the expected bfp8 quantization, independent of the fill op and the tile bug.

---

## File:line index

- `integrations/vllm_plugin/vllm_tt/attention_impls/attention.py`: fill-before-attend order 309-328;
  prefill `paged_fill_cache` + `batch_idx % local_batch` 490-511; `chunked_prefix` branch reading
  `kv_cache[0]/[1]` 545-560; standard prefill over fresh bf16 `inputs.key/value` 575-579;
  identity-preserving `copy_` 513-515.
- `python_package/tt_torch/custom_ops.py`: `paged_fill_cache` pure-scatter CPU ref 1019-1092 (partial
  final block 1084-1090); `chunked_scaled_dot_product_attention` exact single-softmax oracle
  1339-1429.
- `integrations/vllm_plugin/vllm_tt/model_runner.py`: KV dtype / bfp8 staging + on-device convert
  357-366, 3427-3433; `_chunked_sdpa_active` gate (block%8, chunk<max_model_len) 432-449;
  `chunk_start_idx` set only on cached-prefix chunk 1461-1475; `initialize_kv_cache` 3331-3444.
- `tests/integrations/vllm_plugin/generative/test_prefill.py`: `experimental_kv_cache_dtype="bfp_bf8"`
  179-180, 270-271; long-prompt-needed-for-multi-chunk note 51-53.
- `devstral_batch128_notes/high_seq_length_support/decisions.md`: device bfp8 packer inaccuracy / host
  packer #8140 (H23) 423; tile-bug garbage block addresses 170,187; this analysis referenced as
  separate question 446.
- `devstral_batch128_notes/high_seq_length_support/paged_fill_cache_deepdive.md`: tile-bug root cause
  recap + fix-makes-both-correct.
