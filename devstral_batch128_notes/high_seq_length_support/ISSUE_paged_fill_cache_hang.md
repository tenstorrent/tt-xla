# paged_fill_cache hangs at optimization_level>=1: page_table left TILE-layout (row-major workaround not enabled)

## Summary
On the DP+TP chunked-prefill path, `ttnn.paged_fill_cache` (the paged KV-cache write) hangs the device at
`optimization_level >= 1`. Root cause: the compiler leaves the `page_table` operand in **TILE** layout, but the
tt-metal kernel reads it as **row-major**, so it interprets the int32 block indices as tile-swizzled garbage and
issues writes to invalid DRAM/NoC addresses that never complete. The corrective row-major operand workaround exists
but is gated off at `opt>=1` because `PagedFillCacheOp` (and the chunked/paged SDPA index ops) are missing from the
optimizer's enabled-workaround set.

## Environment
- Model: Devstral-2-123B (any fp8 model on the DP+TP chunked-prefill path)
- Mesh: `[4,8]` (DP=4 × TP=8), Blackhole galaxy (32-chip)
- `optimization_level=1`, `prefill_chunk_size=128` (chunked prefill on), `experimental_kv_cache_dtype=bfp_bf8`
- tt-xla vLLM plugin; tt-mlir around the `#9027` era

## Symptom / repro
- Run a DP+TP chunked-prefill generation at opt1. The device hangs during warmup/execution; tt-metal reports
  `TT_FATAL: Timeout ... waiting for physical cores to finish: 15-3, 15-2` (fixed cores — the CCL/write cores on the
  hanging op).
- With `TT_RUNTIME_SYNC_AFTER_OP=1` the hang pinpoints exactly to `ttnn.paged_fill_cache` in the chunked
  (`prefix_chunk=True`) graph. Without sync it can surface later as an apparent CCL (all_reduce/all_gather) hang —
  those are async-pipeline artifacts; the true stall is the KV write.
- **Arch-specific manifestation:** hangs on Blackhole; Wormhole tolerates the tiled page_table and instead produces
  **silently-corrupt KV** (no hang). The compiler bug is mesh/arch-independent; only the *hang vs silent-corruption*
  symptom is arch-dependent.
- At `optimization_level=0` the issue does not occur (opt0 applies operand workarounds to all ops).

## Root cause
- tt-mlir's base layout pass tilizes **every** op operand unconditionally (`TTNNLayout.cpp:243`, `tiled=true`), so the
  `page_table` becomes TILE regardless of the frontend.
- `PagedFillCacheOp` has a row-major page_table operand workaround
  (`createPagedFillCacheOpOperandsWorkarounds`, `IR/TTNNWorkaroundsPass.cpp:499-521`, forces `page_table` [and
  `batch_idx`] to `Layout::RowMajor`), but the workaround rewriter only applies it to ops in
  `enabledOpsForWorkaroundWithOptimizer` when `optimization_level >= 1`
  (`Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`, the opt-level gate + the static set).
- `PagedFillCacheOp` (and `ChunkedScaledDotProductAttentionOp`, `PagedScaledDotProductAttentionDecodeOp`,
  `PagedFlashMultiLatentAttentionDecodeOp`) were **absent** from that set → their page_table stays TILE at opt>=1 →
  tt-metal `paged_fill_cache` misreads block indices → hang.
- tt-metal `paged_fill_cache` `validate` checks page_table `dtype==INT32` + `INTERLEAVED` but has **no**
  `layout()==ROW_MAJOR` check (`paged_fill_cache_device_operation.cpp:42-44`), so a TILE page_table passes validation
  silently (this is why the op's OpModel doesn't catch it either).

## Fix (what resolved it)
Add the paged-cache + chunked/paged SDPA index ops to `enabledOpsForWorkaroundWithOptimizer` so their row-major
page_table workaround fires at `optimization_level >= 1`:
- `ttnn::PagedFillCacheOp` (the write — the direct cause)
- `ttnn::ChunkedScaledDotProductAttentionOp` (chunked-prefix read)
- `ttnn::PagedScaledDotProductAttentionDecodeOp`, `ttnn::PagedFlashMultiLatentAttentionDecodeOp` (decode reads)

Validated: with these added and the compiler rebuilt, the run gets **past** `paged_fill_cache` (and the chunked SDPA
read) — the hang is gone.

## Notes
- This is an **upstream gap**: the enabled-workaround set is simply missing the paged-cache/SDPA index-op family that
  require a row-major page_table at opt>=1 (mirrors the existing SDPA/TopK/ArgMax entries). Worth landing in tt-mlir
  `main` so submodule uplifts don't drop it.
- Alternative/complementary principled fix: add a `page_table.layout()==ROW_MAJOR` `TT_FATAL` to tt-metal's
  `paged_fill_cache` validate (mirroring `sdpa_device_operation.cpp:216`), which gives the existing OpModel "teeth"
  so the optimizer enforces row-major natively. Caveat: `paged_fill_cache` is a no-result in-place op, so verify the
  optimizer's operand-relayout fallback actually re-lays the operand for such ops (otherwise this converts the hang
  into a clean compile error rather than a fix).
