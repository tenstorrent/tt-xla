# tt-mlir changes required for DeepSeek Sparse Attention (DSA) in the vLLM plugin

This file records every change made to the **tt-mlir** submodule while bringing
DeepSeek Sparse Attention up end-to-end in the tt-xla vLLM plugin. It exists so
the changes can be recreated as a PR against the real
[tt-mlir](https://github.com/tenstorrent/tt-mlir) repository — the tt-xla
submodule pin is not a place to land them permanently.

* Submodule path: `third_party/tt-mlir/src/tt-mlir`
* Base branch / commit the changes were made on: `hshah/all-dsa-ops` @ `7d4fe61b98`
* Files touched: 4 (1 source, 3 tests)
* A ready-to-apply patch of exactly these changes is reproduced verbatim in
  [Appendix A](#appendix-a-complete-patch).

---

## Change 1 — `sparse_sdpa` decomposition: build the sparsity mask with a scatter

### Status

Implemented and verified. **This is the only functional change.** The other three
files are the FileCheck tests that assert on the ops it replaces.

### Problem

`tt.sparse_sdpa` lowers to a `ttcore.composite` whose decomposition is inlined on
any non-Blackhole target (the TTNN kernel is Blackhole-only). That decomposition
derived its sparsity mask by materialising a **one-hot slot tensor**: broadcast the
`indices` operand to `[B, S, TOPK, T]`, compare it elementwise against an `arange`
of key positions, then `sum` the slot axis away to get a per-`(query, key)` hit
count.

That intermediate is `O(S · TOPK · T)`. At DeepSeek-V3.2 prefill shapes it is
catastrophic:

| `S` = `T` = `TOPK` | one-hot `[B,S,TOPK,T]` | dense score `[B,H,S,T]`, `H`=128 |
| --- | --- | --- |
| 256 | 16.8 M elements (67 MB f32) | 8.4 M |
| 1024 | **1.07 G elements (4.3 GB f32)** | 134 M |

Observed symptom: a 3-layer DeepSeek-V3.2 vLLM run with a 1024-token prefill on
Wormhole (8-chip llmbox) compiled for **49 minutes without completing**, versus
~3 minutes for the same model with DSA disabled. The one-hot tensor was ~32× the
size of the dense score tensor the decomposition already needs, making it the sole
reason the fallback path could not reach production shapes.

### Fix

Replace the one-hot compare-and-reduce with a single `ttir.scatter` accumulate into
a `[B, S, T]` hit-count buffer. That is `O(S · T)` — 1.05 M elements at
`S = T = 1024`, a ~1000× reduction — and leaves the `[B, H, S, T]` score tensor as
the peak allocation, i.e. the same footprint as ordinary dense attention.

`ttir.scatter` → `ttnn.scatter` → `ttnn::scatter(..., "add")` already exists
end-to-end; no new ops, flatbuffer schema, or runtime support are required:

* op: `TTIR_ScatterOp` (`include/ttmlir/Dialect/TTIR/IR/TTIROps.td`)
* lowering: `ScatterOpConversionPattern` (`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`)
* runtime: `runtime/lib/ttnn/operations/data_movement/scatter.cpp` maps
  `ReduceType::Sum` → `"add"`

### Location

* File: `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
* Function: `static Value buildSparseSdpaDecompositionBody(...)`
* The replaced region is the block between the scaled-scores computation and the
  additive `0`/`-inf` mask construction — i.e. everything that computed `visible`.

### Two correctness details that must be preserved

These are the parts most likely to be broken by a well-meaning simplification, so
they are called out explicitly.

**1. The reduction must be `SUM`, not an overwrite.**

Masked slots carry the `0xFFFFFFFF` sentinel, which is out of range for a scatter
into a `T`-wide axis. They are therefore redirected onto key `0` while carrying a
**zero** contribution. With `SUM`, a sentinel adds `0.0` and is harmless. With an
overwrite (`ReduceType::Invalid` / plain `scatter`), the winner among duplicate
destinations is unspecified, so a sentinel slot could clear a *genuine* hit on
key `0` in the same row — silently dropping a selected key.

**2. Both bounds must be tested, not just the upper one.**

The sentinel reaches the decomposition in two different guises depending on the
producer:

* `~4.29e9` when `indices` is `ui32` (the Blackhole `tt.topk_large_indices` kernel
  output passed straight through), and
* `-1` when `indices` is `si32` (tt-xla's non-Blackhole repair path — see
  `topk_large_indices_mask_invalid_slots` in `python_package/tt_torch/custom_ops.py`,
  which cannot construct a `uint32` sentinel inside a `torch.compile` graph).

A `< T` test alone accepts `-1` and scatters at a negative index. The predicate is
therefore `(idx >= 0) && (idx < T)`. The previous one-hot formulation was
accidentally immune to this, because `-1.0` never equals any `keyPos` in `[0, T)`.

### Op sequence after the change

Index arithmetic stays in **f32** (as before): `bf16` only represents integers
exactly up to 256, far below realistic `T`, so comparing key positions in `bf16`
would conflate distinct keys. f32 is exact to 2²⁴, which bounds `T`.

```
indices [B,1,S,TOPK] (ui32|si32)
  ttir.typecast            -> [B,1,S,TOPK] f32
  ttir.reshape             -> [B,S,TOPK]   f32          (slotType)

  ttir.full 0.0 / 1.0 / (float)T           -> [B,S,TOPK] f32
  ttir.lt   (idx, T)                       -> [B,S,TOPK] f32   0/1
  ttir.ge   (idx, 0)                       -> [B,S,TOPK] f32   0/1
  ttir.multiply (lt, ge)                   -> validSlot
  ttir.where (validSlot, 1.0, 0.0)         -> scatterSource     f32
  ttir.where (validSlot, idx, 0.0)         -> safeIndex          f32
  ttir.typecast                            -> [B,S,TOPK] i32    (scatter index)

  ttir.full 0.0                            -> [B,S,T] f32       (hitType)
  ttir.scatter(dim=2, reduce=sum)          -> [B,S,T] f32       hit counts
  ttir.reshape                             -> [B,1,S,T] f32
  ttir.gt (hits, 0.0)                      -> visible
```

`visible` then feeds the pre-existing additive `0`/`-inf` mask + broadcast-over-heads
code, which is unchanged.

### Notes for the reviewer

* **Index dtype.** `ttir.scatter`'s verifier
  (`ScatterOp::verify`, `lib/Dialect/TTIR/IR/TTIROps.cpp`) checks only that
  input/index/source have equal rank and that `index.shape == source.shape`; it
  does **not** constrain the index element type. The TTNN side does assume an
  integer index — `createScatterOpOperandsWorkarounds`
  (`lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`) states *"The index tensor is
  always integer-typed"* and forces int32 indices to row-major to dodge tt-metal's
  256-element tiled-scatter-axis limit. Hence the explicit f32 → i32 typecast. The
  cast is exact: every surviving value is an in-range key position.
* **A new `intIndexTensorType` lambda** was added next to the existing
  `indexTensorType` lambda, differing only in element type (`i32` vs `f32`).
* **Rank change.** The mask tensors are now rank 3 (`[B,S,TOPK]`, `[B,S,T]`) where
  they used to be rank 4, because `ttir.scatter` requires input, index and source
  to have equal rank. The reshape back to `[B,1,S,T]` happens after the scatter.
* **`TOPK > T` is permitted** by torch scatter semantics along the scatter dim, and
  `TTNN_SparseSdpaOp`'s verifier does not require `TOPK <= T`. It does not arise in
  practice (you cannot select more keys than exist) and was not tested.
* This changes only the **decomposition**. The `ttcore.composite`, its promotion
  guard, `TTNN_SparseSdpaOp` and the Blackhole kernel path are all untouched, so
  Blackhole codegen is bit-identical to before.

---

## Change 2 — update the three FileCheck tests that assert on the removed ops

Mechanical, but required: the old tests assert `ttir.eq` / `ttir.arange` /
`ttnn.eq` / `ttnn.arange` / `ttnn.sum`, none of which the new decomposition emits.

### 2a. `test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir`

TTIR-level (pre-canonicalisation) assertions on the private
`@sparse_sdpa_decomp` function.

* Removed: `CHECK-DAG: "ttir.arange"{{.*}} -> tensor<{{.*}}xf32>` and
  `CHECK-DAG: "ttir.eq"{{.*}}`.
* Added: `CHECK-DAG` for `"ttir.lt"`, `"ttir.ge"`, and
  `"ttir.scatter"{{.*}}scatter_reduce_type = #ttcore.reduce_type<sum>`.
* Added a **`CHECK-NOT: "ttir.eq"`** guard so the one-hot formulation cannot
  silently return.
* Comment updated to explain the memory rationale and the `sum` requirement.

### 2b & 2c. `test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir` and `..._bsz.mlir`

End-to-end decomposition assertions after `--ttir-to-ttnn-backend-pipeline`
(`_wh` = Wormhole target, `_bsz` = Blackhole with batch > 1; both exercise the
decomposition rather than the kernel).

* Removed: `CHECK-DAG` for `"ttnn.arange"`, `"ttnn.eq"`, `"ttnn.sum"`.
* Added: `CHECK-DAG` for `"ttnn.ge"` and `"ttnn.scatter"`.
* **Only `ge` is checked, not `lt`.** At the TTNN level the upper-bound test is
  canonicalised into a second `ttnn.gt` with swapped operands, so `ttnn.lt` does
  not appear in the output. The existing `CHECK-DAG: "ttnn.gt"` covers it. This
  cost one debugging cycle; the comment in the test records it.
* The comment about *"the key-position arange depends only on shapes"* (explaining
  why ops may be hoisted into a const-eval function) was reworded, since there is
  no longer an arange.

---

## Verification performed

Environment: 8-chip **Wormhole_b0** llmbox (`n300-llmbox` class). Blackhole was not
available, so the kernel-promotion path was verified only via `mock-system-desc-arch`
and not on silicon.

### Build

```bash
cd third_party/tt-mlir/src/tt-mlir
source env/activate
cmake --build build --target ttmlir-opt -j 16     # incremental
```

### Generated IR inspected by hand

The new decomposition for
`q [1,32,32,64] bf16, kv [1,1,64,64] bf16, idx [1,1,32,32] ui32, v_dim=32`
contains exactly one scatter and no rank-4 slot tensor:

```mlir
%16 = "ttir.typecast"(%15) : (tensor<1x32x32xf32>) -> tensor<1x32x32xi32>
%17 = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32: 1, 32, 64>}> : () -> tensor<1x32x64xf32>
%18 = "ttir.scatter"(%17, %16, %14) <{dim = 2 : i32, scatter_reduce_type = #ttcore.reduce_type<sum>}>
    : (tensor<1x32x64xf32>, tensor<1x32x32xi32>, tensor<1x32x32xf32>) -> tensor<1x32x64xf32>
```

### End-to-end evidence at the shape that motivated the change

The vLLM E2E test at 3-layer DeepSeek-V3.2, 1024-token prefill,
`index_topk = 1024`, TP mesh `[2,4]`, Wormhole llmbox now **converts all its graphs
to `ttnn_runtime` MLIR in ~20 s**, where before the change nothing completed.
Confirmed from the run's exported MLIR (`additional_config["export_path"]`):

| Check | Result |
| --- | --- |
| `tt.indexer_score_dsa` / `tt.topk_large_indices` / `tt.sparse_sdpa` in `shlo_*` | 3 each (one per layer) |
| `ttir.scatter` in `ttir_*` | 3 (one per layer) |
| `ttir.eq` in `ttir_*` | 0 — the one-hot form is gone |
| any `tensor<1x1024x1024x1024x...>` | none — the 4.3 GB intermediate is eliminated |
| graphs reaching `ttnn_runtime_*` | 12 (previously: none completed) |

A standalone `tt.sparse_sdpa` device call also matches its CPU golden at
**PCC 0.9998**, and the tt-xla DSA suite is unchanged at 68 passed / 1 skipped
(42 op-level in `tests/torch/ops/test_dsa_ops*.py`, 26 plugin-level in
`tests/integrations/vllm_plugin/oot_backends/`).

**Scope of this claim.** The change removes the memory blowup and lets the graphs
compile; it does **not** make that E2E test pass. Execution still stalls after graph
conversion (host parks in
`tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue`), and this
reproduces at both 256 and 1024 token lengths, so it is a separate,
shape-independent problem in the DSA sparse path rather than anything to do with the
decomposition's memory footprint. `dsa_mode="off"` on the same model and config
completes in ~3 minutes. The tt-xla test is committed `@pytest.mark.skip` with that
diagnostic; see its skip reason for the suggested next bisect step. Nothing in this
tt-mlir change depends on that being resolved — the lit tests, the golden-PCC check
and the memory reduction stand on their own.

### Lit tests — all 7 `sparse_sdpa` tests pass

`llvm-lit` could not be driven directly in this tree (it needs the CMake-generated
`lit.site.cfg`), so each test's `RUN` lines were executed manually:

| Test | Result |
| --- | --- |
| `Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir` | PASS (updated) |
| `Conversion/StableHLOToTTIR/transformer/sparse_sdpa_negative.mlir` | PASS |
| `Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir` | PASS (updated) |
| `Dialect/TTNN/transformer/sparse_sdpa_decomposition_bsz.mlir` | PASS (updated) |
| `Dialect/TTNN/transformer/sparse_sdpa_positive.mlir` | PASS |
| `Dialect/TTNN/transformer/sparse_sdpa_arch_fallback.mlir` | PASS |
| `Dialect/TTNN/transformer/sparse_sdpa_batch_fallback.mlir` | PASS |

Reproduce with e.g.:

```bash
cd third_party/tt-mlir/src/tt-mlir
build/bin/ttmlir-opt --stablehlo-to-ttir-pipeline \
  test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir -o /tmp/o.mlir
FileCheck test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir --input-file=/tmp/o.mlir

build/bin/ttmlir-opt --stablehlo-to-ttir-pipeline \
  test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir -o /tmp/a.mlir
build/bin/ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0" \
  /tmp/a.mlir -o /tmp/b.mlir
FileCheck test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir \
  --input-file=/tmp/b.mlir --implicit-check-not="ttnn.sparse_sdpa"
```

### Recommended additional verification before landing upstream

Not run here, and worth doing in the tt-mlir repo where the harness is set up:

1. `cmake --build build --target check-ttmlir` — the full lit suite, to catch any
   other test asserting on `sparse_sdpa`'s decomposition.
2. `pytest test/python/golden/test_stablehlo_ops.py -k sparse_sdpa` — the numeric
   golden test (`test/python/golden/test_stablehlo_ops.py:2670`), which compares
   against `sparse_sdpa_golden` in `tools/golden/mapping.py`. This is the strongest
   available numeric check of the decomposition and needs
   `ttrt query --save-artifacts` + `SYSTEM_DESC_PATH` first.
3. A **sentinel-bearing** golden case, if one does not already exist: the existing
   golden test appears to use fully-valid index rows, which would not exercise the
   two correctness details above. A row whose tail is `0xFFFFFFFF` *and* a row that
   legitimately selects key `0` in the same tensor is the case that distinguishes
   `SUM` from an overwrite.
4. An **si32-index** case, covering tt-xla's `-1` sentinel, which is what makes the
   `>= 0` half of the predicate load-bearing.

---

## Related tt-mlir / tt-metal gaps found while doing this work

Not fixed here; each is independently actionable and each blocks either
performance or correctness of DSA on TT. Listed roughly by value.

### G1. `TTNN_SparseSdpaOp` does not expose `cache_batch_idx` (performance, high value)

`ttnn::transformer::sparse_sdpa` already accepts `cache_batch_idx`, documented in
`ttnn/cpp/ttnn/operations/transformer/sdpa/sparse_sdpa.hpp` as:

> when set, kv is a shared `[B, 1, T, K_DIM]` cache and this selects the batch slot
> to attend to (indices are page ids within that slot) … The value is a dynamic
> runtime arg, so changing the slot (or the cache length T) does not recompile the
> kernels.

`TTNN_SparseSdpaOp` (`include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`) exposes only
`v_dim`, `scale`, `k_chunk_size`, and
`runtime/lib/ttnn/operations/transformer/sparse_sdpa.cpp` hardcodes
`cache_batch_idx = std::nullopt`. Its verifier also requires `kv` batch == 1.

Consequence in tt-xla: DSA **decode** cannot read the paged latent KV cache
directly, so `TTMLAAttentionBackendImpl._forward_decode_sparse` gathers each user's
cache into a dense buffer every step. The attention arithmetic is `O(top-k)` but the
memory traffic is `O(context)`, making sparse decode *slower* than dense decode.

> ⚠️ **CORRECTION (verified against tt-metal `f1f4ff75579`).** The paragraph that
> used to close this section claimed exposing `cache_batch_idx` "removes the gather
> entirely and is the single change that makes DSA decode fast". **That is wrong** —
> exposing it is necessary-at-best and on a paged cache buys nothing on its own. Two
> constraints block it, neither in tt-mlir:
>
> 1. **`cache_batch_idx` assumes a batch-CONTIGUOUS cache.**
>    `sparse_sdpa_device_operation.cpp` computes
>    `kv_batch_page_offset = cache_batch_idx * T`, i.e. slot `b` must occupy rows
>    `[b*T, (b+1)*T)`. vLLM's cache is paged and a request's blocks are deliberately
>    scattered, so no flat per-slot offset can describe it.
> 2. **A layout conflict that cannot be reconciled by exposing anything.**
>    `sparse_sdpa` requires `TT_FATAL(kv.layout() == Layout::ROW_MAJOR)`, while
>    `update_cache_device_operation.cpp:32` requires
>    `input.layout() == TILE && cache.layout() == TILE`. The same tensor cannot be
>    both, and converting per step is `O(cache)` -- strictly worse than the
>    `O(context)` gather being removed. The tt-mlir ROW_MAJOR operand workaround
>    therefore reflects the kernel faithfully; removing it just moves the failure.
>
> The idea underneath G1 is still right -- stop materializing the context -- but the
> enabler is a **tt-metal capability**, not a tt-mlir attribute. See
> [`dsa_blackhole_tt-metal_changes.md`](./dsa_blackhole_tt-metal_changes.md) §2.8 for
> the concrete ask (make `sparse_sdpa` paged-aware like its dense sibling
> `paged_flash_mla_decode` already is) and for the index-translation trick that
> supplies the `O(top-k)` half once the layout question is settled.

The traffic analysis in the addendum below still holds as *arithmetic* -- what changes
is which layer has to deliver it.

### G2. `paged_flash_multi_latent_attention_decode` cannot be masked (correctness, upstream tt-metal)

`ttnn::prim::sdpa_decode` hard-asserts `is_causal`
(`sdpa_decode_device_operation.cpp:28`, *"Multi-latent attention decode only tested
for causal!"*). A non-causal lowering therefore aborts the process with a `TT_FATAL`
rather than honouring the mask.

This ruled out the otherwise-natural DSA decode implementation (restrict the paged
kernel to the selected keys with an additive `-inf` mask) and forced the gather
approach in G1. tt-xla now raises a Python-level `NotImplementedError` for
`is_causal=False` on device so the abort cannot happen silently, but note that
`tt.paged_flash_mla_decode` still *advertises* an `attn_mask` operand that no
Blackhole/Wormhole path can actually execute.

### G1/G2 addendum — mask vs gather, concretely

G2 calls the mask "otherwise-natural" without quantifying it, which invites the wrong
conclusion. The mask approach is **not** asymptotically better than the gather: both
are `O(context)`. Only G1 breaks that.

The crux is that a masked paged decode still reads the *entire* cache.
`sdpa_flash_decode.cpp:344-350` fuses the mask as an **add** onto the scores per
K-chunk (`add_mask_fusion`); there is no "this chunk is all `-inf`, skip it" path.
The chunk range comes from `cur_pos` (causal masking is applied only at the last
chunk, `apply_mask_at_last_chunk`), and the sole early exit
(`reader_decode_all.cpp:181`) is for cores with no assigned work, not mask-driven.
Masked-out keys are read, multiplied, and then discarded. Chunk-skipping would not
rescue it either: DSA's selected tokens are scattered, not clustered, so almost every
chunk retains at least one live key.

Per layer, per decode step, per user. DeepSeek-V3.2 latent width `L+R = 576`, bf16:

| approach | traffic | T=4096, k=2048 | T=8192, k=2048 |
|---|---|---|---|
| gather (current) | `2·T·576·2` (paged read + dense write) + `k·576·2` (sparse read) | 11.25 MiB | 20.25 MiB |
| additive `-inf` mask | `T·576·2` (in-place paged read) + `≈users·T·2` (mask) | 4.50 MiB | 9.00 MiB |
| `cache_batch_idx` (G1) | `k·576·2`, **flat in T** | 2.25 MiB | 2.25 MiB |

The gather pays for the context twice — once reading the paged cache, once writing
the dense staging buffer — before `sparse_sdpa` reads it a third time. The mask pays
once; building the mask is `O(T)` writes on a 1-wide vector rather than a 576-wide
one, i.e. ~1/576th the cost of the copy it replaces.

So the mask would have been worth having, on three counts:

1. **2-2.5x less traffic** — a constant factor, not a change in asymptotics. 2.5x at
   `T=4096`, decaying towards 2x as context grows (the gather's fixed `k·576·2` term
   shrinks in proportion, leaving the 2x read+write penalty).
2. **It batches over users; the gather cannot.** `tt.sparse_sdpa` requires
   `B == 1`, so `_forward_decode_sparse` runs a Python `for u in range(users)` loop:
   N gathers, N kernel calls and a `cat`, all unrolled into the traced graph.
   `tt.paged_flash_mla_decode` takes `query [1, num_users, nqh, dh_qk]` with
   `page_table [num_users, …]` and a mask broadcasting as
   `[num_users, nqh, 1, max_seq_len]` — one call regardless of batch size.
3. **Far smaller diff** — it reuses the already-validated dense paged decode path
   instead of introducing a staging buffer and a per-user loop.

None of that makes it the fix. Decode cost would still grow linearly with context.
**If only one of G1/G2 is landed, land G1** — and landing G1 makes G2 moot, because
`sparse_sdpa` reading the paged cache directly needs no mask at all.

That the kernel path is genuinely `O(top-k)` is confirmed in
`sparse_sdpa_reader.cpp:94`:

> binary-search the first sentinel -> nv (valid-key count); only `ceil(nv/k_chunk)`
> chunks are active.

This is also why the contiguous-sentinel-tail contract is load-bearing rather than
cosmetic: the reader binary-searches for the *first* sentinel, so a non-contiguous
tail truncates the valid-key count and silently drops real keys.

**Arch caveat.** G1's benefit is Blackhole-only, and that is a hard tt-metal
constraint rather than a tt-mlir policy choice — all three DSA kernels `TT_FATAL` on
arch: `sparse_sdpa_device_operation.cpp:73`,
`topk_large_indices_device_operation.cpp:31`, and
`indexer_score_device_operation.cpp:235` (reason at :232 — "the compute kernel relies
on BH fast-untilize + custom BH LLK paths"). tt-mlir's `requireBlackhole` promotion
guards mirror these faithfully. Nothing is lost by that: on Wormhole the
`sparse_sdpa` composite inlines a **dense** decomposition (full `[B,H,S,T]` scores,
then mask), so DSA there is strictly more expensive than plain dense MLA with or
without G1 — Wormhole is a correctness/bring-up vehicle, never a perf target. What
G1 would change on Wormhole is only code organisation: honouring `cache_batch_idx`
would move the gather from tt-xla Python into the tt-mlir decomposition, same
traffic but one implementation instead of two, and tt-xla's decode path sheds its
per-user loop on both architectures.

### G3. `TTNN_TopKLargeIndicesOp` does not expose `valid_length` (performance / simplicity)

tt-metal's `topk_large_indices` has a `valid_length` attribute
(`topk_large_indices_device_operation_types.hpp`):

> Restrict the search to the first `valid_length` columns of each row … Runtime-only
> (hash-excluded, validated on cache hit) so a serving loop growing valid_length
> reuses one program.

That is precisely DSA decode's causal bound. `TTNN_TopKLargeIndicesOp` exposes only
`k`, so tt-xla instead adds an explicit additive `-inf` mask over the full bucket
width before the top-k. Exposing it would remove that mask and the associated
`arange`/`where`.

### G4. `indexer_score_dsa` has no operand workarounds (robustness)

Every sibling op in this family has a `TTNNOperandsWorkaroundsFactory` entry that
coerces layout/dtype (`sparse_sdpa` forces ROW_MAJOR/DRAM/bf16/uint32;
`topk_large_indices` forces ROW_MAJOR + bf16/uint32). `TTNN_IndexerScoreDsaOp` has
none, so nothing inserts a cast if a caller supplies a non-bf16 operand — it fails
inside the kernel instead of degrading. tt-xla compensates with a hard `bfloat16`
assert in the `tt::indexer_score_dsa` wrapper, but a workaround entry would make the
op behave like its siblings.

### G5. `ttir.paged_update_cache` fails to lower when `fill_value` is a bare graph input (bug)

Reproduced on Wormhole with `cache [4,1,32,576] bf16`, `fill [1,1,1,576] bf16`,
`cache_position [1] i32`, `page_table [1,4] i32`:

* `fill_value` produced by a computation (e.g. `cat` + `transpose`) → compiles.
* the identical op with `fill_value` as a function argument → `Failed to run
  TTIRToTTNNCommon pipeline`, PJRT `Error code: 13`.

The emitted TTIR is byte-identical in both cases, so this is a layout/const-eval
issue on graph-input operands rather than anything about the op's semantics. It does
not affect production (the real indexer always computes its keys) but it makes the
op awkward to unit-test in isolation; tt-xla's indexer test works around it by
building `k_op` with an on-device `cat`.

### G6. Not tt-mlir: `MLAAttentionSpec.merge` in vLLM does not validate equality

Recorded here because it was found in the same bring-up and is the most damaging of
the set. `vllm/v1/kv_cache_interface.py::MLAAttentionSpec.merge` checks only
`cache_dtype_str` / `compress_ratio` / `model_version` and then takes
`specs[0].head_size`, unlike `FullAttentionSpec.merge` which asserts every
`AttentionSpec` field matches. With DeepSeek-V3.2's 576-wide MLA specs and the
128-wide indexer spec in one model, `is_kv_cache_spec_uniform` therefore returns
`True` and **every layer silently receives a 128-wide cache**, surfacing far away as
`ttir.paged_fill_cache: Input must have same head dimension as cache`. Belongs in a
vLLM PR.

### G7. Decomposition memory scales with head count — DSA cannot run at production `index_topk` on Wormhole (performance / capacity)

**Symptom.** `test_tensor_parallel_generation_deepseek_v32_3l[sparse-topk2048]`
(3 layers, `[2, 4]` mesh, stock `index_topk=2048`, 2106-token prompt) dies during
prefill execution:

```
Out of Memory: Not enough space to allocate 34359738368 B (32 GiB) DRAM buffer
across 12 banks, where each bank needs to store 2863312896 B, but bank size is
1070773184 B          [bank_manager.cpp:462; device DRAM = 12 GiB]
```

Production `index_topk` requires a prefill bucket >= 2048, and `min_context_len` is
rounded up to a power of two (`model_runner.py::_adjust_min_token`), so the smallest
bucket holding a >2048-token prompt is **4096**. The param is currently skipped with
this reason.

**Isolation.** Each op run standalone on Wormhole at the production shape:

| op | shape | result |
| --- | --- | --- |
| `tt.topk_large_indices` | `[1,1,4096,4096]`, k=2048 | OK |
| `tt.indexer_score_dsa` | q `[1,64,4096,128]`, k `[1,1,4096,128]` | OK (2 GiB intermediate, fits) |
| `tt.sparse_sdpa` | q `[1,32,4096,576]` (per-device) | OK |
| `tt.sparse_sdpa` | q `[1,128,4096,576]` (global) | **OOM** |

So the driver is `sparse_sdpa`'s decomposition scaling linearly in head count via its
`[1, H, S, T]` score tensor. Largest tensors in the exported TTIR are
`[1,64,4096,4096]` bf16 (indexer, replicated Hi=64) and `[1,32,4096,4096]` bf16
(sparse_sdpa, sharded) — 2 GiB and 1 GiB.

**Not fully explained.** The model's TTIR already shows `sparse_sdpa` sharded to
H=32, the shape that passes standalone, and no 32 GiB tensor appears at any exported
stage (TTIR, TTNN, ttnn_runtime all top out at 2 GiB / 0.43 GiB). So the in-model
failure is most likely cumulative pressure — several 1-2 GiB intermediates live at
once across 3 layers — rather than one oversized op. Confirming that is the first
step next time.

**Fixes, cheapest first.**

1. *tt-xla, exact:* query-row-chunk the indexer. `tt.indexer_score_dsa` already takes
   `chunk_start_idx` (masks `t > chunk_start_idx + s`), which is precisely "score a
   slice of query rows against the full key" — see
   `test/ttmlir/Conversion/StableHLOToTTIR/transformer/indexer_score_dsa.mlir`'s
   `indexer_score_dsa_chunked`. Chunking `Sq` into C-row blocks turns `[1,64,S,T]`
   into `[1,64,C,T]` and shrinks the top-k input rows too. `_select`'s
   `visible_count` must be offset by the chunk start.
2. *tt-xla, exact:* head-chunk `tt.sparse_sdpa`. Heads are independent, so
   `cat([sparse_sdpa(q[:, h0:h1], kv, idx) ...], dim=1)` is bit-exact and divides the
   score tensor by the chunk factor. Protects any mesh whose model axis leaves more
   than 32 heads per device. The same row-chunking as (1) applies on the S axis.
3. *tt-mlir, durable:* accumulate over heads inside the decompositions instead of
   materializing `[B, H, S, T]`. `relu` / `softmax` block folding the reduction into
   the matmul, but a chunked accumulate works. Direct precedent: Change 1 above
   already restructured `sparse_sdpa`'s mask into a scatter to kill a
   `[1, S, TOPK, T]` blowup. Fixes every frontend rather than just tt-xla.

None of this affects Blackhole, where all three composites promote to kernels that
stream in chunks and materialize none of these tensors.

---

## Appendix A — complete patch

Apply with `git apply` from the tt-mlir repository root.

<!-- BEGIN PATCH -->
```diff
diff --git a/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp b/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
index 0cf834eacc..4bee0d5b91 100644
--- a/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
+++ b/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
@@ -9823,6 +9823,12 @@ static Value buildSparseSdpaDecompositionBody(
   auto indexTensorType = [&](ArrayRef<int64_t> shape) {
     return RankedTensorType::get(shape, rewriter.getF32Type(), encoding);
   };
+  // ttir.scatter requires an integer-typed index operand (see
+  // TTNNOperandsWorkaroundsFactory::createScatterOpOperandsWorkarounds), so the
+  // f32 slot positions are cast to i32 just before the scatter.
+  auto intIndexTensorType = [&](ArrayRef<int64_t> shape) {
+    return RankedTensorType::get(shape, rewriter.getI32Type(), encoding);
+  };

   // Fold the query heads into the sequence dim so a single batched matmul
   // against kv's single latent head works without broadcasting kv across
@@ -9865,10 +9871,21 @@ static Value buildSparseSdpaDecompositionBody(
       rewriter.create<ttir::MultiplyOp>(loc, scoresType, scores, scaleConst)
           .getResult();

-  // Sparsity mask. `selected[b, s, t] != 0` iff key t appears in
-  // indices[b, 0, s, :]; built by comparing every (query, slot) index against
-  // every key position and reducing the slot axis away.
-  auto slotType = indexTensorType({batch, querySeqLen, topK, keySeqLen});
+  // Sparsity mask. `selected[b, 0, s, t] != 0` iff key t appears in
+  // indices[b, 0, s, :], accumulated with a single scatter into a per-(query,
+  // key) hit-count buffer.
+  //
+  // The direct formulation -- broadcast the indices to [B, S, TOPK, T], compare
+  // against an arange of key positions, and reduce the slot axis away -- needs an
+  // O(S * TOPK * T) intermediate. At DeepSeek-V3.2 prefill shapes
+  // (S = T = TOPK = 1024) that is 1.07e9 elements (~4.3 GB in f32), which does
+  // not fit; scattering needs only O(S * T) (1.05e6 for the same shapes), which
+  // leaves the [B, H, S, T] score tensor as the peak -- i.e. the same footprint
+  // as ordinary dense attention.
+  auto slotType = indexTensorType({batch, querySeqLen, topK});
+  auto slotIndexType = intIndexTensorType({batch, querySeqLen, topK});
+  auto hitType = indexTensorType({batch, querySeqLen, keySeqLen});
+
   Value indicesF32 =
       rewriter
           .create<ttir::TypecastOp>(
@@ -9876,31 +9893,71 @@ static Value buildSparseSdpaDecompositionBody(
           .getResult();
   Value indicesSlot =
       ttir::utils::createReshapeOp(rewriter, loc, indicesF32,
-                                   {batch, querySeqLen, topK, 1})
+                                   {batch, querySeqLen, topK})
           .getResult();
-  Value indicesBcast =
+
+  // Masked slots must not scatter. Producers mark them with the 0xFFFFFFFF
+  // sentinel, which arrives here as ~4.29e9 from a uint32 index tensor or as -1
+  // from an int32 one, so both bounds are tested: a slot counts only when its
+  // index lies in [0, T). Invalid slots are redirected onto key 0 carrying a zero
+  // contribution, which is why the reduction below must be SUM and not an
+  // overwrite -- a genuine hit on key 0 elsewhere in the same row must not be
+  // cleared by a neighbouring sentinel slot.
+  Value slotZeros =
       rewriter
-          .create<ttir::BroadcastOp>(loc, slotType, indicesSlot,
-                                     SmallVector<int64_t>{1, 1, 1, keySeqLen})
+          .create<ttir::FullOp>(loc, slotType, rewriter.getF32FloatAttr(0.0f))
           .getResult();
-  Value keyPos = rewriter
-                     .create<ttir::ArangeOp>(loc, slotType, /*start=*/0,
-                                             /*end=*/keySeqLen, /*step=*/1,
-                                             /*arange_dimension=*/3)
-                     .getResult();
-  Value slotHit =
-      rewriter.create<ttir::EqualOp>(loc, slotType, indicesBcast, keyPos)
+  Value slotOnes =
+      rewriter
+          .create<ttir::FullOp>(loc, slotType, rewriter.getF32FloatAttr(1.0f))
+          .getResult();
+  Value keyLimit =
+      rewriter
+          .create<ttir::FullOp>(
+              loc, slotType,
+              rewriter.getF32FloatAttr(static_cast<float>(keySeqLen)))
+          .getResult();
+  Value belowLimit =
+      rewriter.create<ttir::LessThanOp>(loc, slotType, indicesSlot, keyLimit)
           .getResult();
-  // Sum over the top-k slot axis: [B, S, 1, T] -> [B, 1, S, T]. A key selected
-  // more than once in a row still yields a nonzero count, so the > 0 test below
-  // does not depend on the producer emitting distinct indices.
+  Value nonNegative =
+      rewriter
+          .create<ttir::GreaterEqualOp>(loc, slotType, indicesSlot, slotZeros)
+          .getResult();
+  Value validSlot =
+      rewriter.create<ttir::MultiplyOp>(loc, slotType, belowLimit, nonNegative)
+          .getResult();
+  Value scatterSource =
+      rewriter
+          .create<ttir::WhereOp>(loc, slotType, validSlot, slotOnes, slotZeros)
+          .getResult();
+  Value safeIndexF32 =
+      rewriter
+          .create<ttir::WhereOp>(loc, slotType, validSlot, indicesSlot,
+                                 slotZeros)
+          .getResult();
+  // The scatter index must be integer-typed. f32 -> i32 is exact here because
+  // every surviving value is an in-range key position and T is far below 2^24.
+  Value scatterIndex =
+      rewriter.create<ttir::TypecastOp>(loc, slotIndexType, safeIndexF32)
+          .getResult();
+
+  Value hitZeros =
+      rewriter
+          .create<ttir::FullOp>(loc, hitType, rewriter.getF32FloatAttr(0.0f))
+          .getResult();
+  // A key selected more than once in a row just accumulates a larger count, so
+  // the `> 0` test below does not depend on the producer emitting distinct
+  // indices.
   Value hitCount =
       rewriter
-          .create<ttir::SumOp>(
-              loc, indexTensorType({batch, querySeqLen, 1, keySeqLen}), slotHit,
-              rewriter.getBoolAttr(/*keep_dim=*/true),
-              rewriter.getI32ArrayAttr({2}))
+          .create<ttir::ScatterOp>(
+              loc, hitType, hitZeros, scatterIndex, scatterSource,
+              rewriter.getI32IntegerAttr(/*dim=*/2),
+              ttcore::ReduceTypeAttr::get(rewriter.getContext(),
+                                          ttcore::ReduceType::Sum))
           .getResult();
+
   auto maskIndexType = indexTensorType({batch, 1, querySeqLen, keySeqLen});
   Value selected =
       ttir::utils::createReshapeOp(rewriter, loc, hitCount,
diff --git a/test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir b/test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir
index 972eeb7153..26df6a5e43 100644
--- a/test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir
+++ b/test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir
@@ -49,9 +49,18 @@ module @sparse_sdpa attributes {} {
   // CHECK-DAG: "ttir.slice_static"
   // The mask index arithmetic runs in f32, not the bf16 element type, so that
   // key positions past bf16's exact-integer range (256) are not conflated. The
-  // typecast of `indices`, the key-position arange, the equality test and the
-  // slot-count reduction are all f32; only the additive 0/-inf mask is bf16.
-  // CHECK-DAG: "ttir.arange"{{.*}} -> tensor<{{.*}}xf32>
-  // CHECK-DAG: "ttir.eq"{{.*}}(tensor<{{.*}}xf32>, tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
+  // typecast of `indices` and the in-range predicate are f32; only the additive
+  // 0/-inf mask is bf16.
+  //
+  // The membership test is a scatter-accumulate into a [B, S, T] hit-count
+  // buffer, NOT a one-hot [B, S, TOPK, T] compare-and-reduce: the latter needs
+  // O(S * TOPK * T) memory (1.07e9 elements at S = T = TOPK = 1024) and does not
+  // fit. Reduction must be `sum` so that a masked slot redirected onto key 0
+  // (contributing 0.0) cannot clear a genuine hit on key 0 in the same row.
+  // CHECK-DAG: "ttir.lt"{{.*}}(tensor<{{.*}}xf32>, tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
+  // CHECK-DAG: "ttir.ge"{{.*}}(tensor<{{.*}}xf32>, tensor<{{.*}}xf32>) -> tensor<{{.*}}xf32>
+  // CHECK-DAG: "ttir.scatter"{{.*}}scatter_reduce_type = #ttcore.reduce_type<sum>
   // CHECK-DAG: "ttir.where"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) {{.*}}: (tensor<{{.*}}xf32>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
+  // No one-hot slot tensor may reappear.
+  // CHECK-NOT: "ttir.eq"
 }
diff --git a/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_bsz.mlir b/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_bsz.mlir
index 34f28f04f4..fdcde97d31 100644
--- a/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_bsz.mlir
+++ b/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_bsz.mlir
@@ -10,14 +10,19 @@
 module @sparse_sdpa {
   func.func public @sparse_sdpa(%q: tensor<4x32x32x64xbf16>, %kv: tensor<4x1x64x64xbf16>, %idx: tensor<4x1x32x32xui32>) -> tensor<4x32x32x32xbf16> {
     // The primitive ops may be split across the main function and a hoisted
-    // const-eval function (the key-position arange depends only on shapes), so
+    // const-eval function (the shape-only constants get hoisted), so
     // match them anywhere in the module rather than scoping to a single
     // function.
     // CHECK-DAG: "ttnn.matmul"
     // CHECK-DAG: "ttnn.multiply"
-    // CHECK-DAG: "ttnn.arange"
-    // CHECK-DAG: "ttnn.eq"
-    // CHECK-DAG: "ttnn.sum"
+    // The sparsity mask is a scatter-accumulate into a [B, S, T] hit-count
+    // buffer. It must NOT be a one-hot [B, S, TOPK, T] compare-and-reduce
+    // (arange + eq + sum over the slot axis): that needs O(S * TOPK * T) memory,
+    // which is 1.07e9 elements at S = T = TOPK = 1024 and does not fit.
+    // The upper-bound test canonicalizes into a second "ttnn.gt" with swapped
+    // operands, so only the lower bound shows up as "ttnn.ge".
+    // CHECK-DAG: "ttnn.ge"
+    // CHECK-DAG: "ttnn.scatter"
     // CHECK-DAG: "ttnn.gt"
     // CHECK-DAG: "ttnn.where"
     // CHECK-DAG: "ttnn.softmax"
diff --git a/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir b/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir
index c9169e7a08..1aed397650 100644
--- a/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir
+++ b/test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir
@@ -10,14 +10,19 @@
 module @sparse_sdpa {
   func.func public @sparse_sdpa(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
     // The primitive ops may be split across the main function and a hoisted
-    // const-eval function (the key-position arange depends only on shapes), so
+    // const-eval function (the shape-only constants get hoisted), so
     // match them anywhere in the module rather than scoping to a single
     // function.
     // CHECK-DAG: "ttnn.matmul"
     // CHECK-DAG: "ttnn.multiply"
-    // CHECK-DAG: "ttnn.arange"
-    // CHECK-DAG: "ttnn.eq"
-    // CHECK-DAG: "ttnn.sum"
+    // The sparsity mask is a scatter-accumulate into a [B, S, T] hit-count
+    // buffer. It must NOT be a one-hot [B, S, TOPK, T] compare-and-reduce
+    // (arange + eq + sum over the slot axis): that needs O(S * TOPK * T) memory,
+    // which is 1.07e9 elements at S = T = TOPK = 1024 and does not fit.
+    // The upper-bound test canonicalizes into a second "ttnn.gt" with swapped
+    // operands, so only the lower bound shows up as "ttnn.ge".
+    // CHECK-DAG: "ttnn.ge"
+    // CHECK-DAG: "ttnn.scatter"
     // CHECK-DAG: "ttnn.gt"
     // CHECK-DAG: "ttnn.where"
     // CHECK-DAG: "ttnn.softmax"
```
<!-- END PATCH -->
