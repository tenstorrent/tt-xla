# [vLLM] Batched prefill is non-deterministic across identical prompts (`fp32_dest_acc_en=False`, batch >= 4)

## Summary

Greedy (temperature=0) decoding of **N identical prompts in one batch** produces
**different outputs across rows**. Each row in a batch is an independent
computation, so argmax decoding must be identical per row — divergence is a
correctness bug.

It reproduces with the default benchmark configuration (which sets
`fp32_dest_acc_en=False` and `batch_size=32`). Root cause is low-precision
(bf16) **destination accumulation**, which is **batch-position-dependent**: the
per-row logits differ slightly and flip near-tie greedy tokens. Enabling fp32
destination accumulation makes the **prefill** path fully agree; a residual in
the **decode** path keeps batch=32 from being fully clean under fp32 (see below).

The sharpest, most actionable signature (isolated this session — see
"A clean 16-user prefill boundary"): in a **pure-prefill** run (single generated
token, greedy), the 32 identical prompts split as a **clean, deterministic 16/16
partition** — batch indices 0–15 produce the correct token, indices **>= 16**
produce a degenerate/different token. The boundary is a hard **16 users**
(independent of sequence length and total batch), holds on a 1-layer truncation
and the full 28-layer model, is **not** in the sampler (reproduces with host
sampling), and is **fully eliminated by `fp32_dest_acc_en=True`**. No single op
reproduces it in isolation even at the exact production shape *and* the exact
tt-mlir-emitted compute config — so it is a **graph-composition** effect under
bf16 dest-accumulation, not an inherent single-kernel bug. It reproduces
**byte-identically on the stock baseline** (tt-xla `8a54156b5` + stock tt-mlir
`c5f398432`), so it is pre-existing, not a branch artifact (see "Reproduced on
stock baseline").

## Reproduction

Single-device env:
```
export TT_MESH_GRAPH_DESC_PATH=$HOME/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
```

Default benchmark (batch=32, BFP8 weights, `fp32_dest_acc_en=False`):
```
pytest -svv tests/benchmark/test_vllm_benchmarks.py -k "llama-3.2-3b and not embedding"
```
The 32 prompts are identical; the printed `[i] prompt -> text` outputs split into
**3 distinct texts** (e.g. 25 / 6 / 1). Confirmed on both
**Llama-3.2-3B-Instruct** and **Llama-3.1-8B-Instruct** (not model-specific).

Minimal: send N identical prompts via `LLM.generate([prompt]*N, greedy)` with
`max_num_seqs=N`, `additional_config={"fp32_dest_acc_en": False,
"experimental_weight_dtype": "bfp_bf8"}`. batch=4 already triggers; **batch=2
does not**.

## Live in CI (already)

A recent Performance Benchmark run shows this on the exact default config, and
the job is marked **success**:

- Job: `perf vllm_llama_3_2_3b (n150-perf)` —
  https://github.com/tenstorrent/tt-xla/actions/runs/27064401476/job/79883258025
  (2026-06-06)
- Config logged: `max_num_seqs=32`, `fp32_dest_acc_en=False`,
  `experimental_weight_dtype=bfp_bf8`, `max_model_len=128`.
- The 32 identical prompts print **3 distinct completions (26 / 5 / 1)** — yet
  the job passes.

## Why it isn't caught today

The benchmarks assert token *counts* and PCC of **User 0** only — so per-user
divergence (Users 1..N) is invisible. This matches the prior observation in
**#4785** ("Assess benchmark LLM output quality"): "Some LLMs pass this check but
produce invalid output for some or all users (garbage, repetitive, empty)."

## Root cause

`fp32_dest_acc_en` controls whether matmul/reduction kernels accumulate into a
fp32 destination or a lower-precision (bf16/TF32) one. Identical batch rows are
independent, so any correct kernel — at any precision — must yield identical
per-row results. The observed per-row divergence means the low-precision
accumulation result **depends on the row's position within the batched op's
tiling** (reduction order / tile packing across the batch dimension). fp32
accumulation has enough headroom that the per-position deltas round away.

Evidence it is precision-driven (not gross corruption):
- Appears only with `fp32_dest_acc_en=False`; `=True`/default agrees on small
  batch (clean A/B, all else equal).
- Needs depth (full model; a 4-layer truncation agrees) and near-tie inputs
  (repetitive prompt / BFP8 weights) — tiny per-row deltas crossing argmax
  boundaries.
- Batch threshold (>=4, not 2) points at a batch-dimension tiling effect.

This is the same family as **tt-mlir #8666** ("`fp32_dest_acc_en` not working
properly for `ttnn.matmul`"): the `TTNNSetComputeKernelConfig` pass builds an
incomplete `ComputeKernelConfig`, so L1->dest copies happen in TF32 (~bf16)
rather than FP32.

## Does the tt-mlir matmul fix (#8666 / PR #8708) resolve it?

Reconstructed PR #8708's logic in tt-mlir and rebuilt `libTTMLIRCompiler.so`:

1. **As-is: no-op for inference.** #8708 only forces `fp32_dest_acc_en` +
   `packer_l1_acc` when a matmul's **inner dim > 50000** (a training/lm_head
   case). Llama inference matmul inner dims are <= ~8192 (seq_len, hidden, MLP),
   so the rewrite never fires.
2. **Relaxed (force fp32+packer on all matmuls/linears, no threshold),
   rebuilt + retested** with the default benchmark:

   | batch | default (`fp32_dest_acc_en=False`) | relaxed matmul fix | `fp32_dest_acc_en=True` option |
   |------:|:----|:----|:----|
   | 4  | diverges | **1 (fixed)** | 1 (fixed) |
   | 32 | 3 distinct | **2 distinct (still diverges)** | 2 distinct |

**Conclusions:**
- Widening the threshold so the fix covers inference matmuls **does help** —
  batch=4 becomes deterministic.
- But it is **necessary-not-sufficient**: a **batch=32 case persists** even with
  fp32 accumulation forced on matmul+linear+softmax+reductions. Forcing
  `packer_l1_acc` added nothing observable beyond `fp32_dest_acc_en=True` alone.
- Note: the fused **SDPA** op also has no fp32-accumulation knob set
  (`ttnn.scaled_dot_product_attention*` don't implement
  `TTNNComputeKernelConfigOpInterface`; tt-mlir passes `compute_kernel_config=
  nullopt`; cf. tt-mlir **#8657**) — but forcing fp32 on SDPA too did **not**
  remove the residual (see the isolation section below), so SDPA accumulation is
  not the remaining source either.

## Deeper root-cause: it's deterministic per-batch-position numerics (SDPA fp32 experiment)

To test whether SDPA accumulation is the residual, the tt-mlir runtime SDPA ops
(prefill + both decode variants, which all pass `compute_kernel_config=nullopt`)
were patched to force `fp32_dest_acc_en` + `packer_l1_acc` via
`init_device_compute_kernel_config(...)`, rebuilt, and retested.

Results (Llama-3.2-3B, BFP8, with matmul `fp32_dest_acc_en=True`):

| case | matmul fp32 only | + SDPA fp32 (all variants) |
|:--|:--|:--|
| batch=4 single-shot | 1 (clean) | 1 (clean) |
| batch=32 (benchmark, decode-heavy) | 2 (1 outlier) | **2 (unchanged)** |
| batch=4 multi-chunk | 3 | **4** (but row0 now == single-shot) |

Two findings:
1. **It is deterministic, not a race.** Re-running an identical config twice
   produced **byte-identical per-row** outputs (all rows SAME across runs). So a
   given batch *position* reproducibly yields a slightly different result than
   another position for the *same* prompt.
2. **fp32 accumulation mitigates but does not eliminate it.** Forcing fp32 on
   matmul fixes batch≤4 single-shot and cuts batch=32 from 3→2; forcing fp32 on
   SDPA additionally makes the chunked-path row0 match the single-shot reference
   — but a residual divergence persists for the *other* batch rows at batch=32
   and in the chunked path, **resistant to fp32 on matmul + SDPA + softmax +
   reductions**.

It is a **deterministic, batch-position-dependent numerical asymmetry** (same
prompt at a different batch row reproducibly differs), and fp32 accumulation
mitigates but does not eliminate it. (The SDPA-fp32 patch above was reverted —
partial mitigation, not a fix.)

## Isolation: which layer? (standalone ttnn probes — points away from tt-metal kernels)

Two follow-up experiments to localize the source:

**1. Ruled out the batched `paged_fill_cache` (tt-xla commit `c8cc0e739`).**
Reverting that commit's batched fill back to the per-user loop (single-element
`batch_idx`) leaves the b32 divergence **unchanged** (still 3 distinct, same
26/5/1 split). So how the KV cache is written is not the cause.

**2. The isolated tt-metal kernels are row-position-invariant.** A standalone
`ttnn` program (no tt-mlir, no vLLM) ran each op with **32 identical batch rows**
and compared the per-row outputs **bit-exactly** (random inputs, TILE layout,
default interleaved memory):

| op | default cfg | fp32 dest-acc cfg |
|:--|:--|:--|
| `ttnn.matmul` (M=64,K=3072,N=8192) | bit-identical | bit-identical |
| `ttnn.transformer.scaled_dot_product_attention` (dense) | bit-identical | bit-identical |
| `..._decode` (Flash-Decode, b-parallel) | bit-identical | bit-identical |

All six cases are **bit-identical across rows**. So the matmul / dense-SDPA /
Flash-Decode kernels are *not* inherently row-position-dependent.

**Conclusion (revised).** Since the isolated kernels are row-invariant but the
full model diverges per row, the divergence is **introduced by the compiled
graph** — i.e. the layout / sharding / program-config / op choices that the
tt-xla → tt-mlir lowering selects for the batched ops — **not** by an inherent
tt-metal kernel bug. This points at **tt-mlir** (config/layout generation; cf.
the same-family #8666 matmul-config and #8657 SDPA-config issues), and fp32
dest-acc is a partial mitigation, not the fix.

## A clean 16-user prefill boundary (sharpest signature, fp32-fixable)

Narrowing to a **pure-prefill** probe (single generated token, `max_tokens=1`,
so no decode steps are involved) turns the "fuzzy per-row noise" into a sharp,
structural signature. Repro (`tmp/batch_determinism.py`, Llama-3.2-3B-Instruct,
greedy, identical prompts, `fp32_dest_acc_en=False`):

```
BD_BATCH=<N> BD_MML=128 BD_MAX_TOKENS=1 BD_BENCH_PROMPT=1 BD_FP32_ACC=0 \
  BD_NLAYERS=<1 or 0> python tmp/batch_determinism.py
```

Per-row first-token results (each cell = the single generated token id per row):

| batch N | rows 0–15 | rows 16–N | result |
|--------:|:----------|:----------|:-------|
| 4  | tok 25 | — | **all identical** |
| 8  | tok 25 | — | **all identical** |
| 16 | tok 25 | — | **all identical** |
| 24 | tok 25 (correct) | tok 0 (degenerate) | split at index 16 |
| 32 | tok 25 (correct) | tok 0 (degenerate) | split at index 16 |

Key properties, all verified:

1. **Hard boundary at batch index 16.** Indices 0–15 are always correct; indices
   >= 16 degenerate. The threshold is **16 users**, not "half the batch" (batch=24
   splits 16/8, not 12/12) and not a row-count limit: with a 2-tile prompt
   (31 tokens → 64-token padding) the boundary **stays at user 16**, it does not
   move to user 8. So it is a **batch-dimension grouping of 16**, independent of
   sequence length.
2. **Deterministic.** Re-running the identical config gives byte-identical
   per-row output (same 16/16 split, same token ids) every time.
3. **Layer-count independent.** Same boundary on a 1-layer truncation
   (`BD_NLAYERS=1`) and the full 28-layer model (`BD_NLAYERS=0`); only the
   specific token ids differ.
4. **Not the sampler.** Forcing host sampling (`BD_CPU_SAMPLING=1`, i.e.
   `additional_config={"cpu_sampling": True}`) reproduces the split — so the
   divergence is in the **logits coming off the device**, not in the on-device
   `tt::sampling`/argmax.
5. **`fp32_dest_acc_en=True` fully fixes it.** Same batch=32 prefill run with
   `BD_FP32_ACC=1` yields **all 32 rows identical**. So this prefill component is
   completely covered by fp32 destination accumulation (unlike the residual
   decode-path divergence that keeps the full benchmark at 2 distinct outputs).

**No single op reproduces it — even config-matched.** Re-running the standalone
`ttnn` probe (`tmp/ttnn_prod_shapes.py`) at the **exact production shapes** from
the lowered prefill graph (qkv/o_proj/gate_up/down matmuls at M=1024=32×32,
dense SDPA `[32,24,32,128]` `is_causal`, rms_norm) with **N=32 identical rows**:

| compute config passed to the op | result (all ops) |
|:--|:--|
| `None` (ttnn default) | bit-identical across 32 rows |
| `fp32_dest_acc_en=True` | bit-identical across 32 rows |
| **exact tt-mlir config** (`math_fidelity=hifi4, fp32_dest_acc_en=False`) | **bit-identical across 32 rows** |

Even with the precise config the lowered TTNN MLIR specifies, no isolated op
splits at 16. The lowered graph's intermediates are all **interleaved DRAM**
(no sharded layouts; checked the `#ttnn_layout` encodings), and the matmul ops
carry only `compute_config = hifi4` with **no explicit `program_config`** — so
tt-metal selects the core grid from the `[1024, N]` shape at runtime.

**Interpretation.** A row-local op (matmul / elementwise / rms_norm) cannot, by
construction, turn 32 identical input rows into a 16/16 split — so the split is
**introduced by graph composition**: the in-graph program config / core-grid
that tt-mlir+tt-metal pick for the batched prefill ops partitions the 32-user
(1024-row) batch into two groups of 16 whose bf16 destination-accumulation
reductions round differently; fp32 accumulation gives enough headroom that the
two groups agree. The standalone calls let ttnn pick its own (uniform) grid and
so never create the partition. This is the **#8666 family** (tt-mlir compute
kernel config: bf16 dest-acc where fp32 is needed), with a precise new signature:
a 16-user prefill grouping. Closing it the rest of the way needs an in-graph
per-op activation dump (tt-mlir runtime op callback / `dumpTensor`), which is a
tt-mlir build-side change and is flagged, not done here.

## Reproduced on stock baseline (no branch changes)

To rule out the development branch's tt-mlir cherry-picks and plugin changes, the
16-user boundary was reproduced on the **unmodified baseline**:

- tt-xla **`8a54156b5`** ("Add relative L2 error similarity metric...") — the
  commit before any of the branch's vLLM-plugin / tt-mlir-uplift work.
- stock tt-mlir **`c5f398432`** (the version `8a54156b5` pins), rebuilt and
  installed; stock tt-metal kernels.

Baseline results (Llama-3.2-3B, greedy, identical prompts, `fp32_dest_acc_en=False`),
**byte-identical** to the development branch:

| config | result |
|:--|:--|
| batch 16, prefill-only, 1-layer | all identical (clean) |
| batch 32, prefill-only, 1-layer | 16/16 split (rows 0–15 = tok 25, rows >= 16 = tok 0) |
| batch 32, prefill-only, full 28-layer | 16/16 split (rows 0–15 = tok 720, rows >= 16 = tok 320) |
| batch 32, prefill-only, 1-layer, **`fp32_dest_acc_en=True`** | all identical (fixed) |

The four tt-mlir commits and one tt-metal commit between this baseline and the
dev branch are: paged_fill_cache batched-`batch_idx` (verifier + kernel, #45117),
a tt-metal paged-prefill bump, BF16-query-over-BFP8-KV in paged SDPA *decode*
(#8668), and an optimizer empty-beam-sink fallback (#8678). **None** touch matmul
or SDPA compute-kernel-config / destination accumulation, and the
`paged_fill_cache` change is never exercised by this single-chunk prefill path
(it's a side-effect write whose output the prefill SDPA does not read; baseline
tt-xla also uses the per-user fill loop). So the bug is **pre-existing in stock
tt-xla + stock tt-mlir**, not introduced by any branch change.

## Aside: an experimental chunked-prefill change makes it worse

A POC chunked-prefill change (extra per-chunk ops: a dense gather of the cached
prefix + a masked SDPA over it) was tested and **adds** divergence on top of the
matmul issue. Same model/prompt (Llama-3.2-3B, BFP8), batch=4, the only variable
is the chunk budget (off = single-shot vs on = multi-chunk):

| `fp32_dest_acc_en` | chunking | distinct outputs (of 4 identical prompts) |
|:--|:--|--:|
| True | off (single-shot) | 1 (clean) |
| True | on (multi-chunk, budget 128) | 3 |
| forced fp32+packer | off | 1 (clean) |
| forced fp32+packer | on (budget 128) | 3 |

So in a precision regime where the **non-chunked** batch=4 path is perfectly
clean, enabling chunking **introduces** divergence (and the multi-chunk rows
don't match the single-shot reference). Expected, since the chunked path runs
more low-precision matmuls and leans harder on the (fp32-knob-less) SDPA op.
Caveat: n=1 (one batch size / prompt / model); no clean fp32-off off-vs-on pair.
batch=1 chunked is unaffected (token-identical to single-shot).

## Mitigation

- **tt-xla:** `fp32_dest_acc_en=True` **fully fixes the prefill path** (the clean
  16-user boundary above disappears — all 32 rows agree), but a residual in the
  **decode** path keeps the full decode-heavy batch=32 benchmark from going clean,
  and it risks the **#6920** OOM on Llama/Ministral-8B (the reason `fp32_dest_acc`
  is disabled for opt-level>0, per **#3206**). So it is a strong partial
  mitigation (complete for prefill), not a complete fix.
- **tt-mlir (proper fix):** (a) extend #8666/#8708 to cover inference matmul
  shapes (smaller/configurable threshold, or gate by dtype/output rather than a
  fixed inner-dim cutoff); (b) give the SDPA op an fp32-accumulation path.

## Related issues
- tt-mlir **#8666** (open) — `fp32_dest_acc_en` not working properly for
  `ttnn.matmul` (root cause).
- tt-mlir **#8708** (open PR, closes #8666) — fix, gated on inner dim > 50000.
- tt-mlir **#8657** (open) — generic `SDPAProgramConfigAttr` (SDPA config).
- tt-xla **#4785** (closed) — benchmark output-quality assessment; observed the
  per-user divergence masked by User-0 PCC.
- tt-xla **#3206** (closed) + tt-mlir **#6920** — `fp32_dest_acc` disabled for
  opt>0 due to 8B OOM when enabled.

## Suggested follow-ups
- Add an output-consistency assertion to the benchmark (compare all users, not
  just User-0 PCC) so this can't regress silently (cf. #4785).
- Track the SDPA fp32-accumulation gap as the residual batch=32 source.
- Confirm whether `fp32_dest_acc_en=True` reintroduces the #6920 OOM on 8B.

## Draft comment for tt-mlir #8666 / PR #8708 (before merge)

> Heads-up from the inference/serving side (tt-xla vLLM): we're hitting what
> looks like the same `fp32_dest_acc_en` matmul-accumulation issue, but the
> `inner dim > 50000` gate in this PR means the fix **won't apply to inference**.
>
> Symptom: greedy decoding of N *identical* prompts in one batch diverges per
> row at batch >= 4 (Llama-3.2-3B / 3.1-8B default b32 benchmarks produce 2-3
> distinct outputs for 32 identical prompts; masked in our benchmarks because
> they PCC-check only User 0).
>
> Findings from reconstructing this PR's logic locally and removing the 50000
> threshold (forcing `fp32_dest_acc_en`+`packer_l1_acc` on all matmuls/linears):
> - Llama inference matmul inner dims are <= ~8192, so the 50000 gate never
>   fires — the PR as-is is a no-op for inference.
> - Widening it fixes batch=4 but **batch=32 still diverges**, even with fp32
>   accumulation forced on matmul+linear+softmax+reductions.
> - The remaining divergence appears to come from the fused **SDPA** op, which
>   doesn't implement `TTNNComputeKernelConfigOpInterface` and so has no
>   fp32-dest-acc path at all (cf. #8657).
>
> Requests: (1) consider a smaller/configurable threshold (or gate by
> dtype/output rather than a fixed inner-dim cutoff) so inference matmuls are
> covered; (2) track an fp32-accumulation path for SDPA, since a matmul-only fix
> won't fully resolve inference batch>1 determinism.
