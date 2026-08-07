# Falcon3-7B-Instruct: where the ifeval gap actually is (TT vs L4 GPU)

Follow-up to [tt-inference-server#4752](https://github.com/tenstorrent/tt-inference-server/issues/4752).
Uses the L4 reference `samples_*.jsonl` @ddilbazTT attached on 2026-08-06 — the first time we have
per-sample GPU data to diff against.

Artifacts: `falcon3_sdpa_ccfg_2026-08-02/gpu_reference/`. TT side is the four full-set builds from
`FALCON3_SDPA_NUMERICS_DEBUG.md`.

## Headline

| | ifeval (541) | gpqa (198) |
|:--|--:|--:|
| GPU run 1 / run 2 | 73.94 / 74.12 | 43.43 / 43.43 |
| TT (mean of 4 builds) | 66.96 | 38.89–39.90 |
| **gap** | **−7.07** | −4.5 |

The GPU measures **74.03**, not the 72.64 recorded as `gpu_reference_score` in `eval_config.py`.
So the real deficit is ~7 points, larger than the ~5 previously quoted.

## Correction to the earlier read

An earlier comment on #4752 said the loss was "concentrated in counting/structure instructions
(`length_constraints` ~62%, `combination` ~52-57%)". **That was TT-absolute numbers with no
control.** With the GPU baseline, `combination` is only +2.3 behind — GPU scores 55.4 there too;
it is simply a hard category. The same applies to `length_constraints:number_words`, where TT
actually *beats* GPU.

## Per-category deficit (instruction-level strict)

| category | GPU | TT | gap | instr. lost/run |
|:--|--:|--:|--:|--:|
| detectable_format | 92.4 | 79.8 | **+12.6** | 19.8 |
| startend | 91.0 | 79.9 | **+11.2** | 7.5 |
| length_constraints | 67.1 | 60.1 | +7.0 | 10.0 |
| language | 96.8 | 91.1 | +5.6 | 1.8 |
| detectable_content | 94.3 | 89.2 | +5.2 | 2.8 |
| change_case | 82.6 | 77.5 | +5.1 | 4.5 |
| combination | 55.4 | 53.1 | +2.3 | 1.5 |
| keywords | 75.5 | 74.1 | +1.4 | 2.2 |
| punctuation | 93.9 | 93.9 | 0.0 | 0.0 |

`detectable_format` + `startend` = **55% of the total deficit** (27.3 of 50 instructions/run).

### Worst individual instructions

| instruction | GPU | TT | gap | n | lost/run |
|:--|--:|--:|--:|--:|--:|
| detectable_format:number_bullet_lists | 74.2 | 43.5 | +30.6 | 31 | 9.5 |
| length_constraints:number_paragraphs | 92.6 | 66.7 | +25.9 | 27 | 7.0 |
| detectable_format:json_format | 94.1 | 63.2 | +30.9 | 17 | 5.3 |
| change_case:english_capital | 96.0 | 79.0 | +17.0 | 25 | 4.2 |
| startend:end_checker | 88.5 | 73.1 | +15.4 | 26 | 4.0 |
| startend:quotation | 92.7 | 84.1 | +8.5 | 41 | 3.5 |
| detectable_content:postscript | 100.0 | 90.4 | +9.6 | 26 | 2.5 |

### Where TT matches or beats GPU

`length_constraints:number_words` (−1.4), `change_case:english_lowercase` (−1.3),
`keywords:frequency` (−1.2), `combination:two_responses` (−1.0), `keywords:forbidden_words` (0.0),
`punctuation:no_comma` (0.0).

The split is **"exact global structure / stop in the right place"** (TT loses) vs
**"local or statistical"** (TT matches).

## Mechanism: over-generation, not truncation

Truncation was the obvious first hypothesis and is **refuted** — TT outputs are *longer*:

| | median chars | mean | max |
|:--|--:|--:|--:|
| GPU | 803 | 1158 | 6308 |
| TT | 943 | 1372 | **17789** |

Three independent confirmations:

1. **`number_bullet_lists`**: TT emits *too many* bullets in **12/31** docs vs GPU **4/31**.
   Too-few is identical (4 vs 4) — so it is purely an overshoot, not a miscount.
2. **Length ratio tracks the outcome**: TT-fails/GPU-passes → TT **1.15×** longer;
   TT-passes/GPU-fails → **0.92×**; both-pass → **1.00×**.
3. TT produces **59** responses >3000 chars vs GPU **35**.

Over-generation breaks exactly the failing set: too many bullets/paragraphs, prose wrapped around
JSON, text after the required closing phrase, unclosed quotes, content after `P.S.` — while leaving
word counts and forbidden-word checks intact.

## gpqa

198 docs: 70 both-right, **16 GPU-only**, 7 TT-only, 105 both-wrong. A real but small 9-question
deficit with no obvious pattern; the ifeval signal is far stronger and should be chased first.

## Biggest finding: TT output depends on batch composition

Two runs of the **same TT build** (expfix / mathapprox), temp 0 greedy:

| | byte-identical responses |
|:--|--:|
| TT expfix, run A vs run B | 76 / 271 (**28.0%**) |
| TT mathapprox, run A vs run B | 87 / 271 (**32.1%**) |
| **GPU run 1 vs run 2 (ifeval)** | **541 / 541 (100.0%)** |
| **GPU run 1 vs run 2 (gpqa)** | **198 / 198 (100.0%)** |

Under greedy argmax at temperature 0, the same model on the same prompt must produce the same
tokens. The GPU does, exactly, on both tasks. TT reproduces fewer than a third of its own outputs.

This also explains the repetition-loop reshuffling: 105 distinct docs looped across the four
builds and **none looped in all four**, which is what you'd expect if the output is unstable
run-to-run rather than a fixed property of the prompt.

**RESOLVED — it is batch-composition dependence, not non-determinism.** A direct probe against a
live server (`determinism_probe.py`, build 4, device sampling, temp 0) separates the two:

| test | result |
|:--|:--|
| Same prompt, 5 reps, **one at a time** | **1 unique output on all 4 prompts — deterministic** |
| Same prompt **alone vs alongside 24 concurrent requests** | **2 of 4 prompts produce different output** |

So TT greedy decoding *is* reproducible when requests run sequentially. What changes the output is
**what else is in the batch**. Example divergences:

```
"Write a JSON object describing a book"  diverges at char 281:
    alone   : 'pages": 281,\n  "isbn": "978-0061121618"\n'
    batched : 'characters": [\n    {\n      "name": "Scou'
"List exactly 3 bullet points about trees"  diverges at char 328 (whitespace)
```

This is a **batch-invariance violation**, and it explains every odd behaviour we have seen:

- the 28-32% byte-identity between a `--limit 0.5` run and a full run of the same build (different
  doc counts -> different batching)
- the repetition loops reshuffling across builds with none stable in all four
- run-to-run eval score variance that looked like noise

The GPU's 100% reproducibility was measured on two runs with **identical** workload, so it does not
by itself prove the GPU is batch-invariant — it proves the GPU is reproducible under a fixed
workload, which TT also is when the workload is fixed sequentially. The open question is whether
the GPU is *also* batch-dependent; the same probe should be run against it.

### Matched-run result: the eval cannot resolve the differences we were measuring

Two full runs, **identical build, config, server process and settings** (541 ifeval docs each):

| | run A | run B |
|:--|--:|--:|
| ifeval | **66.36** | **63.59** |
| gpqa | 38.38 | 37.88 |
| byte-identical responses | — | **187 / 541 (34.6%)** |
| correctness flips | — | **77** (B gained 31, lost 46) |
| repetition loops | 35 | 42 |

> **Build note for runs A and B.** These two ran on tt-mlir `f209dddc9e` (trace fix + Darko's SDPA
> ccfg) but **without** the tt-metal accurate-exp patch — a `cmake --build` reset the tt-metal
> submodule checkout to its pinned `f1f4ff75579`, silently discarding the local commit. A and B are
> therefore identical to each other (which is all the matched comparison requires), but they are
> *not* the same configuration as the earlier build-4 run that scored 65.62. **Gotcha to remember:
> tt-metal working-tree patches do not survive a tt-xla rebuild** — unlike tt-mlir patches, which do
> when HEAD already equals the pin. Re-apply with
> `git cherry-pick -n b6371bb6488` in the tt-metal checkout after every rebuild, and verify with
> `grep exp_approx compute_common.hpp`.

A and B differ by **2.77 points**, which **exceeds the entire spread across the four different
builds compared earlier (65.62 - 68.02 = 2.40).**
So the four-build comparison in `FALCON3_SDPA_NUMERICS_DEBUG.md` was measuring run-to-run noise,
not build differences. The conclusion there ("SDPA numerics do not move the evals") still holds, but
for a stronger reason than stated: **the measurement cannot resolve effects of that size at all.**

Note this is *not* captured by lm-eval's reported stderr (2.01-2.04). That stderr is binomial
sampling error assuming a fixed model; it does not account for the model producing different text
on each run. The real run-to-run variance is comparable or larger and is invisible to it.

## Next-step debug leads

1. **Chase batch-invariance** — this is the root lead, and it now also gates the ability to measure
   *anything* on this stack. The same prompt must produce the same tokens
   regardless of what else is in flight. Likely suspects: batch padding to fixed buckets, batch-slot
   dependent reductions/matmul tiling, or the paged-KV block layout varying with concurrency.
   Reproduce with `determinism_probe.py` (fast, no eval harness needed).
2. **Chase the repetition loops** — biggest single identifiable bucket (~3.9 ifeval points).
   Onset is early (median 138 chars), so look at the prefill->decode handoff and first decode
   steps, not long-context drift.
3. **The 17789-char runaway outputs** — check for repetition loops. GPU's max is 6308.
4. **Chat template** — a mismatch produces exactly this signature and is what bit Mistral-Small.
   Compare the rendered prompt TT vs GPU on the same doc.
5. **Confirm it is TT-side before filing.** Both sides report `gen_kwargs: None`, `limit: None`,
   and both hit `/v1/completions`, so harness config matches. But GPU ran `num_concurrent=32` vs
   our 16, and its eval id is `tt-transformers` vs our `forge-vllm-plugin` — different serving
   stacks. Rule out the GPU server's stop-token/template config differing from ours.

**Not a lead:** SDPA numerics. A verified 5.1× rel_l2 improvement moved full-set ifeval by less
than 1 stderr (see `FALCON3_SDPA_NUMERICS_DEBUG.md`).
