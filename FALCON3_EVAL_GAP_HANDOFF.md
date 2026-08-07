# Handoff: Falcon3-7B eval accuracy (tt-inference-server#4752)

Written 2026-08-07 at the end of a long session, for whoever picks this up next — including
future-me with no memory of it. Companion to `FALCON3_EVAL_GAP_GPU_COMPARISON.md` (the evidence)
and `FALCON3_SDPA_NUMERICS_DEBUG.md` (the earlier, negative-result branch of the work).

## Where it landed

The ifeval gap vs the L4 reference is **7.07 points**, and the root cause is a
**batch-invariance violation**, not arithmetic.

Greedy decoding on TT is deterministic when requests run sequentially (5/5 identical reps on four
prompts), but the *same* prompt decoded alongside 24 concurrent requests produces different output
(2 of 4 prompts). The consequence is severe: **two full eval runs of an identical build, config and
server process differ by 2.77 ifeval points with only 34.6% of responses byte-identical.** That
exceeds the 2.40 spread across the four *different* builds compared earlier — so those build
comparisons were measuring noise.

Biggest visible symptom: TT degenerates into token-repetition loops on ~6.7% of prompts versus the
GPU's 0.2%, worth ~3.9 ifeval points on its own.

## Repo state (all pushed unless noted)

| repo | branch | commit | notes |
|---|---|---|---|
| tt-xla | `kmabee/falcon3_7b_debug` | `c08d5af59` | knob plumbing, `math_approx_mode`, docs, `tools/determinism_probe.py` |
| tt-xla | `kmabee/falcon3_spec_tests_debug` | — | spec_tests / conformance work, split off |
| tt-mlir | `kmabee/3abca42835_plus_cherry_picks` | `f209dddc9e` | trace fix + Darko's SDPA ccfg; matches the tt-xla pin |
| tt-metal | `kmabee/sdpa_accurate_exp` | `b6371bb6488` | accurate-exp patch (tt-metal#51927) |
| tt-inference-server | `kmabee/falcon3_hang_accuracy_debug` | `8e7f19b6a` | spec_tests wiring, `MATH_APPROX_MODE` passthrough |

⚠ **The tt-metal patch is applied as an uncommitted `cherry-pick -n` in the local checkout.**
`cmake --build` resets the tt-metal submodule to its pinned commit and silently drops it. After
every rebuild, re-apply and verify:

```bash
cd <tt-metal checkout> && git cherry-pick -n b6371bb6488
grep exp_approx ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp
```

tt-mlir patches do *not* have this problem — they survive because HEAD already equals the pin.

## Read first

- `FALCON3_EVAL_GAP_GPU_COMPARISON.md` — current conclusions and all the evidence
- `FALCON3_SDPA_NUMERICS_DEBUG.md` — the SDPA numerics work and why it was a dead end
- `FALCON3_SPEC_TESTS_DEBUG.md` (on the spec_tests branch) — conformance findings
- `COMMENT_4752_gpu_comparison_update.md` — drafted issue update, **not yet posted**, pending review

## Next steps

1. **Fix batch-invariance.** The root lead, and it gates the ability to measure anything else on
   this stack. Repro with `tools/determinism_probe.py` against a running server — fast, no eval
   harness needed. Suspects: padding to fixed batch buckets, batch-slot-dependent
   reductions/matmul tiling, paged-KV block layout varying with concurrency.
2. **Run the same probe against the GPU.** Its 100% reproducibility was measured under *identical*
   workload, which proves reproducibility, not batch-invariance. Needed before calling this
   TT-specific.
3. **Post the drafted update** once reviewed.
4. Repetition loops (~3.9 points) are the biggest single symptom bucket if a narrower target is
   wanted than (1).

## Do not re-try

- **SDPA numerics.** A verified 5.1× rel_l2 improvement (0.024995 → 0.004872, degenerate case
  bit-exact) moved full-set ifeval by less than one stderr. Two real bugs were found and fixed
  along the way — they were worth fixing on correctness grounds, but they are not this gap.
- **Raising `gpu_reference_score`.** 72.64 was deliberately chosen as the most forgiving of ten
  runs. Do not "correct" it to the measured 74.03.
- **Attributing the deficit to `length_constraints` / `combination`.** That was TT-absolute numbers
  with no control; with the GPU baseline `combination` is only 2.3 behind. The real deficit is
  `detectable_format` (+12.6) and `startend` (+11.2).
- **Truncation as the over-generation explanation.** Refuted — TT outputs are *longer* than the
  GPU's (median 943 vs 803 chars).
- **Trusting single-run eval numbers below ~3 points**, including for the 0.95 acceptance gate,
  until batch-invariance is fixed or runs are repeated and averaged. lm-eval's reported stderr
  (2.01–2.04) does not capture this — it is binomial sampling error assuming a fixed model.
- **Downsampled (CI-nightly) evals for build comparison.** On one build it read 61.99 against a
  full-set 68.02 — opposite direction.

## Environment

Machine `bh-30-special-kmabee-for-reservation-102611`, 1× Blackhole p150b, 24 cores, 503 GiB RAM.
Working dir `/localdev/kmabee/tt-xla`. At handoff a Falcon3 server was running on `:8019` with the
build-4 config (bf16 weights + KV, `math_fidelity=hifi4`, `fp32_dest_acc_en=true`,
`math_approx_mode=false`, opt 1, trace on, device sampling) — kill it if a different config is
needed.

Raw artifacts (eval reports, GPU reference samples, sweep logs, probe output) are in
`falcon3_sdpa_ccfg_2026-08-02/`, untracked and local-only.

## Session gotchas worth keeping

- `venv/activate` is `$(pwd)`-relative: sourcing it outside `/localdev/kmabee/tt-xla` builds bogus
  `PYTHONPATH` entries **and** creates a stray `venv/` in the current directory.
- It also puts `tt-xla/tests` on `PYTHONPATH`, whose `utils.py` shadows tt-inference-server's
  `utils` package — run all tt-inference-server commands with `env -u PYTHONPATH`.
- Eval `rc=1` is the acceptance gate reporting a low score, not a failure. Read the report markdown.
- The eval output directory accumulates across runs — select results/samples files by doc count
  (541 / 198), not by "latest".
- For full-set evals, delete the two `EvalLimitMode.CI_NIGHTLY` lines from the Falcon3 block in
  `evals/eval_config.py`, then `git checkout` it afterwards.
