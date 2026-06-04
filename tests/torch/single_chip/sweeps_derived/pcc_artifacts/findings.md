# `test_matmul_mp` — testing findings and input design options

Standalone summary of what was learned while building this test and the
trade-offs across input regimes. Pairs with `README.md` (snapshot inventory)
and the two `pcc_report*.md` files (raw data).

## Findings

### 1. Input **dtype** changed measured PCC more than input range did
Initial test used `dtype=torch.bfloat16, [0, 1]`. Switching to `[-1, 1]` made
all 192 tests pass (PCC > 0.99). The PCC drop only showed up after switching
to `dtype=torch.float32`. Reason: with bf16 inputs the CPU golden runs in
bf16 too — both sides round identically, so the test measured round-trip
consistency, not precision. Sweeps generates fp32 inputs by default
(`torch.rand(shape)` with `dev_data_format=None`), so its PCC values reflect
**bf16 device vs fp32 golden** — a real precision benchmark. Single-test
match after the dtype fix: tt-xla `pcc=0.5004`, sweeps `pcc=0.5001` (within
seed noise) for the same test_id.

### 2. PCC is scale-invariant but mean-sensitive
With `[0, 1]` inputs the matmul output had a large positive mean and small
variance → bf16 truncation looked correlated with the signal → PCC fell to
0.84–0.99. With `[-1, 1]` (or zero-mean Gaussian) the output is centered
and PCC is dominated by signal, not bias. Always use **zero-mean inputs**
when PCC is the gate. Empirically (see `pcc_report_uniform.md` vs
`pcc_report_realistic.md`), once inputs are zero-mean the actual
distribution shape (uniform `[-1, 1]` vs mixture `N + outliers`) is
**negligible for PCC**: identical failure set, identical group structure,
PCC values differ in the fourth decimal (~0.0005 across both regimes).
The atol values do change (mixture has ~5–10× larger atol because of the
outlier tail), but PCC is the gating signal and PCC is invariant.

### 3. `weight_dtype` / `math_fidelity` / `fp32_dest_acc_en` don't flow
Within any `(shape, input_source, opt_level)` group, all 16 combinations of
those three compiler options produced **identical PCC to 16 digits** in both
the baseline (bf16) and realistic (fp32+mixture) snapshots. Something in
the stack isn't honoring these knobs: it could be tt-xla not forwarding
them, the StableHLO not encoding them, a tt-mlir pass dropping them, or
the tt-metal kernel ignoring them. They are listed in
[`compiler_config.py`](../../../../infra/testers/compiler_config.py) as
valid options but had no effect on numerics in this code path. Triage
by dumping IR at each stage (see follow-ups).

### 4. Two independent precision symptoms at opt=2 — both shape-conditioned
With realistic inputs, two distinct symptoms appear at `opt_level=2`, on
overlapping but not identical shape sets. **Both are symptoms — the actual
defect location (torch_xla SHLO generation, tt-mlir passes, tt-metal kernel
selection, or a mix) is unknown until we dump IR at each stage.**

**Symptom A — `add(x,x)*add(y,y)` prelude:** `FROM_ANOTHER_OP` (with the
prelude) collapses on a set of shapes; `FROM_HOST` on the same shapes is
fine. Analytically `add(x,x)*add(y,y) = 4·x·y` is a linear scale, so PCC
should match the plain matmul; the gap means *something* in the stack
treats the add subgraph differently at opt=2. Could be SHLO emitted
differently, an opt-2 fusion pass kicking in only when there's a producer,
or kernel-config selection branching on graph shape. `opt=0` agrees
regardless, so the divergence is opt-level-gated.

**Symptom B — plain matmul at large N:** `FROM_HOST` (no prelude) also
collapses, on a different set of shapes (typically large N). Same
caveats: could be wrong tile layout chosen for that N, a sharding pass
that misbehaves, or a less precise kernel variant selected. No prelude is
needed to trigger this — it's a bare matmul precision symptom.

Per-shape behavior in the current `_SHAPE_PAIRS` grid:

| Shape `(B,S,K)x(K,N)` | FROM_ANOTHER_OP × opt=2 | FROM_HOST × opt=2 |
|---|---|---|
| `(32,128,1024)x(1024,2048)` | 0.500 — Bug A | >0.99 |
| `(32,128,2304)x(2304,1024)` | 0.500 — Bug A | >0.99 |
| `(32,128,2560)x(2560,1024)` | 0.500 — Bug A | >0.99 |
| `(32,128,1024)x(1024,3072)` | 0.500 — Bug A | 0.500 — Bug B |
| `(32,128,4864)x(4864,896)` | >0.99 (control) | >0.99 (control) |

`(1024, 3072)` exhibits **both bugs at once**: the prelude collapses
FROM_ANOTHER_OP, and the plain matmul independently collapses FROM_HOST.
`(4864, 896)` is the only fully-clean shape in the grid (large K, small
non-power-of-2-times-tile N = 896 = 7·128).

Sweeps confirms the same split: the failing shapes appear in the relevant
xfail txt for `matmul_mp`, the passing ones don't. So the predicate has to
enumerate failing shapes per input source rather than apply universally.
The discriminator is probably tile-alignment of `N` and/or its absolute
magnitude, but the grid is still too thin to pin down. This is the most
actionable finding — two distinct real bugs.

### 5. Sweeps' AutomaticValueChecker effectively decides on PCC alone
Sweeps configures the comparator with `(pcc=0.99, rtol=1e-2, atol=1e-2)`
(both PCC and allclose), but `check_pcc_error_level` reclassifies failures
into low / medium / high range buckets **based on PCC value only**. Allclose
failures don't make it into the xfail txt files. Implication: matching
sweeps' xfail set requires PCC as the gate, not allclose. Enabling allclose
here made all 192 fail on outlier-scaled atol differences — too noisy to be
a useful predicate.

### 6. `clear_torchxla_computation_cache` fixture == `--forked` for this suite
`compare_runs.sh` proved the conftest cache-clear fixture produces a
byte-for-byte identical outcome sequence to `--forked`, ~2.5× faster. The
JIT build cache below it (TT-metal) stays warm and is not a correctness
issue. Earlier confusion (all-pass when caches were stale) came from running
without **either** isolation method.

### 7. Sweeps xfail txt files are snapshots, not authoritative
The sweeps xfail lists for matmul_mp under
`third_party/tt_forge_sweeps/.../xfail/` came from older runs. The current
hardware + toolchain pass many entries that are listed there, and fail some
that aren't. Treat them as a useful starting point for the predicate but
re-measure rather than trust.

## Input design options

Picking inputs is a series of independent knobs. Each row is what the test
**actually feeds into the model**; the table compares what they expose.

| Knob | Option | What it measures | Used by |
|---|---|---|---|
| dtype | `bf16` | round-trip consistency (CPU and device both round) | original tt-xla op tests |
| dtype | `fp32` | precision loss of bf16 hardware path vs fp32 truth | sweeps default, **current test** |
| range/dist | uniform `[0, 1]` | biases output mean → PCC penalty | `run_op_test_with_random_inputs` default |
| range/dist | uniform `[-1, 1]` | zero-mean, easy to reason about | sweeps `ValueRanges.SMALL` |
| range/dist | Gaussian `N(0, σ)` | matches post-LayerNorm activations and Kaiming weights | typical ML benchmarks |
| range/dist | mixture `N(0, σ) + ε·N(0, kσ)` | adds outlier features (LLM regime) | **current test** |
| RHS scale | `1` (same as LHS) | unrealistic; output magnitudes ~50× real | sweeps default |
| RHS scale | `1/√K` (Kaiming) | matches initialization of trained weights | **current test** |
| RHS scale | sampled-empirical | match a specific model's weight stats | offline calibration only |

### Practical guidance

| Goal | Recipe |
|---|---|
| Match sweeps numbers 1:1 for a specific test_id | `fp32, uniform [-1, 1]` on both operands |
| Detect mixed-precision regressions on realistic models | **current**: `fp32, mixture N(0, σ) + 1% N(0, 10σ)` with `σ_LHS=1`, `σ_RHS=1/√K` |
| Detect optimizer regressions in pure bf16 path | `bf16, uniform [-1, 1]` (the baseline snapshot) |
| Sanity check a kernel change is bit-identical to before | `bf16, fixed seed, allclose-only check` |
| Investigate `add+matmul` opt=2 bug | run `FROM_ANOTHER_OP` and `FROM_HOST` side-by-side at opt=0 vs opt=2, any reasonable inputs |

### Knobs intentionally **not** parameterized in this test

- Random seed: stays at the conftest-default (`torch.manual_seed(0)`) so
  reports are reproducible. Sweeps' generator passes its own `generator=…`;
  numerical agreement to sweeps is within seed noise, which is fine for
  characterizing PCC regime but not for bit-exact replication.
- Per-test input range override: there is no `input_profile` parametrize
  axis. Adding it would quadruple test count for marginal new signal. If
  needed later, generate snapshots by editing `_mixture_normal` and saving
  the resulting report under a distinct suffix.

## Cross-validation against sweeps export (`sweeps_export.csv`)

After getting a pivot CSV of the full sweeps matmul_mp suite — 1072 rows
across 134 shapes and 4 CI runs — the following holds:

### Run timeline (provided by Vladimir, 2026-06)

| Run ID | tt-xla version | Date | Seed |
|---|---|---|---|
| `26165109284` | `1.1.0.dev20260429` | 2026-05-20 | (Test matmul_mp) |
| `26165164203` | `1.1.0.dev20260430` | 2026-05-20 | (Test matmul_mp) |
| `26171346075` | `1.1.0.dev20260428` | 2026-05-20 | (Test matmul_mp) |
| `26756567247` | `1.2.0.dev20260601` | 2026-06-01 | INPUT_SEED=44 |

Sibling sweeps runs at the same `1.2.0.dev20260601003137` exist for
`INPUT_SEED=42`/`43` and `RANDOM_SEED=42/43/44` but aren't in this CSV
snapshot.

### My grid is a subset of the sweeps grid

All 5 shapes in `_SHAPE_PAIRS` appear in the CSV (8 rows each: 2 input
sources × 4 runs). The per-shape minimum-PCC profile at `opt=2`:

| Shape | FROM_ANOTHER_OP across 4 runs | FROM_HOST across 4 runs |
|---|---|---|
| `(1024,2048)` | 0.345, 0.345, 0.345, **0.5012** | 1.000, 1.000, 1.000, 1.000 |
| `(2304,1024)` | 1.000, 1.000, 1.000, **0.4998** | 1.000, 1.000, 1.000, 1.000 |
| `(2560,1024)` | 1.000, 1.000, 1.000, **0.4996** | 1.000, 1.000, 1.000, 1.000 |
| `(1024,3072)` | 1.000, 1.000, 1.000, **0.500** (Bug A, new) | **0.500, 0.500, 0.500, 0.500** (Bug B, persistent) |
| `(4864,896)` | 1.000, 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000, 1.000 |

`(1024,2048)` collapsed already in 1.1.0 (PCC ~0.345 even worse than my
1.2.0 measurement of 0.500); `(2304,1024)`, `(2560,1024)`, and the
FROM_ANOTHER_OP path of `(1024,3072)` collapsed only in the 1.2.0 upgrade.
The FROM_HOST path of `(1024,3072)` is broken in all 4 sweeps runs — a
**pre-existing** plain-matmul precision bug. `(4864,896)` is unaffected
throughout (control shape).

My tt-xla measurements (`pcc_report_realistic.md`) line up with the
`1.2.0.dev20260601` run to 3 decimals — confirming the test reproduces
what sweeps sees.

### What the upgrade broke (`1.1.0` → `1.2.0`)

Three `(src, shape)` pairs are catastrophic (PCC < 0.7 at opt=2) only in
the 1.2.0 run — all on the prelude (FROM_ANOTHER_OP) path:

- `FROM_ANOTHER_OP × ((32,128,1024),(1024,3072))` — in `_SHAPE_PAIRS`
- `FROM_ANOTHER_OP × ((32,128,2304),(2304,1024))` — in `_SHAPE_PAIRS`
- `FROM_ANOTHER_OP × ((32,128,2560),(2560,1024))` — in `_SHAPE_PAIRS`

This is a real **bisect target** between dev builds
`1.1.0.dev20260428` and `1.2.0.dev20260601` — ~5 weeks of commits across
**tt-xla, tt-mlir, and tt-metal** to walk. Any of those repos could host
the regressor; bisect the combined timeline rather than picking one. The
test now covers all three regression cases.

### Persistent FROM_HOST failures

22 `(src, shape)` pairs are catastrophic in **every** sweeps run. Of those,
3 are `FROM_ANOTHER_OP` and **19 are `FROM_HOST`** on shapes with large
output dim N (`3072`, `8192`, `11008`, `16384`, `9216`, etc.). The
"FROM_HOST + opt=2 passes" conclusion from earlier was **only true for
the small-N shapes initially in our grid**; globally FROM_HOST at opt=2
is broken on many shapes too. The grid now carries `(1024, 3072)` as a
representative — the smallest of the failing FROM_HOST set, so cheapest
to run, and conveniently also a `FROM_ANOTHER_OP` regression so a single
shape exercises both bugs. The other 18 persistent FROM_HOST cases are
not in the grid; add more shapes here if we want broader coverage of the
plain-matmul-precision bug.

## Open follow-ups

### Triage: localize Symptom A and Symptom B in the compiler stack

Currently we know *where* (shape + opt=2 + optional prelude) but not
*who* (torch_xla, tt-xla PJRT plugin, tt-mlir passes, or tt-metal kernel
selection). Concrete steps:

1. Run with `CompilerConfig.export_path=<dir>` and
   `export_model_name="matmul_mp"` for three side-by-side cases on the
   same shape, e.g. `(32,128,1024)x(1024,3072)`:
   - `FROM_ANOTHER_OP × opt=2` (Symptom A, fails)
   - `FROM_HOST × opt=2` (Symptom B, fails on this shape; passes on the
     smaller-N shapes)
   - `FROM_ANOTHER_OP × opt=0` (passes)
2. Diff the StableHLO dumps:
   - If SHLO differs between the failing and passing cases beyond the
     expected `add` ops / opt-flag, the regression sits in
     **torch_xla → SHLO emission** or **tt-xla's compile-option
     handoff** (i.e. tt-xla is sending tt-mlir a graph it shouldn't).
   - If SHLO is structurally identical (modulo `add` for the prelude
     variant), the regression is downstream.
3. If SHLO is clean, diff the TTIR dumps before/after the optimizer
   passes. The pass whose output first shows the corruption is the
   culprit — locates it in **tt-mlir**.
4. If TTIR looks correct, dump TTNN / kernel-config and check which
   matmul kernel variant and tile layout each case selects. Different
   variant or different fp32-accumulation setting between cases points
   to **tt-mlir lowering** or **tt-metal kernel selection**.

Until those dumps exist, "tt-mlir issue" vs "tt-xla issue" is a guess.

### Once the stack location is known, file targeted issues

- **Symptom A** (`add → matmul` at opt=2): collapses PCC to ~0.5 on
  `(K,N) ∈ {(1024,2048), (1024,3072), (2304,1024), (2560,1024)}` but
  passes on `(4864, 896)`. What about the failing shapes triggers the
  divergence — N alignment? Tile geometry? Pre-allocation? — and how
  to gate or fix the responsible transform.
- **Symptom B** (plain matmul at opt=2, large N): `(1024,3072)` is the
  smallest case in the grid; sweeps has 18 more at N ∈ {3072, 8192,
  11008, 16384, 24576, ...}. Persistent across all 4 sweeps runs and
  independent of Symptom A.

### Other follow-ups

- Bisect the 1.1.0 → 1.2.0 regression: `(2304,1024)`, `(2560,1024)`,
  and the FROM_ANOTHER_OP path of `(1024,3072)` collapsed only at
  `1.2.0.dev20260601` (date range `2026-04-28` → `2026-06-01`).
  `(1024,2048)` was already broken in 1.1.0, so the regressor is
  shape-conditioned on top of a pre-existing bug. The bisect window
  spans tt-xla, tt-mlir, and tt-metal — walk all three.
- Pull the sibling 1.2.0 seeds (sweeps runs `26756536362` /
  `26756551955` for INPUT_SEED=42/43) into the CSV and re-run
  `compare_with_sweeps_export.py` to confirm whether the 1.2.0 collapse
  is seed-stable or just our seed.
- Investigate why `weight_dtype` / `math_fidelity` / `fp32_dest_acc_en`
  make no measurable difference for this op path. Could be tt-xla not
  encoding them in SHLO/compile-options, tt-mlir dropping them, or the
  selected kernel ignoring them. Same `export_path` triage applies.
- Expand the shape grid further to characterize the failing-vs-passing
  boundary. Currently 1 passing point isn't enough to choose between
  "N < 1024", "N not a power-of-2-times-128", "K < some threshold", etc.
