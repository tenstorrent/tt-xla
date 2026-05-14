# Validation matrix for tt-mlir host-bfp-packing change

This branch (`dgolubovic/host-bfp-packing-validation`, off `main`) is the
integration test bed for the tt-mlir change in branch
`dgolubovic/host-bfp-packing` (off `7a32a979c5fc...`). It contains:

- The **CPU-teacher-forcing infrastructure** needed to run gpt-oss-120b
  end-to-end accuracy tests (`accuracy_prompt`,
  `accuracy_max_decode_tokens`, `accuracy_use_chat_template` plumbed
  through `test_llm` → `benchmark_llm_torch_xla` → `init_accuracy_testing`
  → new `generate_reference_from_prompt`).
- A **multi-prompt sweep** test for gpt-oss-120b (4 prompts × ~60 tokens
  each, designed to be invoked one prompt at a time to avoid
  per-pytest-session memory leaks).
- This document — the regression matrix protocol + before/after table.

## What's being validated

tt-mlir branch `dgolubovic/host-bfp-packing` adds two commits on top of
`7a32a979c5fc`:

1. `c9e66f0c0` — Adds `ttnn.to_dtype` op; emits
   `from_device + to_dtype + to_device` chain for bfp4 weight casts.
2. `09fb61877` — Extends the same chain to bfp8 weight casts.

Goal: confirm the change doesn't regress accuracy on any existing
`test_llms.py` test that depends on bfp4 or bfp8 weight conversion.

If a regression appears with both commits ON, the next diagnostic step
is to run that same test with **only commit 1** (bfp4) cherry-picked —
i.e. revert just the bfp8 change. That isolates whether the regression
is from the bfp4 path (unlikely; we have +17pt evidence it helps
gpt-oss-120b) or the bfp8 path (untested at scale before this matrix).

## Protocol

Each test is run twice. The only difference between runs is the
`TT_MLIR_VERSION` pinned in `third_party/CMakeLists.txt`:

| Run | `TT_MLIR_VERSION` | Description |
|---|---|---|
| **OFF** | `7a32a979c5fcf19203015f6baa93dbd1effab6ac` | tt-mlir at the base SHA (no host-pack changes; current main). |
| **ON** | `09fb61877` (tip of `dgolubovic/host-bfp-packing`) | both commits applied. |

For a regression that needs further isolation:

| Run | `TT_MLIR_VERSION` | |
|---|---|---|
| **bfp4-only** | `c9e66f0c0` (commit 1 of the branch) | only the bfp4 chain. |

### Switching SHAs and rebuilding

Edit `third_party/CMakeLists.txt`:

```cmake
set(TT_MLIR_VERSION "<sha>")
```

Then rebuild:

```bash
docker exec --workdir /home/dgolubovic/repos/tt-xla tt-xla-ird-dgolubovic bash -lc '
  source venv/activate &&
  rm -rf build/_deps/tt-mlir* third_party/tt-mlir &&
  cmake -G Ninja -B build &&
  cmake --build build
'
```

(The `rm -rf build/_deps/tt-mlir*` and `third_party/tt-mlir` purge
forces ExternalProject to re-fetch tt-mlir at the new SHA.)

### Running an accuracy test

Each test must be a fresh pytest invocation (don't run multiple
parametrize variants in one session — see `prompt_sweep` test
docstring for why):

```bash
docker exec --workdir /home/dgolubovic/repos/tt-xla tt-xla-ird-dgolubovic bash -lc '
  source venv/activate &&
  cd tests/benchmark &&
  python -m pytest -svv "test_llms.py::<test_name>[<params>]" --accuracy-testing
'
```

Capture the `Token accuracy: TOP1=…, TOP5=…` line from each run.

## Tests in scope (matrix rows)

### Galaxy 4×8 — explicit bfp4 weight overrides

These exercise the bfp4 chain directly (commit 1 effect).

| Test | Notes |
|---|---|
| `test_gpt_oss_120b_tp_galaxy_batch_size_64` | original prompt-based ("What are the prime factors of 1?"), 30 decode tokens. |
| `test_gpt_oss_120b_tp_galaxy_totc_teacher_forcing[chat]` | TOTC chat-wrapped, 64 decode tokens. tt-metal reference: ~87.5%. |
| `test_gpt_oss_120b_tp_galaxy_totc_teacher_forcing[raw]` | TOTC raw, 64 decode tokens. tt-metal reference: ~84.4%. |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[totc_opening]` | "It was the best of times…" |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[pride_and_prejudice]` | Austen opening |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[hamlet_soliloquy]` | "To be, or not to be…" |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[photosynthesis]` | Wikipedia-style technical |

### Galaxy 4×8 — bfp8 default (no explicit bfp4 override on experts)

These exercise the bfp8 chain (commit 2 effect). All the rest of
`test_llms.py`'s Galaxy tests fall here.

| Test | Notes |
|---|---|
| `test_llama_3_1_70b_tp_galaxy` | Llama 3.1 70B Galaxy, default bfp8. |
| `test_gpt_oss_20b_tp_galaxy_batch_size_64` | gpt-oss-20b Galaxy, default bfp8. |
| `test_gpt_oss_120b_tp_dp_galaxy_batch_size_128` | gpt-oss-120b TP+DP, batch 128. |
| `test_gpt_oss_120b_tp_qb2` | gpt-oss-120b qb2 variant. |

### Galaxy 4×8 — Llama 70B with explicit bfp4 (gate_proj/up_proj)

| Test | Notes |
|---|---|
| `test_llama_3_1_70b_tp` | Llama 3.1 70B with bfp_bf4 on `gate_proj.weight` + `up_proj.weight`. |

### Single-chip / smaller TP — bfp8 default

| Test | |
|---|---|
| `test_llama_3_1_8b_tp` | Llama 3.1 8B TP. |
| `test_llama_3_1_8b_instruct_tp` | Llama 3.1 8B Instruct TP. |
| `test_mistral_7b_tp` | Mistral 7B TP. |
| `test_qwen_2_5_7b` | Qwen2.5 7B single-chip. |
| `test_qwen_3_8b` | Qwen3 8B single-chip. |
| `test_phi3_5_mini` | Phi3.5 Mini single-chip. |

(Owner: pick a representative subset; full single-chip pass is overkill
for first-cut validation.)

## Comparison table — fill in as runs complete

Format: `TOP1 / TOP5` (or `n/a` if not run / OOM / etc.). Mark
regressions in **bold**.

### Galaxy bfp4-explicit (gpt-oss-120b)

Validated on Galaxy 4×8 on 2026-04-27. Build: tt-xla `fa7661792`, tt-mlir
tip `09fb61877` (= `c9e66f0c0` bfp4 + `09fb61877` bfp8 on top of
`7a32a979c`). Each test was a fresh pytest invocation. OFF and ON runs
used the same freshly-built tt-xla; only `TT_MLIR_VERSION` differed.
Both phases ran the full matrix back-to-back without container restart
between rows.

| Test | OFF (`7a32a979c`) | ON (`09fb61877`) | Δ TOP1 |
|---|:---:|:---:|---:|
| `test_gpt_oss_120b_tp_galaxy_batch_size_64` | 39.06% / 71.88% | 87.50% / 100.00% | **+48.44** |
| `test_gpt_oss_120b_tp_galaxy_totc_teacher_forcing[chat]` | 65.62% / 90.62% | 81.25% / 96.88% | **+15.63** |
| `test_gpt_oss_120b_tp_galaxy_totc_teacher_forcing[raw]` | 64.06% / 89.06% | 79.69% / 100.00% | **+15.63** |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[totc_opening]` | 78.12% / 98.44% | 95.31% / 100.00% | **+17.19** |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[pride_and_prejudice]` | 68.75% / 96.88% | 95.31% / 100.00% | **+26.56** |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[hamlet_soliloquy]` | 78.12% / 98.44% | 90.62% / 100.00% | **+12.50** |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[photosynthesis]` | 68.75% / 96.88% | 89.06% / 100.00% | **+20.31** |

**4-prompt sweep mean:** OFF 73.44% / 97.66% → ON 92.58% / 100.00%
(**+19.14pt TOP1**, **+2.34pt TOP5**) — matches the runtime-hack
reference (92.58% / 100%) exactly, confirming the graph-level chain is
semantically equivalent.

### Galaxy bfp8 default

| Test | OFF (`7a32a979c`) | ON (`09fb61877`) | Δ TOP1 |
|---|:---:|:---:|---:|
| `test_llama_3_1_70b_tp_galaxy` | gated HF (no token) | not run | n/a |
| `test_gpt_oss_20b_tp_galaxy_batch_size_64` | 79.69% / 96.88% | 82.81% / 98.44% | **+3.12** |
| `test_gpt_oss_120b_tp_dp_galaxy_batch_size_128` | TBD | TBD | TBD |
| `test_gpt_oss_120b_tp_qb2` | TBD | TBD | TBD |

### Llama bfp4 (gate_proj/up_proj)

| Test | OFF | ON | Δ TOP1 |
|---|:---:|:---:|---:|
| `test_llama_3_1_70b_tp` | TBD | TBD | TBD |

### Single-chip representatives (bfp8 default)

| Test | OFF | ON | Δ TOP1 |
|---|:---:|:---:|---:|
| `test_llama_3_1_8b_tp` | TBD | TBD | TBD |
| `test_llama_3_1_8b_instruct_tp` | TBD | TBD | TBD |
| `test_mistral_7b_tp` | TBD | TBD | TBD |
| `test_qwen_2_5_7b` | TBD | TBD | TBD |
| `test_qwen_3_8b` | TBD | TBD | TBD |
| `test_phi3_5_mini` | TBD | TBD | TBD |

## Pass criteria

- **Pass**: every row's `Δ TOP1 ≥ -0.5%` (decode-noise floor on a
  64-token sequence). Larger negative deltas trigger isolation runs.
- **Win**: `Δ TOP1 ≥ +1%`, especially on bfp4-explicit gpt-oss rows
  (we expect +5 to +20 pt there).

If any row fails the pass criterion AND it's a bfp8-default test, run
the same test against `c9e66f0c0` (bfp4-only commit) to determine if
the regression is from the bfp8 chain (commit 2) — in that case revert
commit 2 and re-PR with bfp4 only.

## Validation summary (2026-04-27, Galaxy 4×8)

Across 8 gpt-oss-120b/20b tests on Galaxy:

- Every test improved or held; **no regressions**.
- gpt-oss-120b bfp4 paths: gains of **+12.5 to +48.4 TOP1**.
- gpt-oss-20b bfp8 default: small but real gain (**+3.12 TOP1**).
- 4-prompt sweep mean on gpt-oss-120b: **+19.14pt TOP1, +2.34pt TOP5**.
- 4-prompt sweep mean matches the `TT_BFP4_HOST_PACK=1` runtime-hack
  reference (92.58 / 100 in both), confirming the graph rewrite emits
  the intended `from_device → to_dtype → to_device` chain.

Note: a tt-mlir-level fix was required during this validation —
`TTNN_ToDtypeOp` needed the `OpModelExempt` trait (the original
declaration as a plain `TTNN_Op` made tablegen emit references to
`getOpConstraints`/`getOpRuntime` symbols that nobody implemented and
the link failed). The fix has been folded back into the bfp4 commit;
the matrix above was rerun on the cleaned-up history.

## Already validated (carried from runtime-hack experiments before
this branch existed)

These results were obtained by toggling the runtime hack
(`TT_BFP4_HOST_PACK=1`) inside `runtime/lib/ttnn/operations/layout/typecast.cpp`
on the previous branch (`dgolubovic/gpt-oss-prompt-based-accuracy`).
The graph-level approach was confirmed byte-identical to the runtime
hack via per-matmul PCC (0.9899 in both) and Galaxy e2e (89.06% / 100%
on the same prompt).

| Test | OFF | ON (hack/graph) | Δ TOP1 |
|---|:---:|:---:|---:|
| Per-matmul PCC (gate_up bfp4, layer 18, expert 0) | 0.9849 | 0.9899 | +0.005 |
| `totc_teacher_forcing[raw]` | 64.06% / 90.62% | 81.25% / 98.44% | +17.19 |
| `refpt_bfp4_raw` (TOTC long-form, 64+64) | n/a (no baseline ran) | 82.81% / 98.44% | n/a |
| 4-prompt sweep mean | n/a | 92.58% / 100% | n/a |
