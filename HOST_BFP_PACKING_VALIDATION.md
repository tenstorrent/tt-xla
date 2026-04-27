# Host BFP packing — validation results

Galaxy and single-chip accuracy regression matrix for tt-mlir branch
`dgolubovic/host-bfp-packing` (commits `c9e66f0c0` + `09fb61877` on top
of `7a32a979c`).

- **OFF**: tt-mlir at `7a32a979c5fcf19203015f6baa93dbd1effab6ac` (base; no host-pack changes).
- **ON**:  tt-mlir at `09fb61877` (both commits applied).

A/B done by toggling `TT_MLIR_VERSION` in `third_party/CMakeLists.txt`
between the two SHAs and rebuilding. Each row is a fresh pytest
invocation. 64 decode tokens for accuracy comparison.

## Galaxy 4×8 — bfp4 weight overrides

These tests declare bfp4 overrides on MLP weights (MoE expert weights
on gpt-oss; `gate_proj` + `up_proj` on llama-3-1-70b), so they exercise
the new `from_device → to_dtype → to_device` chain on every overridden
matmul. This is where the change is intended to help.

| Test | OFF TOP1 / TOP5 | ON TOP1 / TOP5 | Δ TOP1 | Δ TOP5 |
|---|---|---|---|---|
| `test_gpt_oss_120b_tp_galaxy_totc_teacher_forcing[raw]` | 64.06% / 89.06% | 79.69% / 100.00% | +15.63% | +10.94% |
| `test_gpt_oss_120b_tp_galaxy_totc_teacher_forcing[chat]` | 65.62% / 90.62% | 81.25% / 96.88% | +15.63% | +6.26% |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[totc_opening]` | 78.12% / 98.44% | 95.31% / 100.00% | +17.19% | +1.56% |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[pride_and_prejudice]` | 68.75% / 96.88% | 95.31% / 100.00% | +26.56% | +3.12% |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[hamlet_soliloquy]` | 78.12% / 98.44% | 90.62% / 100.00% | +12.50% | +1.56% |
| `test_gpt_oss_120b_tp_galaxy_prompt_sweep_bfp4_raw[photosynthesis]` | 68.75% / 96.88% | 89.06% / 100.00% | +20.31% | +3.12% |
| `test_gpt_oss_120b_tp_galaxy_batch_size_64` | 39.06% / 71.88% | 87.50% / 100.00% | +48.44% | +28.12% |
| `test_gpt_oss_20b_tp_galaxy_batch_size_64` | 79.69% / 96.88% | 82.81% / 98.44% | +3.12% | +1.56% |
| `test_llama_3_1_70b_tp_galaxy` | 90.62% / 98.44% | 92.19% / 100.00% | +1.57% | +1.56% |

4-prompt sweep mean on gpt-oss-120b: **OFF 73.44% / 97.66% → ON 92.58% / 100.00%**
(+19.14pt TOP1, +2.34pt TOP5).

## Single-chip — no bfp4 weight overrides

These tests do not declare bfp4 overrides on any matmul, so they don't
exercise the new chain. They were run only as a regression check — to
confirm the change does not introduce drift on the default-dtype paths.

| Test | OFF TOP1 / TOP5 | ON TOP1 / TOP5 | Δ TOP1 | Δ TOP5 |
|---|---|---|---|---|
| `test_llama_3_2_1b` | 84.38% / 98.44% | 82.81% / 98.44% | -1.57% | 0.00% |
| `test_llama_3_2_3b` | 84.38% / 100.00% | 87.50% / 100.00% | +3.12% | 0.00% |
| `test_llama_3_1_8b` | 82.81% / 92.19% | 89.06% / 100.00% | +6.25% | +7.81% |
| `test_mistral_7b` | 100.00% / 100.00% | 98.44% / 100.00% | -1.56% | 0.00% |
| `test_qwen_2_5_0_5b` | 90.62% / 100.00% | 90.62% / 100.00% | 0.00% | 0.00% |
| `test_qwen_2_5_1_5b` | 87.50% / 98.44% | 85.94% / 98.44% | -1.56% | 0.00% |
| `test_qwen_2_5_3b` | 92.19% / 100.00% | 90.62% / 100.00% | -1.57% | 0.00% |
| `test_qwen_2_5_7b` | 81.25% / 90.62% | 79.69% / 90.62% | -1.56% | 0.00% |
| `test_qwen_3_0_6b` | 95.31% / 100.00% | 95.31% / 100.00% | 0.00% | 0.00% |
| `test_qwen_3_1_7b` | 96.88% / 100.00% | 95.31% / 100.00% | -1.57% | 0.00% |
| `test_qwen_3_4b` | 87.50% / 98.44% | 89.06% / 100.00% | +1.56% | +1.56% |
| `test_qwen_3_8b` | 98.44% / 100.00% | 98.44% / 100.00% | 0.00% | 0.00% |

Single-chip deltas are at or below the 64-token decode-noise floor
(±1.6pt corresponds to a single token flip per row) — no regressions.
