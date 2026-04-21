# Wan 2.2 TI2V-5B E2E Test — Noise Output Fix Design

**Status:** Design
**Date:** 2026-04-21
**Author:** mstojkovic
**Target files:** `tests/torch/models/wan2_2/test_wan22_e2e.py`, `tests/torch/models/wan2_2/shared.py`

## Problem

`tests/torch/models/wan2_2/test_wan22_e2e.py` produces pure TV-static output (with faint color tinting) even on CPU with `NUM_STEPS=50`. The test hand-rolls a pipeline instead of delegating to `diffusers.WanPipeline`, and it has diverged from both canonical reference implementations.

## References Consulted

- HuggingFace model config: `Wan-AI/Wan2.2-TI2V-5B-Diffusers/model_index.json`, `scheduler_config.json`, `transformer/config.json`
- HF diffusers pipelines (in venv): `diffusers/pipelines/wan/pipeline_wan.py`, `pipeline_wan_i2v.py`
- Original Wan-Video repo: `github.com/Wan-Video/Wan2.2/wan/text2video.py`, `textimage2video.py`, `modules/t5.py`

## Canonical configuration (ground truth for TI2V-5B)

From `model_index.json`:
- Scheduler class: `UniPCMultistepScheduler` (NOT `FlowMatchEulerDiscreteScheduler`)
- `expand_timesteps: true` — DiT receives per-patch timesteps
- `boundary_ratio: null`, `transformer_2: [null, null]` — single-stage, one transformer

From `scheduler_config.json`:
- `flow_shift: 5.0`, `prediction_type: "flow_prediction"`, `use_flow_sigmas: true`, `solver_order: 2`

From `transformer/config.json`:
- `in_channels: 48`, `patch_size: [1, 2, 2]`, `text_dim: 4096`, `num_layers: 30`

## Root-cause fixes (in priority order)

### 🔴 CRITICAL

**C1. Wrong scheduler class** — `test_wan22_e2e.py:269-272`
Test loads `UniPCMultistepScheduler`'s config (`flow_shift=5.0`, flow sigmas, flow_prediction) into `FlowMatchEulerDiscreteScheduler`. Mismatched kwargs are silently dropped; the sigma/timestep trajectory is wrong, which is sufficient on its own to produce pure-noise output.
- **Fix:** import and use `UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")`.

**C2. Prompt embeds not zero-padded past real token length** — `test_wan22_e2e.py:92-107` (`_encode_prompt`)
Both references truncate each embed to `mask.gt(0).sum(dim=1)` and re-pad with explicit zeros. UMT5's feed-forward + residuals produce non-trivial values at pad positions; DiT cross-attention is trained expecting zeros there.
- **Fix:** inside `_encode_prompt`, compute `seq_lens` from the attention mask, slice each sample, re-pad with `new_zeros` back to max-length. Output shape (1, 512, 4096) stays the same; pad positions are now explicit zero.

### 🟠 HIGH

**H3. Latents sampled in bfloat16 instead of float32** — `test_wan22_e2e.py:110-119` (`_init_latents`)
Both references sample in `float32` (diffusers: `randn_tensor(..., dtype=torch.float32)`; Wan repo: `torch.randn(..., dtype=torch.float32)`). bf16's ~8-bit mantissa noticeably quantizes a Gaussian and drifts the denoising trajectory.
- **Fix:** sample in `float32`. Keep latents in float32 through the denoise loop; DiT wrapper accepts any input dtype.

**H4. Timestep tensor in bfloat16** — `test_wan22_e2e.py:157`
`torch.full((1, num_patches), float(t), dtype=torch.bfloat16)`. Since `t` can reach ~1000, bf16 quantizes to ~8-unit increments — wrong sinusoidal time embeddings.
- **Fix:** build timestep in `float32`. Matches reference's scheduler timestep dtype.

**H5. `GUIDANCE_SCALE = 1.0` default (CFG disabled)** — `test_wan22_e2e.py:45`
Both references default to 5.0. Not the root cause of pure noise, but a quality contributor and it short-circuits the negative-prompt path entirely.
- **Fix:** set default to 5.0.

**H6. Missing `prompt_clean`** — `test_wan22_e2e.py:92-107`
Diffusers applies `ftfy.fix_text` + `html.unescape` + whitespace collapse before tokenizing.
- **Fix:** port the three-line helper (ported from diffusers `pipeline_wan.py:78-93`). `ftfy` dependency is soft.

### 🟡 MEDIUM (i2v-only, orthogonal to current t2v noise bug)

**M7. i2v: first-frame tokens should get timestep 0, not timestep t** — `test_wan22_e2e.py:149-157`
Reference i2v builds a `first_frame_mask` (0 at first latent frame, 1 elsewhere) and applies `temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()`. Current code uses uniform `t`.
- **Fix:** when `image_latent is not None`, build the spatial mask (num_latent_frames × latent_h/2 × latent_w/2) with `mask[0] = 0`, flatten, multiply by `float(t)`.

### 🟢 LOW (nit, fix while touching)

**L8. Hardcoded `48` latent channels** — `test_wan22_e2e.py:113`
- **Fix:** pull from `dit.config.in_channels` at call site in `_run` (we already have `dit_wrapper`).

## What stays unchanged

- Per-component TT hook structure (`TT_TEXT_ENCODER`, `TT_VAE_ENCODER`, `TT_DIT`, `TT_VAE_DECODER`) and all shard-spec functions.
- All `load_*` functions, all wrappers, `run_component`.
- CFG formula `v_neg + scale * (v_pos - v_neg)` — matches both references.
- VAE un-scale math `latents / (1/std) + mean = latents * std + mean` — matches reference.
- `num_patches = T * (H // 2) * (W // 2)` patchification — matches `patch_size=[1,2,2]`.
- UMT5 weight-tying workaround (`shared.py:69`) — still needed for the pinned transformers version.
- Pytest contract: file exists, non-empty — eyeball-quality smoke test only.

## Default prompt update

- **Current:** "A gentle ocean wave rolling across a quiet sandy beach at sunrise, cinematic, high detail" — three descriptors, stylistic tags, motion requirement.
- **New default:** `"A red apple on a white table"` — single subject, single color, single surface, no motion, deeply in-distribution. Failure is immediately obvious by eye; success is unambiguous.
- **Fallback ladder** for debugging (in order of increasing complexity): `"A cat sitting on grass"` → `"A dog running on a beach"` → Wan 2.2 README canonical `"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."`
- **Negative prompt:** unchanged (`"low quality, blurry, distorted, watermark, text"`).

## Scope of changes

| File | Changes |
|---|---|
| `shared.py` | Add `prompt_clean` helper (+ `basic_clean`, `whitespace_clean`). No other changes. |
| `test_wan22_e2e.py` | Scheduler swap (C1), `_encode_prompt` rewrite (C2, H6), `_init_latents` dtype (H3, L8), `_denoise` timestep dtype + i2v mask (H4, M7), default config (H5, prompt). |

No changes to C++ / PJRT / `load_*` / wrappers / shard specs / `run_component`.

## Verification plan (CPU-only; TT paths unchanged)

1. `MODE="t2v"`, `NUM_STEPS=4`, all `TT_*=False`, prompt="A red apple on a white table" — fast iteration loop. Pure-noise signature should be gone; expect blurry but coherent.
2. If (1) still looks bad, bump `NUM_STEPS=50` to rule out underdenoising.
3. If (2) still looks bad, bisect: revert H3, H4, H5, H6 one at a time while holding C1 and C2 on, find the residual culprit.
4. `MODE="i2v"`, `NUM_STEPS=4` — confirms M7 mask is correct.
5. Flip `TT_TEXT_ENCODER` → `TT_VAE_ENCODER` → `TT_DIT` → `TT_VAE_DECODER` one at a time — confirms TT paths unaffected by refactor.

## Out of scope

- Refactoring into `diffusers.WanPipeline` (user explicitly wants to keep hand-rolled structure).
- `num_frames % 4 == 1` input validation (current constants already satisfy it).
- Two-transformer high/low-noise split (TI2V-5B is single-stage per `boundary_ratio: null`).
- `(1 - mask) * condition + mask * latents` per-step i2v mix (current hard re-assignment after `scheduler.step` is net-equivalent).
- Changing pytest assertions or making quality assertions (smoke test contract stays).
