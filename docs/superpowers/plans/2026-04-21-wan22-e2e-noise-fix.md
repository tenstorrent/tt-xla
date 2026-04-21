# Wan 2.2 TI2V-5B E2E Test — Noise Output Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `tests/torch/models/wan2_2/test_wan22_e2e.py` so the generated video is coherent instead of TV-static noise, by aligning the hand-rolled pipeline with the canonical `diffusers.WanPipeline` + Wan-Video original repo behavior for the TI2V-5B single-stage model.

**Architecture:** Keep the test's hand-rolled pipeline structure and per-component TT hook pattern intact. Surgical in-place fixes to six places: scheduler class, prompt-embed zero-padding, latent dtype, timestep dtype, i2v timestep mask, and default constants. One new helper trio (`prompt_clean` + friends) added to `shared.py`.

**Tech Stack:** PyTorch 2.7, diffusers 0.35.x (`UniPCMultistepScheduler`, `WanTransformer3DModel`, `AutoencoderKLWan`), transformers (`UMT5EncoderModel`), bfloat16 weights, CPU (and optionally TT-XLA when any `TT_*` flag is enabled).

**Testing note:** This is a visual-quality smoke test — there is no unit assertion we can write for "video is not noise". The verification model is therefore: make each change → commit → eyeball the output video at the end. Frequent commits make git-bisect trivial if any later regression appears. Each code task includes a self-check (syntax / import sanity), with the real quality verification deferred to the end-to-end verification tasks.

**Reference:** `docs/superpowers/specs/2026-04-21-wan22-e2e-noise-fix-design.md`

---

## Task 1: Add `prompt_clean` helpers to `shared.py`

**Files:**
- Modify: `tests/torch/models/wan2_2/shared.py` (add helpers at module level)

The diffusers `WanPipeline` applies `ftfy.fix_text` → `html.unescape` → whitespace-collapse before tokenization. We port the three-line helper so the test normalizes prompts the same way the trained pipeline does. `ftfy` and `regex` are already installed in the venv.

Also adds a single named constant `LATENT_CHANNELS = 48` matching `transformer/config.json` so the denoise loop can reference it by name (fixes spec item L8).

- [ ] **Step 1: Read the current imports block of shared.py**

Read: `tests/torch/models/wan2_2/shared.py:1-22`

Expected: confirms current imports are `torch`, `torch.nn`, and two `infra.utilities` helpers; the `MODEL_ID` constant is defined around line 22.

- [ ] **Step 2: Add the new imports after the existing ones**

Edit `tests/torch/models/wan2_2/shared.py` — replace the block:

```python
import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
```

with:

```python
import html

import regex as re
import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Matches WanTransformer3DModel.config.in_channels for TI2V-5B
# (= VAE z_dim = 48). Named here so the denoise loop avoids a magic number.
LATENT_CHANNELS = 48
```

- [ ] **Step 3: Add `prompt_clean` helpers after the `RESOLUTIONS` dict**

Edit `tests/torch/models/wan2_2/shared.py` — find the block ending with the closing brace of `RESOLUTIONS` (around line 44) and the `# ---- Model loaders` header (around line 47). Insert between them:

```python
# ---------------------------------------------------------------------------
# Prompt cleaning — matches diffusers/pipeline_wan.py:78-93 and Wan repo.
# ---------------------------------------------------------------------------


def _basic_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    """Normalize prompt text the same way diffusers.WanPipeline does."""
    return _whitespace_clean(_basic_clean(text))
```

- [ ] **Step 4: Sanity-check the module parses and exports the new symbols**

Run: `python -c "from tests.torch.models.wan2_2.shared import prompt_clean, LATENT_CHANNELS; print(repr(prompt_clean('  hello   world\n')), LATENT_CHANNELS)"` (run from repo root).

Expected output: `'hello world' 48`

- [ ] **Step 5: Commit**

```bash
cd /home/ttuser/mstojkovic/tt-xla
git add tests/torch/models/wan2_2/shared.py
git commit -m "$(cat <<'EOF'
test(wan2_2): add prompt_clean helper and LATENT_CHANNELS constant

prompt_clean mirrors diffusers WanPipeline preprocessing (ftfy + html
unescape + whitespace collapse). LATENT_CHANNELS names the in_channels
value from transformer/config.json so downstream call sites avoid a
magic 48.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Swap scheduler class from `FlowMatchEulerDiscreteScheduler` to `UniPCMultistepScheduler` (🔴 C1)

**Files:**
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:231`, `:269-271`

`model_index.json` on HuggingFace declares the scheduler as `UniPCMultistepScheduler`; the config in the `scheduler/` subfolder uses `flow_shift=5.0`, `use_flow_sigmas=true`, `prediction_type="flow_prediction"` — values that `FlowMatchEulerDiscreteScheduler` silently drops. This is the primary root cause of the noise output.

- [ ] **Step 1: Read the current `_run` function**

Read: `tests/torch/models/wan2_2/test_wan22_e2e.py:230-290`

- [ ] **Step 2: Replace the scheduler import and instantiation**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — change:

```python
def _run(out_path: Path) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler

    from .shared import MODEL_ID
```

to:

```python
def _run(out_path: Path) -> None:
    from diffusers import UniPCMultistepScheduler

    from .shared import MODEL_ID
```

And change:

```python
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler"
    )
    scheduler.set_timesteps(num_inference_steps=NUM_STEPS)
```

to:

```python
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler"
    )
    scheduler.set_timesteps(num_inference_steps=NUM_STEPS)
```

(The `set_timesteps(num_inference_steps=NUM_STEPS)` call is unchanged — both schedulers accept the same keyword.)

- [ ] **Step 3: Sanity-check the instantiation**

Run: `python -c "from diffusers import UniPCMultistepScheduler; s = UniPCMultistepScheduler.from_pretrained('Wan-AI/Wan2.2-TI2V-5B-Diffusers', subfolder='scheduler'); s.set_timesteps(num_inference_steps=4); print(s.timesteps)"`

Expected: four timesteps printed in descending order (roughly `[~996, ~980, ~920, ~720]` — the flow-shifted UniPC schedule). No `ValueError` about unexpected config keys.

- [ ] **Step 4: Commit**

```bash
cd /home/ttuser/mstojkovic/tt-xla
git add tests/torch/models/wan2_2/test_wan22_e2e.py
git commit -m "$(cat <<'EOF'
test(wan2_2): swap scheduler to UniPCMultistepScheduler

model_index.json declares UniPCMultistepScheduler; loading the
scheduler/ config (flow_shift=5.0, flow_prediction, flow_sigmas)
into FlowMatchEulerDiscreteScheduler silently dropped those kwargs
and produced a wrong sigma trajectory — root cause of noise output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rewrite `_encode_prompt` — zero-pad past real token length, apply `prompt_clean` (🔴 C2 + 🟠 H6)

**Files:**
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:21-36` (imports from `.shared`)
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:92-107` (`_encode_prompt` function)

The diffusers `_get_t5_prompt_embeds` (pipeline_wan.py:158-197) and the Wan-Video repo's `T5EncoderModel.__call__` both truncate each embed to `mask.gt(0).sum(dim=1)` and re-pad with explicit zeros. UMT5's feed-forward + residuals produce non-trivial values at pad positions; DiT cross-attention is trained expecting zeros there. Without the zero-pad, pad-position garbage leaks into every latent query via cross-attention.

- [ ] **Step 1: Add `prompt_clean` to the `.shared` import block**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — replace:

```python
from .shared import (
    RESOLUTIONS,
    UMT5Wrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    WanDiTWrapper,
    load_dit,
    load_tokenizer,
    load_umt5,
    load_vae,
    run_component,
    shard_dit_specs,
    shard_umt5_specs,
    shard_vae_decoder_specs,
    shard_vae_encoder_specs,
)
```

with:

```python
from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    UMT5Wrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    WanDiTWrapper,
    load_dit,
    load_tokenizer,
    load_umt5,
    load_vae,
    prompt_clean,
    run_component,
    shard_dit_specs,
    shard_umt5_specs,
    shard_vae_decoder_specs,
    shard_vae_encoder_specs,
)
```

- [ ] **Step 2: Replace the `_encode_prompt` function body**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — replace the current function:

```python
def _encode_prompt(tokenizer, encoder_wrapper, text: str) -> torch.Tensor:
    ids = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"]
    attention_mask = ids["attention_mask"]
    return run_component(
        encoder_wrapper,
        [input_ids, attention_mask],
        on_tt=TT_TEXT_ENCODER,
        shard_spec_fn=(lambda m: shard_umt5_specs(m.encoder)),
    )
```

with:

```python
def _encode_prompt(tokenizer, encoder_wrapper, text: str) -> torch.Tensor:
    ids = tokenizer(
        prompt_clean(text),
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"]
    attention_mask = ids["attention_mask"]
    seq_lens = attention_mask.gt(0).sum(dim=1).long()

    embeds = run_component(
        encoder_wrapper,
        [input_ids, attention_mask],
        on_tt=TT_TEXT_ENCODER,
        shard_spec_fn=(lambda m: shard_umt5_specs(m.encoder)),
    )

    # Zero out positions past the real token length. UMT5 produces non-zero
    # values at pad positions via residuals/FFN; DiT cross-attention is
    # trained expecting zero there. Mirrors diffusers _get_t5_prompt_embeds
    # (pipeline_wan.py:186-190) and Wan repo T5EncoderModel.__call__.
    max_len = embeds.shape[1]
    trimmed = [e[:n] for e, n in zip(embeds, seq_lens)]
    return torch.stack(
        [
            torch.cat([e, e.new_zeros(max_len - e.size(0), e.size(1))])
            for e in trimmed
        ],
        dim=0,
    )
```

- [ ] **Step 3: Sanity-check the imports resolve and function is well-formed**

Run: `python -c "import tests.torch.models.wan2_2.test_wan22_e2e as m; print(m._encode_prompt.__doc__, m.prompt_clean('x'))"` (run from repo root with venv activated).

Expected: no `ImportError` or `NameError`; prints `None 'x'`.

- [ ] **Step 4: Commit**

```bash
cd /home/ttuser/mstojkovic/tt-xla
git add tests/torch/models/wan2_2/test_wan22_e2e.py
git commit -m "$(cat <<'EOF'
test(wan2_2): zero-pad prompt embeds past real token length

UMT5 produces non-trivial values at pad positions; DiT cross-attention
is trained expecting zeros there. Mirrors diffusers _get_t5_prompt_embeds
and Wan repo T5EncoderModel.__call__. Also applies prompt_clean
preprocessing to match diffusers normalization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Float32 latent init + named channel constant (🟠 H3 + 🟢 L8)

**Files:**
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:110-119` (`_init_latents`)
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:258-265` (i2v assertion), `:252` (call site)

Both the diffusers pipeline (`pipeline_wan.py:559-569` via `prepare_latents` → `randn_tensor(dtype=torch.float32)`) and the Wan-Video repo (`text2video.py: torch.randn(..., dtype=torch.float32)`) sample the initial latent in `float32`. bf16's 8-bit mantissa visibly quantizes a Gaussian. We'll sample in float32 and keep `latents` float32 through the denoise loop; dtype casts at the DiT boundary happen in Task 5.

- [ ] **Step 1: Replace `_init_latents`**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — replace:

```python
def _init_latents(shapes: dict, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        1,
        48,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
        generator=generator,
    )
```

with:

```python
def _init_latents(shapes: dict, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        1,
        LATENT_CHANNELS,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.float32,
        generator=generator,
    )
```

- [ ] **Step 2: Update the i2v shape assertion to use the named constant**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — replace:

```python
        assert image_latent.shape == (
            1,
            48,
            1,
            shapes["latent_h"],
            shapes["latent_w"],
        ), image_latent.shape
```

with:

```python
        assert image_latent.shape == (
            1,
            LATENT_CHANNELS,
            1,
            shapes["latent_h"],
            shapes["latent_w"],
        ), image_latent.shape
```

- [ ] **Step 3: Sanity-check latent dtype**

Run: `python -c "
import torch
from tests.torch.models.wan2_2.test_wan22_e2e import _init_latents
from tests.torch.models.wan2_2.shared import RESOLUTIONS
g = torch.Generator().manual_seed(0)
x = _init_latents(RESOLUTIONS['480p'], g)
print(x.shape, x.dtype)
"`

Expected output: `torch.Size([1, 48, 21, 30, 52]) torch.float32`

- [ ] **Step 4: Commit**

```bash
cd /home/ttuser/mstojkovic/tt-xla
git add tests/torch/models/wan2_2/test_wan22_e2e.py
git commit -m "$(cat <<'EOF'
test(wan2_2): sample initial latents in float32

Matches diffusers randn_tensor(dtype=torch.float32) and Wan repo
torch.randn(..., dtype=torch.float32). bf16 mantissa quantizes the
Gaussian and drifts the denoising trajectory. Also uses the new
LATENT_CHANNELS constant instead of a hardcoded 48.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fix `_denoise` — float32 timestep, bf16 boundary at DiT, i2v first-frame mask (🟠 H4 + 🟡 M7 + H3 completion)

**Files:**
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:140-183` (`_denoise` function)

Three changes at once in the same function:
- Timestep tensor becomes `float32` (bf16 quantizes values near 1000 to ~8-unit increments, corrupting sinusoidal embedding).
- DiT forward receives `latents.to(bfloat16)` (DiT weights are bf16). Velocity cast back to `float32` before `scheduler.step` so the float32 `latents` trajectory is preserved.
- When `image_latent is not None`, the timestep tensor's first latent frame gets 0 instead of `t` — mirrors diffusers `first_frame_mask[0][0][:, ::2, ::2] * t` in `pipeline_wan_i2v.py:757-764`.

- [ ] **Step 1: Replace the `_denoise` function body**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — replace the current function:

```python
def _denoise(
    dit_wrapper: WanDiTWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds,
    scheduler,
    shapes: dict,
    image_latent: torch.Tensor | None,
) -> torch.Tensor:
    t_dim, h_dim, w_dim = (
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
    )
    num_patches = t_dim * (h_dim // 2) * (w_dim // 2)

    for t in scheduler.timesteps:
        timestep = torch.full((1, num_patches), float(t), dtype=torch.bfloat16)

        v_pos = run_component(
            dit_wrapper,
            [latents, timestep, prompt_embeds],
            on_tt=TT_DIT,
            shard_spec_fn=(lambda m: shard_dit_specs(m.dit)),
        )
        if negative_embeds is not None:
            v_neg = run_component(
                dit_wrapper,
                [latents, timestep, negative_embeds],
                on_tt=TT_DIT,
                shard_spec_fn=(lambda m: shard_dit_specs(m.dit)),
            )
            velocity = v_neg + GUIDANCE_SCALE * (v_pos - v_neg)
        else:
            velocity = v_pos

        latents = scheduler.step(velocity, t, latents).prev_sample

        if image_latent is not None:
            # Wan 2.2 TI2V convention: the first latent frame is fixed to
            # the conditioning image and should not be denoised.
            latents[:, :, 0:1, :, :] = image_latent

    return latents
```

with:

```python
def _denoise(
    dit_wrapper: WanDiTWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds,
    scheduler,
    shapes: dict,
    image_latent: torch.Tensor | None,
) -> torch.Tensor:
    t_dim, h_dim, w_dim = (
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
    )
    num_patches = t_dim * (h_dim // 2) * (w_dim // 2)

    # For i2v (Wan 2.2 TI2V expand_timesteps=True): first latent frame is
    # the conditioning image and should receive timestep 0, not t. Mirrors
    # diffusers pipeline_wan_i2v.py:757-764 — first_frame_mask[0][0][:,::2,::2]*t.
    if image_latent is not None:
        ts_mask = torch.ones(t_dim, h_dim // 2, w_dim // 2, dtype=torch.float32)
        ts_mask[0] = 0
        ts_mask = ts_mask.flatten()  # shape (num_patches,)
    else:
        ts_mask = None

    latents_fp32 = latents  # keep float32 authoritative copy

    for t in scheduler.timesteps:
        if ts_mask is not None:
            timestep = (ts_mask * float(t)).unsqueeze(0)
        else:
            timestep = torch.full(
                (1, num_patches), float(t), dtype=torch.float32
            )

        # DiT has bf16 weights — cast latents at the boundary; keep the
        # float32 copy for scheduler.step precision.
        latents_bf16 = latents_fp32.to(torch.bfloat16)

        v_pos = run_component(
            dit_wrapper,
            [latents_bf16, timestep, prompt_embeds],
            on_tt=TT_DIT,
            shard_spec_fn=(lambda m: shard_dit_specs(m.dit)),
        )
        if negative_embeds is not None:
            v_neg = run_component(
                dit_wrapper,
                [latents_bf16, timestep, negative_embeds],
                on_tt=TT_DIT,
                shard_spec_fn=(lambda m: shard_dit_specs(m.dit)),
            )
            velocity = v_neg + GUIDANCE_SCALE * (v_pos - v_neg)
        else:
            velocity = v_pos

        # Cast velocity back to float32 for numerically-careful scheduler.step.
        latents_fp32 = scheduler.step(
            velocity.to(torch.float32), t, latents_fp32
        ).prev_sample

        if image_latent is not None:
            # Keep the conditioning frame fixed across iterations.
            latents_fp32[:, :, 0:1, :, :] = image_latent.to(torch.float32)

    return latents_fp32
```

- [ ] **Step 2: Verify the function parses and timestep/mask shapes are right**

Run: `python -c "
import torch
from tests.torch.models.wan2_2.test_wan22_e2e import _denoise
import inspect
src = inspect.getsource(_denoise)
assert 'torch.float32' in src and 'ts_mask' in src
print('OK, _denoise updated')
"`

Expected: `OK, _denoise updated`

- [ ] **Step 3: Commit**

```bash
cd /home/ttuser/mstojkovic/tt-xla
git add tests/torch/models/wan2_2/test_wan22_e2e.py
git commit -m "$(cat <<'EOF'
test(wan2_2): float32 timestep, bf16 boundary at DiT, i2v first-frame mask

- Timestep tensor: bf16 -> float32. bf16 quantizes t~1000 to 8-unit
  increments, corrupting the sinusoidal time embedding.
- latents stay float32 through the loop; cast to bf16 only at DiT
  forward boundary. Velocity cast back to float32 for scheduler.step.
  Matches diffusers pattern exactly.
- i2v: first latent-frame timestep becomes 0 (not t) via ts_mask,
  mirroring diffusers first_frame_mask in pipeline_wan_i2v.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update default constants — CFG on, easier prompt (🟠 H5 + prompt)

**Files:**
- Modify: `tests/torch/models/wan2_2/test_wan22_e2e.py:42-58` (config block)

Default guidance scale becomes 5.0 (matches both reference implementations) which activates the negative-prompt path. Default prompt becomes a single-subject, single-surface scene that is impossible to misread by eye.

- [ ] **Step 1: Replace the configuration block**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — replace:

```python
MODE = "t2v"  # "t2v" or "i2v"
RESOLUTION = "480p"  # "480p" or "720p"
NUM_STEPS = 4  # denoising steps
GUIDANCE_SCALE = 1.0  # 1.0 disables classifier-free guidance

TT_TEXT_ENCODER = False
TT_VAE_ENCODER = False  # only used when MODE == "i2v"
TT_DIT = False
TT_VAE_DECODER = False

PROMPT = (
    "A gentle ocean wave rolling across a quiet sandy beach at sunrise, "
    "cinematic, high detail"
)
NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text"
SEED = 42
FPS = 16
```

with:

```python
MODE = "t2v"  # "t2v" or "i2v"
RESOLUTION = "480p"  # "480p" or "720p"
NUM_STEPS = 4  # denoising steps (bump to 50 for quality check)
GUIDANCE_SCALE = 5.0  # matches diffusers / Wan repo default; CFG on

TT_TEXT_ENCODER = False
TT_VAE_ENCODER = False  # only used when MODE == "i2v"
TT_DIT = False
TT_VAE_DECODER = False

# Simple single-subject prompt for smoke-test readability. Failure is
# immediately obvious by eye; success is unambiguous. Fallback options
# (if quality seems poor) documented in the design spec:
#   "A cat sitting on grass"
#   "A dog running on a beach"
#   Wan 2.2 README canonical (complex):
#   "Two anthropomorphic cats in comfy boxing gear and bright gloves fight
#    intensely on a spotlighted stage."
PROMPT = "A red apple on a white table"
NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text"
SEED = 42
FPS = 16
```

- [ ] **Step 2: Confirm the file still parses**

Run: `python -c "import tests.torch.models.wan2_2.test_wan22_e2e as m; print(m.PROMPT, m.GUIDANCE_SCALE)"`

Expected output: `A red apple on a white table 5.0`

- [ ] **Step 3: Commit**

```bash
cd /home/ttuser/mstojkovic/tt-xla
git add tests/torch/models/wan2_2/test_wan22_e2e.py
git commit -m "$(cat <<'EOF'
test(wan2_2): CFG on by default, simpler default prompt

- GUIDANCE_SCALE 1.0 -> 5.0 matches both reference implementations.
  1.0 bypassed CFG entirely; default value in diffusers and Wan repo
  is 5.0.
- Default prompt becomes single-subject ('A red apple on a white table')
  for unambiguous smoke-test readability. Fallback ladder documented
  inline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: End-to-end verification — fast CPU run (4 steps, t2v)

**Files:**
- Run (read-only): `tests/torch/models/wan2_2/test_wan22_e2e.py`
- Generates: `tests/torch/models/wan2_2/generated/wan22_t2v_480p_steps4_cpu.mp4`

Fast iteration loop. 4 steps is not enough for a sharp image, but with the fixes applied it should produce a coherent low-frequency signal rather than TV static. Any regression from pure noise to *some* structure is the primary success criterion for this plan.

- [ ] **Step 1: Delete stale output if present (so we don't confuse the check)**

Run: `rm -f /home/ttuser/mstojkovic/tt-xla/tests/torch/models/wan2_2/generated/wan22_t2v_480p_steps4_cpu.mp4`

Expected: no error.

- [ ] **Step 2: Activate the venv**

Run: `source /home/ttuser/mstojkovic/tt-xla/venv/activate`

Expected: no error; `echo $TTXLA_ENV_ACTIVATED` prints `1` (or similar truthy value).

- [ ] **Step 3: Run the test**

Run: `cd /home/ttuser/mstojkovic/tt-xla && pytest -svv tests/torch/models/wan2_2/test_wan22_e2e.py 2>&1 | tail -30`

Expected: one test passes. Output file exists and is non-empty.

- [ ] **Step 4: Inspect the video manually**

Open `tests/torch/models/wan2_2/generated/wan22_t2v_480p_steps4_cpu.mp4` and eyeball.

Expected at NUM_STEPS=4:
- **NOT** TV static (root cause fixed).
- Blurry but coherent color blobs; low-frequency structure; maybe a reddish region consistent with the prompt subject.
- If output is still pure noise: abort, report which commit(s) are on HEAD, do not continue to Task 8. Use `git bisect` across Task 1-6 commits to identify which change was regressed.

- [ ] **Step 5: If output is coherent, proceed. No commit needed (no file changes).**

---

## Task 8: Quality-level verification — 50-step CPU t2v

**Files:**
- Temporarily edit: `tests/torch/models/wan2_2/test_wan22_e2e.py:44` (`NUM_STEPS = 50`)
- Revert after: same line back to `NUM_STEPS = 4`

Confirms the fixes give recognizable content at the reference step count. This is the high-confidence "did we actually fix it" check.

- [ ] **Step 1: Change `NUM_STEPS` to 50 locally (uncommitted)**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — change:

```python
NUM_STEPS = 4  # denoising steps (bump to 50 for quality check)
```

to:

```python
NUM_STEPS = 50  # denoising steps (bump to 50 for quality check)
```

- [ ] **Step 2: Delete any stale 50-step output**

Run: `rm -f /home/ttuser/mstojkovic/tt-xla/tests/torch/models/wan2_2/generated/wan22_t2v_480p_steps50_cpu.mp4`

- [ ] **Step 3: Run the test (slow — ~several minutes on CPU)**

Run: `cd /home/ttuser/mstojkovic/tt-xla && pytest -svv tests/torch/models/wan2_2/test_wan22_e2e.py 2>&1 | tail -30`

Expected: test passes after the denoise loop completes.

- [ ] **Step 4: Inspect `generated/wan22_t2v_480p_steps50_cpu.mp4`**

Expected at NUM_STEPS=50:
- A recognizable still / mildly-animated red apple on a light background. Details may be crude given the 5B parameter count at 480p, but the subject should be identifiable.
- If still pure noise after 50 steps: stop, record which fix is the residual problem, re-open the spec. Likely candidates in order: prompt-embed zero-pad (C2) not actually taking effect, dtype boundary in `_denoise` wrong (Task 5), or UMT5 embed_tokens weight-tying workaround broken (shared.py:69 — verify `enc.encoder.embed_tokens.weight is enc.shared.weight` prints `True` after `load_umt5()`).

- [ ] **Step 5: Revert `NUM_STEPS` back to 4 (do not commit 50)**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py` — change:

```python
NUM_STEPS = 50  # denoising steps (bump to 50 for quality check)
```

back to:

```python
NUM_STEPS = 4  # denoising steps (bump to 50 for quality check)
```

- [ ] **Step 6: Confirm no uncommitted changes remain**

Run: `cd /home/ttuser/mstojkovic/tt-xla && git status tests/torch/models/wan2_2/test_wan22_e2e.py`

Expected: file is clean, no modifications.

---

## Task 9: i2v verification (optional but recommended — exercises M7)

**Files:**
- Temporarily edit: `tests/torch/models/wan2_2/test_wan22_e2e.py:42` (`MODE = "i2v"`)
- Revert after: same line back to `"t2v"`

Confirms the i2v first-frame timestep mask (Task 5's M7 fix) works. Uses the existing deterministic-random first frame.

- [ ] **Step 1: Flip `MODE` to `"i2v"` (uncommitted)**

Edit `tests/torch/models/wan2_2/test_wan22_e2e.py`:

```python
MODE = "i2v"  # "t2v" or "i2v"
```

- [ ] **Step 2: Delete stale i2v output**

Run: `rm -f /home/ttuser/mstojkovic/tt-xla/tests/torch/models/wan2_2/generated/wan22_i2v_480p_steps4_cpu.mp4`

- [ ] **Step 3: Run the test**

Run: `cd /home/ttuser/mstojkovic/tt-xla && pytest -svv tests/torch/models/wan2_2/test_wan22_e2e.py 2>&1 | tail -30`

Expected: passes. The i2v code path (VAE encoder + first-frame mask) exercised.

- [ ] **Step 4: Inspect `generated/wan22_i2v_480p_steps4_cpu.mp4`**

Expected:
- First frame roughly matches the deterministic-random conditioning image (noisy, high-contrast colors — because the input was `torch.randn().clamp(-1,1)`, not a real photo).
- Subsequent frames show gradual denoising — if M7 is wrong, later frames will stay noisy because the conditioning frame's timestep would be `t` instead of `0`, letting the DiT try to "denoise" it too.

- [ ] **Step 5: Revert `MODE` back to `"t2v"`**

Edit the same line back to `"t2v"`.

- [ ] **Step 6: Confirm clean working tree**

Run: `cd /home/ttuser/mstojkovic/tt-xla && git status tests/torch/models/wan2_2/test_wan22_e2e.py`

Expected: no modifications.

---

## Task 10: TT flag sweep — confirm no regression on the accelerator paths

**Files:**
- Temporarily edit `TT_*` flags in `tests/torch/models/wan2_2/test_wan22_e2e.py:47-50` one at a time.

Each TT flag routes one component to the XLA/TT device. None of our changes touched `run_component` or the shard-spec functions — but we did change latent dtype (float32 on the CPU side now) and timestep dtype (float32). These flow into the TT side via `inputs_on_device = [t.to(device) for t in inputs]` in `run_component` (shared.py:359). Confirming each flag still works end-to-end is cheap insurance.

**This task is optional** if the user has no TT hardware attached or wants to merge the CPU-only fix now and run the TT sweep later.

- [ ] **Step 1: `TT_TEXT_ENCODER = True`, everything else false**

Edit the flag; delete stale output; run `pytest -svv tests/torch/models/wan2_2/test_wan22_e2e.py`; eyeball `wan22_t2v_480p_steps4_tt-textenc.mp4`. Revert flag.

Expected: same coherent-blob output as the CPU-only run. If the TT path produces garbage and the CPU path doesn't, the bug is TT-side and orthogonal to this plan.

- [ ] **Step 2: `TT_DIT = True`, others false**

Same procedure. Output file: `wan22_t2v_480p_steps4_tt-dit.mp4`.

- [ ] **Step 3: `TT_VAE_DECODER = True`, others false**

Same. File: `wan22_t2v_480p_steps4_tt-vaedec.mp4`.

- [ ] **Step 4: (i2v only) `TT_VAE_ENCODER = True`, `MODE = "i2v"`, others false**

Same. File: `wan22_i2v_480p_steps4_tt-vaeenc.mp4`.

- [ ] **Step 5: Confirm clean working tree after all reverts**

Run: `cd /home/ttuser/mstojkovic/tt-xla && git status`

Expected: no modifications to `test_wan22_e2e.py`.

---

## Self-Review Results

- **Spec coverage:** All seven spec fix items (C1, C2, H3, H4, H5, H6, M7, L8) map to tasks: C1→T2, C2→T3, H3→T4+T5, H4→T5, H5→T6, H6→T3, M7→T5, L8→T1+T4. Prompt update → T6. Verification plan → T7, T8, T9, T10. ✓
- **Placeholder scan:** No TBD/TODO/"fill in later". Every code step has complete code. Every command has expected output. ✓
- **Type consistency:** `LATENT_CHANNELS` named identically in shared.py (T1), imported in test file (T3), used in `_init_latents` (T4) and the i2v assertion (T4). `prompt_clean` exported in T1, imported in T3, called in T3. `_denoise` signature unchanged from original; internals only. ✓

## Notes for the executor

- **Working tree must be clean before Task 1.** Current repo has `docs/superpowers/` plus unrelated tmp files — those are fine to leave alone, but the task commits should only touch `tests/torch/models/wan2_2/shared.py` and `tests/torch/models/wan2_2/test_wan22_e2e.py`.
- **Auto mode is enabled.** Execute tasks autonomously; eyeball the generated MP4 at verification tasks. If a verification fails, stop and surface the failure rather than patching and moving on.
- **Pre-commit hooks will run** (`black`, `clang-format`, copyright notice check). `black` will normalize formatting on the Python files we touch — if it reformats something, `git add -u` and re-commit is fine.
- **No network in commits:** the commit messages don't need to fetch anything. The tests themselves hit HuggingFace for model weights on first run; the cache is persistent so re-runs are offline.
