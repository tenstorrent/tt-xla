# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo pipeline: text encoder + transformer on TTNN, VAE decoder on CPU.

Fixed-resolution: 512×512 px (64×64 latent, 32 caption tokens — TTNN compile shapes).

Usage:
    python z_image.py --prompt "sunny mountain range, with peaks peaking through misty clouds"
    python z_image.py --prompt "..." --steps 8 --seed 42 --output out.png
"""

import argparse
import importlib.util
import os
import sys
import time

import torch
import ttnn
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_TE_DIR = os.path.join(_HERE, "text_encoder")
_TR_DIR = os.path.join(_HERE, "transformer")

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# Shapes baked into the TTNN-compiled models — must not change.
IMG_LATENT_H = 64       # 512px / 8 (VAE scale)
IMG_LATENT_W = 64       # 512px / 8
LATENT_CHANNELS = 16
CAP_TOKENS = 32         # caption tokens expected by the TTNN transformer

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


# ── Module loading ─────────────────────────────────────────────────────────────

def _load_module(name, filepath):
    """Load a Python module from an explicit file path and register in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load utils (DeviceGetter singleton) — identical in both subdirs; pick transformer's.
_utils = _load_module("utils", os.path.join(_TR_DIR, "utils.py"))

# Pre-register transformer's consteval before importing transformer model_ttnn.
# transformer/model_ttnn.py does a bare `import consteval` at module level.
_load_module("consteval", os.path.join(_TR_DIR, "consteval.py"))

# Load PyTorch model helpers (needed to build the TTNN transformer).
_tr_model_pt = _load_module("tr_model_pt", os.path.join(_TR_DIR, "model_pt.py"))

# Load the two TTNN model classes.
_te_model_ttnn = _load_module("te_model_ttnn", os.path.join(_TE_DIR, "model_ttnn.py"))
_tr_model_ttnn = _load_module("tr_model_ttnn", os.path.join(_TR_DIR, "model_ttnn.py"))

TextEncoderTTNN = _te_model_ttnn.TextEncoderTTNN
ZImageTransformerTTNN = _tr_model_ttnn.ZImageTransformerTTNN


# ── Tensor helpers ─────────────────────────────────────────────────────────────

def _to_device_bf16(pt, mesh_device):
    """Upload a bfloat16 PyTorch tensor to the mesh device (replicated)."""
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_int32(input_ids_int64, mesh_device):
    """Upload int64 token IDs as an INT32 TTNN tensor (replicated, [1, seq])."""
    return ttnn.from_torch(
        input_ids_int64.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    """Pull a TTNN mesh tensor to CPU, de-duplicating the 4 replicated shards."""
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # ConcatMeshToTensor stacks 4 copies along dim=0 — take the first slice.
    n = host.shape[0] // 4
    return host[:n].float()


# ── Scheduler helpers ──────────────────────────────────────────────────────────

def _compute_mu(latent_h=IMG_LATENT_H, latent_w=IMG_LATENT_W,
                base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
    """Dynamic time-shift mu used by FlowMatchEulerDiscreteScheduler (matches ZImagePipeline)."""
    image_seq_len = (latent_h // 2) * (latent_w // 2)  # 32*32 = 1024 for 512px
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return image_seq_len * m + b


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _pcc(a, b):
    """Pearson correlation coefficient between two tensors."""
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_pipeline(prompt, num_steps=8, seed=42, output_path="output.png", debug=False):
    """
    Generate a 512×512 image from a text prompt using:
      - TTNN text encoder  (Qwen3, 4-way TP on (1,4) mesh)
      - TTNN transformer   (ZImageTransformer2DModel, 4-way TP)
      - CPU VAE decoder    (diffusers AutoencoderKL)

    Args:
        prompt:      Text description of the desired image.
        num_steps:   Denoising steps (default 8, per model docs).
        seed:        Random seed for reproducible noise.
        output_path: Where to save the output PNG.
        debug:       If True, compare TTNN transformer output vs CPU at each step.
    """
    torch.manual_seed(seed)
    print(f"\nPrompt : {prompt!r}")
    print(f"Steps  : {num_steps}  |  seed: {seed}  |  output: {output_path}"
          + ("  |  DEBUG ON" if debug else ""))

    # ── CPU components ────────────────────────────────────────────────────────
    print("\n[1/5] Loading CPU components ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    vae_processor = VaeImageProcessor(vae_scale_factor=16)  # vae_scale(8) × 2 patch

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scheduler.sigma_min = 0.0

    # ── TTNN mesh device ──────────────────────────────────────────────────────
    print("\n[2/5] Opening TTNN (1,4) mesh device ...")
    mesh_device = _utils.DeviceGetter.get_device((1, 4))

    # ── TTNN text encoder ─────────────────────────────────────────────────────
    print("\n[3/5] Building TTNN text encoder ...")
    te_model = TextEncoderTTNN(mesh_device)  # auto-loads Qwen3 weights from HF

    # ── TTNN transformer ──────────────────────────────────────────────────────
    print("\n[4/5] Building TTNN transformer ...")
    print("  Loading PyTorch transformer (for weight extraction) ...")
    transformer_pt = _tr_model_pt.load_model()   # applies patch_rope_for_tt internally
    _tr_model_pt.pad_heads(transformer_pt)        # 30 → 32 heads for 4-way TP

    tr_model = ZImageTransformerTTNN(mesh_device, transformer_pt)
    ZImageTransformerTTNN.dump_dir = None  # off by default; enable below for 1st step
    if not debug:
        del transformer_pt  # free CPU RAM after weight upload (skip in debug mode)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n[5/5] Running inference ...")
    t_total = time.time()

    # ·· Text encoding ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ··
    print("  Text encoding ...")
    t0 = time.time()

    # Apply chat template (matches ZImagePipeline._encode_prompt exactly).
    # Pad to CAP_TOKENS so the text encoder produces 32 non-zero embeddings.
    # Zero-padding would cause TTNN cap_embed (RMSNorm) overflow to inf.
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    input_ids = tokenizer(
        formatted,
        padding="max_length",
        truncation=True,
        max_length=CAP_TOKENS,
        return_tensors="pt",
    )["input_ids"]  # [1, CAP_TOKENS]
    seq_len = int((input_ids != tokenizer.pad_token_id).sum())
    print(f"    {seq_len} tokens with chat template (padded to {CAP_TOKENS})")

    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_cap_out = te_model(tt_ids)                     # [CAP_TOKENS, 2560] TTNN
    cap_cpu = _tt_to_torch(tt_cap_out, mesh_device)   # [CAP_TOKENS, 2560] float32
    cap_padded = cap_cpu[:CAP_TOKENS].bfloat16()      # [CAP_TOKENS, 2560] — already 32 rows

    # Upload once; reused across all denoising steps.
    tt_cap = _to_device_bf16(cap_padded.unsqueeze(0), mesh_device)  # [1, 32, 2560]

    # Check cap_feats for NaN/inf
    cap_nan = int(torch.isnan(cap_padded).sum())
    cap_inf = int(torch.isinf(cap_padded).sum())
    print(f"    cap_feats: mean={float(cap_padded.float().mean()):.3f}"
          f" std={float(cap_padded.float().std()):.3f}"
          f" nan={cap_nan} inf={cap_inf}")
    print(f"    {(time.time()-t0)*1000:.0f} ms")

    # Save cap_feats and latents for debug_nan.py
    os.makedirs("/tmp/zt_debug", exist_ok=True)
    torch.save(cap_padded, "/tmp/zt_debug/pipeline_cap_feats.pt")
    print(f"    cap_feats saved to /tmp/zt_debug/pipeline_cap_feats.pt")

    # ·· Initial noise ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ··
    latents = torch.randn(
        1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W, dtype=torch.float32
    )

    # ·· Scheduler timesteps ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ··
    mu = _compute_mu()
    try:
        scheduler.set_timesteps(num_steps, mu=mu)
    except TypeError:
        scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps  # [num_steps] values in [0, 1000]

    # ·· Denoising loop ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ··
    print(f"  Denoising ({num_steps} steps) ...")
    for i, t in enumerate(timesteps):
        t_step = time.time()

        # Normalized timestep: t ∈ [0, 1000] → (1000 − t) / 1000 ∈ [0, 1].
        t_norm = (1000.0 - float(t)) / 1000.0
        t_bf16 = torch.tensor([t_norm], dtype=torch.bfloat16)
        tt_timestep = _to_device_bf16(t_bf16, mesh_device)

        # Reshape latent: [1, 16, 64, 64] → [16, 1, 64, 64] (transformer format).
        lat_bf16 = latents.squeeze(0).unsqueeze(1).bfloat16()
        tt_lat = _to_device_bf16(lat_bf16, mesh_device)

        # Run TTNN transformer.
        tt_out_list = tr_model([tt_lat], tt_timestep, tt_cap)
        tt_out = tt_out_list[0]  # [16, 1, 64, 64] TTNN

        # Pull result to CPU.
        out_tt = _tt_to_torch(tt_out, mesh_device)         # [16, 1, 64, 64] float32
        out_4d = out_tt.squeeze(1).unsqueeze(0).bfloat16() # [1, 16, 64, 64]

        # ── Debug: compare TTNN vs CPU on the SAME latent input ─────────────
        if debug:
            with torch.no_grad():
                # Use the same latent that TTNN just processed (not a separate CPU track).
                lat_cpu_ref = lat_bf16  # [16, 1, 64, 64] bfloat16
                cpu_out_list = _tr_model_pt.forward(
                    transformer_pt, [lat_cpu_ref], t_bf16, cap_padded
                )
            out_cpu_ref = cpu_out_list[0].float()  # [16, 1, 64, 64]
            p = _pcc(out_tt, out_cpu_ref)
            print(f"    step {i+1:2d}/{num_steps}: t={float(t):6.0f}  t_norm={t_norm:.4f}"
                  f"  | TTNN mean={float(out_tt.mean()):7.3f} std={float(out_tt.std()):6.3f}"
                  f"  | CPU  mean={float(out_cpu_ref.mean()):7.3f} std={float(out_cpu_ref.std()):6.3f}"
                  f"  | PCC={p:.4f}")
        # ───────────────────────────────────────────────────────────────────

        # Flow-matching sign convention: model output is negated before scheduler step.
        noise_pred = -out_4d.float()

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        step_ms = (time.time() - t_step) * 1000
        if not debug:
            print(f"    step {i+1}/{num_steps}: t={float(t):.0f}  {step_ms:.0f} ms"
                  f"  | out: mean={float(out_tt.mean()):.3f} std={float(out_tt.std()):.3f}"
                  f"  | latents: mean={float(latents.mean()):.3f} std={float(latents.std()):.3f}")
        else:
            print(f"      latents: mean={float(latents.mean()):.3f} std={float(latents.std()):.3f}"
                  f"  ({step_ms:.0f} ms)")

    # ·· VAE decode (CPU) ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ·· ··
    print("  VAE decoding (CPU) ...")
    t0 = time.time()
    with torch.no_grad():
        latents_vae = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        print(f"    VAE input: mean={float(latents_vae.mean()):.3f} std={float(latents_vae.std()):.3f}"
              f" min={float(latents_vae.min()):.3f} max={float(latents_vae.max()):.3f}")
        image_tensor = vae.decode(latents_vae, return_dict=False)[0]
        print(f"    VAE output: mean={float(image_tensor.mean()):.3f} std={float(image_tensor.std()):.3f}"
              f" min={float(image_tensor.min()):.3f} max={float(image_tensor.max()):.3f}")
    image = vae_processor.postprocess(image_tensor, output_type="pil")[0]
    print(f"    {(time.time()-t0)*1000:.0f} ms")

    print(f"\n  Total inference: {(time.time()-t_total)*1000:.0f} ms")

    image.save(output_path)
    print(f"\nSaved → {output_path}")
    return image


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo — text-to-image via TTNN (TP=4) + CPU VAE"
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Text description of the image to generate",
    )
    parser.add_argument(
        "--steps", type=int, default=8,
        help="Number of denoising steps (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", default="output.png",
        help="Output file path (default: output.png)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Compare TTNN vs CPU transformer output at each denoising step (PCC + stats)",
    )
    args = parser.parse_args()

    run_pipeline(
        prompt=args.prompt,
        num_steps=args.steps,
        seed=args.seed,
        output_path=args.output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
