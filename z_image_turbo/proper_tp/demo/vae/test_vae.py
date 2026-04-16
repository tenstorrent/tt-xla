#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
E2E test for the generated VAE decoder TTNN code.

Runs the generated TTNN VAE decoder from main.py and compares the output
against a CPU PyTorch golden reference to measure PCC.

Usage (from vae_opt1/ or vae_opt2/):
    cd /path/to/codegen_output/vae_opt1
    python test_vae.py

Expected results:
  PCC >= 0.99  (BF16 vs FP32 with GroupNorm should be close)
"""

import inspect
import os
import re
import sys
import time

import torch
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import utils
import main as generated_main

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
SCALING_FACTOR = 0.3611
SHIFT_FACTOR = 0.1159
LATENT_CHANNELS = 16
LATENT_H = 64
LATENT_W = 64

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _find_latent_arg_idx():
    """Auto-detect the runtime latent arg index from the generated _main source.

    Looks for the pattern 'var_0 = input[N]' — the first direct input access
    that is NOT via the consteval cache.
    """
    src = inspect.getsource(generated_main._main)
    m = re.search(r'var_0\s*=\s*input\[(\d+)\]', src)
    if m:
        return int(m.group(1))
    raise RuntimeError("Could not auto-detect latent arg index from generated _main source")


LATENT_ARG_IDX = _find_latent_arg_idx()
print(f"[init] Latent arg index: {LATENT_ARG_IDX}")


# ── CPU golden ─────────────────────────────────────────────────────────────────

def cpu_decode(raw_latent_nchw):
    """Decode latents on CPU using PyTorch/diffusers.

    Args:
        raw_latent_nchw: [1, 16, 64, 64] float32 (before denorm)

    Returns:
        [1, 3, 512, 512] float32
    """
    from diffusers import AutoencoderKL
    print("  Loading CPU VAE ...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae.config.force_upcast = False
    vae.eval()
    with torch.no_grad():
        z = (raw_latent_nchw / SCALING_FACTOR) + SHIFT_FACTOR
        out = vae.decoder(z)
    return out  # [1, 3, 512, 512] float32


# ── TTNN VAE class ─────────────────────────────────────────────────────────────

class TTNNVae:
    """Wraps the generated VAE TTNN code for repeated inference.

    Static inputs (model weights) are loaded once from tensor files.
    Only the latent input is replaced on each forward call.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

        # Load all static inputs from tensor files (done ONCE)
        print("  Loading static inputs from tensor files ...")
        orig_cwd = os.getcwd()
        os.chdir(_HERE)
        try:
            self._inputs = generated_main.load_inputs_for__main()
        finally:
            os.chdir(orig_cwd)
        print(f"  Loaded {len(self._inputs)} input tensors")

    def __call__(self, raw_latent_nchw):
        """
        Args:
            raw_latent_nchw: [1, 16, 64, 64] float32 (before denorm)

        Returns:
            [1, 3, 512, 512] float32 PyTorch tensor
        """
        # Denormalize the latent
        z_pt = (raw_latent_nchw.float() / SCALING_FACTOR) + SHIFT_FACTOR

        # Upload to mesh (replicated — VAE has no TP sharding)
        z_tt = ttnn.from_torch(
            z_pt.bfloat16(),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.mesh_device,
            memory_config=DRAM_RM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Replace the runtime latent slot (deallocated by _main after use)
        self._inputs[LATENT_ARG_IDX] = z_tt

        # Run forward pass
        outputs = generated_main._main(self._inputs)

        # Gather output from mesh device (take first of 4 replicas)
        out_host = ttnn.to_torch(
            ttnn.from_device(outputs[0]),
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )
        # Shape after concat: [4, 3, 512, 512] (4 replicas × batch=1)
        # Take just the first replica
        return out_host[:out_host.shape[0] // 4].float()


# ── PCC comparison ─────────────────────────────────────────────────────────────

def compare_pcc(tt_out, cpu_out, label=""):
    print(f"\n{'='*58}")
    if label:
        print(f"  {label}")
    tt_f = tt_out.float().flatten()
    cpu_f = cpu_out.float().flatten()

    if tt_f.numel() != cpu_f.numel():
        print(f"  SHAPE MISMATCH: TT={tuple(tt_out.shape)} vs CPU={tuple(cpu_out.shape)}")
        print(f"{'='*58}")
        return -1.0

    pcc = torch.corrcoef(torch.stack([tt_f, cpu_f]))[0, 1].item()
    max_diff = (tt_f - cpu_f).abs().max().item()
    mean_diff = (tt_f - cpu_f).abs().mean().item()
    print(f"  TT  output: {tuple(tt_out.shape)}  "
          f"mean={tt_out.mean():.4f}  std={tt_out.std():.4f}")
    print(f"  CPU output: {tuple(cpu_out.shape)}  "
          f"mean={cpu_out.mean():.4f}  std={cpu_out.std():.4f}")
    print(f"  PCC:         {pcc:.6f}  (>= 0.99 target)")
    print(f"  Max  |diff|: {max_diff:.6f}")
    print(f"  Mean |diff|: {mean_diff:.6f}")

    status = "PASS" if pcc >= 0.99 else "FAIL"
    print(f"  Status:      {status}")
    print(f"{'='*58}")
    return pcc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    print(f"\nVAE Decoder TTNN test  ({os.path.basename(_HERE)})")
    print("─" * 58)

    # Dummy raw latent (before denorm) — the same tensor used for both CPU and TT
    raw_latent = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=torch.float32)

    # ── CPU golden ──────────────────────────────────────────────────────────────
    print("\n[1/3] CPU golden ...")
    t0 = time.time()
    cpu_out = cpu_decode(raw_latent)
    cpu_ms = (time.time() - t0) * 1000
    print(f"  CPU: {cpu_ms:.0f} ms   output={tuple(cpu_out.shape)}")

    # ── Open TTNN device ────────────────────────────────────────────────────────
    print("\n[2/3] Opening TTNN (1,4) mesh device ...")
    mesh_device = utils.DeviceGetter.get_device((1, 4))
    print(f"  Device: {mesh_device}")

    # ── Build TTNN VAE and run inference ────────────────────────────────────────
    print("\n[3/3] Running TTNN VAE ...")
    vae_tt = TTNNVae(mesh_device)

    # Warm-up run (triggers consteval + JIT on first call)
    print("  Warm-up run (triggers consteval) ...")
    t0 = time.time()
    tt_out = vae_tt(raw_latent)
    warmup_ms = (time.time() - t0) * 1000
    print(f"  Warm-up: {warmup_ms:.0f} ms")

    # Measured run
    print("  Measured run ...")
    t0 = time.time()
    tt_out = vae_tt(raw_latent)
    tt_ms = (time.time() - t0) * 1000
    print(f"  Inference: {tt_ms:.0f} ms")

    # ── PCC check ───────────────────────────────────────────────────────────────
    pcc = compare_pcc(tt_out, cpu_out, label="TTNN vs CPU golden (PCC)")

    if pcc >= 0.99:
        print(f"\n  VAE TTNN test PASSED  PCC={pcc:.4f} >= 0.99")
    else:
        print(f"\n  VAE TTNN test FAILED  PCC={pcc:.4f} < 0.99")
        sys.exit(1)


if __name__ == "__main__":
    main()
