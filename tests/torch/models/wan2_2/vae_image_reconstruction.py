# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B VAE encode -> decode round-trip on CPU and TT hardware.

Loads an arbitrary image, resizes it to a Wan 2.2 supported resolution,
runs encoder + decoder on CPU (golden) and on TT, prints per-stage
execution times, and saves four images to
``tests/torch/models/wan2_2/generated/``:

  - ``<stem>_original.png``      resized input
  - ``<stem>_cpu.png``           CPU reconstruction
  - ``<stem>_tt.png``            TT reconstruction
  - ``<stem>_merged.png``        side-by-side: original | CPU | TT

A perfect VAE returns a pixel-identical image; in practice there is some
loss, and that loss is what this script visualizes — i.e. the
reconstruction ceiling of the VAE on CPU vs. TT.

Run as a module so the relative imports resolve:

    python -m tests.torch.models.wan2_2.vae_image_reconstruction \\
        --image tests/torch/models/wan2_2/reze.jpg
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Logging env vars — must be set before torch_xla / tt_torch import.
# ---------------------------------------------------------------------------

# os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
# os.environ["XLA_HLO_DEBUG"] = "1"
# os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"
# os.environ["TT_RUNTIME_MEMORY_LOG_LEVEL"] = "operation"

import torch
from PIL import Image

from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import enable_spmd

from .monkey_patch import (
    _patch_tt_torch_getitem_clamp,
    _patch_wan_resample_rep_sentinel,
    _patch_wan_resample_avoid_4d_fold,
)
from .shared import (
    RESOLUTIONS,
    VAE_SCALE_FACTOR,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    compute_pcc,
    load_first_frame_image,
    load_vae,
    shard_vae_decoder_specs,
    shard_vae_encoder_specs,
    wan22_mesh,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = Path(__file__).parent / "reze.jpg"
OUT_DIR = Path(__file__).parent / "generated"

# Decoder needs the DRAM space saving optimization to fit at 480p; mirrors
# the fix in test_vae_decoder.py (issue #13). The encoder is small enough
# to compile with default options.
_ENCODER_COMPILE_OPTS = {"optimization_level": "1"}
_DECODER_COMPILE_OPTS = {
    "optimization_level": "1",
    "experimental-enable-dram-space-saving-optimization": "true",
}

# ---------------------------------------------------------------------------
# Monkey patches — applied at import time, same as test_vae_decoder.py.
# ---------------------------------------------------------------------------

_patch_tt_torch_getitem_clamp()
_patch_wan_resample_rep_sentinel()
_patch_wan_resample_avoid_4d_fold()

def _log(msg: str) -> None:
    print(f"[vae-recon] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Image <-> tensor
# ---------------------------------------------------------------------------


def _save_first_frame(pixels: torch.Tensor, out_path: Path) -> None:
    """Save frame 0 of a (1, 3, T, H, W) VAE-pixel tensor as a PNG.

    Uses ``diffusers.VideoProcessor`` for the [-1, 1] -> uint8 conversion
    so the saved image matches the path the e2e pipeline takes for video
    frames (see ``_postprocess_and_save`` in test_wan22_e2e.py).
    """
    from diffusers.video_processor import VideoProcessor

    processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR)
    # postprocess_video returns list[batch] of list[frame] of PIL images.
    images = processor.postprocess_video(pixels.float(), output_type="pil")
    images[0][0].save(out_path)


def _log_pcc(label: str, tt_tensor: torch.Tensor, cpu_tensor: torch.Tensor) -> float:
    """Compute Pearson correlation between TT and CPU outputs and log it."""
    pcc = compute_pcc(tt_tensor, cpu_tensor)
    _log(f"  PCC {label}: {pcc:.6f}")
    return pcc


def _save_merged(
    original: torch.Tensor,
    cpu_pixels: torch.Tensor,
    tt_pixels: torch.Tensor,
    out_path: Path,
) -> None:
    """Save a single PNG laying out original | CPU | TT side-by-side."""
    from diffusers.video_processor import VideoProcessor

    proc = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR)
    panels = [
        proc.postprocess_video(t.float(), output_type="pil")[0][0]
        for t in (original, cpu_pixels, tt_pixels)
    ]
    w, h = panels[0].size
    merged = Image.new("RGB", (w * 3, h))
    for i, panel in enumerate(panels):
        merged.paste(panel.resize((w, h)), (w * i, 0))
    merged.save(out_path)


# ---------------------------------------------------------------------------
# TT runner — like ``shared.run_component`` but lets us pass the DRAM-saving
# compile flag the decoder needs.
# ---------------------------------------------------------------------------


def _run_on_tt(
    wrapper: torch.nn.Module,
    inputs: list,
    *,
    mesh: Optional[Mesh],
    shard_module: Optional[torch.nn.Module],
    shard_fn: Optional[Callable[[torch.nn.Module], dict]],
    compile_options: dict,
) -> torch.Tensor:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    use_sharding = (
        shard_fn is not None and mesh is not None and len(mesh.device_ids) > 1
    )
    if use_sharding:
        enable_spmd()

    xr.set_device_type("TT")
    device = xm.xla_device()
    torch_xla.set_custom_compile_options(compile_options)

    wrapper_on_device = wrapper.to(device)
    inputs_on_device = [t.to(device) for t in inputs]

    if use_sharding:
        assert shard_module is not None, "shard_fn requires shard_module"
        for tensor, spec in shard_fn(shard_module).items():
            xs.mark_sharding(tensor, mesh, spec)

    compiled = torch.compile(wrapper_on_device, backend="tt")
    with torch.no_grad():
        out = compiled(*inputs_on_device)
    return out.to("cpu")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def reconstruct(
    image_path: Path,
    *,
    resolution: str = "480p",
    sharded: bool = True,
    output_dir: Path = OUT_DIR,
) -> dict[str, Path]:
    """Encode-then-decode ``image_path`` on both CPU and TT, log per-stage
    execution times, and write four images to ``output_dir``:

      ``<stem>_original.png``  resized input
      ``<stem>_cpu.png``       CPU reconstruction
      ``<stem>_tt.png``        TT reconstruction
      ``<stem>_merged.png``    side-by-side: original | CPU | TT

    Returns a dict mapping each label to its saved path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = RESOLUTIONS[resolution]
    h, w = shapes["video_h"], shapes["video_w"]

    _log(
        f"image={image_path.name} resolution={resolution} target={h}x{w} "
        f"sharded={sharded}"
    )

    image = load_first_frame_image(image_path, h, w)
    _log(f"input shape={tuple(image.shape)} dtype={image.dtype}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = (
        f"{image_path.stem}_{resolution}_"
        f"{'sharded' if sharded else 'unsharded'}_{ts}"
    )

    orig_out = output_dir / f"{stem}_original.png"
    _save_first_frame(image, orig_out)
    _log(f"saved original -> {orig_out}")

    encoder = VAEEncoderWrapper(load_vae()).eval().bfloat16()
    decoder = VAEDecoderWrapper(load_vae()).eval().bfloat16()

    # ---- CPU encoder ------------------------------------------------------
    _log("running encoder on CPU...")
    t0 = time.perf_counter()
    with torch.no_grad():
        cpu_latent = encoder(image)
    cpu_enc_time = time.perf_counter() - t0
    _log(f"CPU latent shape={tuple(cpu_latent.shape)}")

    # ---- CPU decoder ------------------------------------------------------
    _log("running decoder on CPU...")
    t0 = time.perf_counter()
    with torch.no_grad():
        cpu_pixels = decoder(cpu_latent.to(torch.bfloat16))
    cpu_dec_time = time.perf_counter() - t0
    _log(f"CPU output shape={tuple(cpu_pixels.shape)}")

    cpu_out = output_dir / f"{stem}_cpu.png"
    _save_first_frame(cpu_pixels, cpu_out)
    _log(f"saved CPU image -> {cpu_out}")

    # ---- Mesh -------------------------------------------------------------
    mesh = wan22_mesh() if sharded else None
    if mesh is not None:
        _log(
            f"mesh shape={mesh.mesh_shape} "
            f"axis_names={mesh.axis_names} "
            f"devices={len(mesh.device_ids)}"
        )

    # ---- TT encoder -------------------------------------------------------
    _log("running encoder on TT...")
    t0 = time.perf_counter()
    tt_latent = _run_on_tt(
        encoder,
        [image],
        mesh=mesh,
        shard_module=encoder.vae,
        shard_fn=shard_vae_encoder_specs if sharded else None,
        compile_options=_ENCODER_COMPILE_OPTS,
    )
    tt_enc_time = time.perf_counter() - t0
    _log(f"TT latent shape={tuple(tt_latent.shape)}")

    # ---- TT decoder -------------------------------------------------------
    _log("running decoder on TT...")
    t0 = time.perf_counter()
    tt_pixels = _run_on_tt(
        decoder,
        [tt_latent.to(torch.bfloat16)],
        mesh=mesh,
        shard_module=decoder.vae,
        shard_fn=shard_vae_decoder_specs if sharded else None,
        compile_options=_DECODER_COMPILE_OPTS,
    )
    tt_dec_time = time.perf_counter() - t0
    _log(f"TT output shape={tuple(tt_pixels.shape)}")

    tt_out = output_dir / f"{stem}_tt.png"
    _save_first_frame(tt_pixels, tt_out)
    _log(f"saved TT image -> {tt_out}")

    # ---- Timing summary ---------------------------------------------------
    _log("--- execution time ---")
    _log(f"  encoder on CPU: {cpu_enc_time:7.2f}s")
    _log(f"  decoder on CPU: {cpu_dec_time:7.2f}s")
    _log(f"  encoder on TT : {tt_enc_time:7.2f}s (compile + run)")
    _log(f"  decoder on TT : {tt_dec_time:7.2f}s (compile + run)")

    # ---- PCC summary (TT vs CPU) -----------------------------------------
    _log("--- PCC (TT vs CPU) ---")
    _log_pcc("encoder latent", tt_latent, cpu_latent)
    _log_pcc("decoder pixels", tt_pixels, cpu_pixels)

    # ---- Merged comparison image -----------------------------------------
    merged_out = output_dir / f"{stem}_merged.png"
    _save_merged(image, cpu_pixels, tt_pixels, merged_out)
    _log(f"saved merged comparison -> {merged_out}")

    return {
        "original": orig_out,
        "cpu": cpu_out,
        "tt": tt_out,
        "merged": merged_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Wan 2.2 VAE encoder + decoder on an image on both CPU "
            "and TT, print per-stage execution times, and save the resized "
            "original alongside both reconstructions and a merged "
            "side-by-side comparison."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Path to input image (default: {DEFAULT_IMAGE.name}).",
    )
    parser.add_argument(
        "--resolution",
        choices=list(RESOLUTIONS),
        default="480p",
        help="Wan 2.2 target resolution (default: 480p).",
    )
    parser.add_argument(
        "--no-shard",
        dest="sharded",
        action="store_false",
        help="Disable SPMD sharding (no-op on a single-device setup).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR}).",
    )
    args = parser.parse_args()

    reconstruct(
        image_path=args.image,
        resolution=args.resolution,
        sharded=args.sharded,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
