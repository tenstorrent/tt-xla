"""
Runner for the refactored VAE decoder TTNN model.

Loads the PyTorch golden model, initializes the TTNN model on a (1,4) mesh device,
runs inference, and compares PCC.
"""

import os
import sys
import time

import torch
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import utils
from model_pt import VaeDecoderPT, get_input, SCALING_FACTOR, SHIFT_FACTOR
from model_ttnn import VaeDecoderTTNN


DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def compare_pcc(tt_out, cpu_out, label=""):
    """Compute PCC between two tensors and print stats."""
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    tt_f = tt_out.float().flatten()
    cpu_f = cpu_out.float().flatten()

    if tt_f.numel() != cpu_f.numel():
        print(f"  SHAPE MISMATCH: TT={tuple(tt_out.shape)} vs CPU={tuple(cpu_out.shape)}")
        print(f"{'='*60}")
        return -1.0

    pcc = torch.corrcoef(torch.stack([tt_f, cpu_f]))[0, 1].item()
    max_diff = (tt_f - cpu_f).abs().max().item()
    mean_diff = (tt_f - cpu_f).abs().mean().item()
    print(f"  TT  output: {tuple(tt_out.shape)}  "
          f"mean={tt_out.mean():.4f}  std={tt_out.std():.4f}")
    print(f"  CPU output: {tuple(cpu_out.shape)}  "
          f"mean={cpu_out.mean():.4f}  std={cpu_out.std():.4f}")
    print(f"  PCC:         {pcc:.6f}  (baseline: 0.998709)")
    print(f"  Max  |diff|: {max_diff:.6f}")
    print(f"  Mean |diff|: {mean_diff:.6f}")
    status = "PASS" if pcc >= 0.99 else "FAIL"
    print(f"  Status:      {status}")
    print(f"{'='*60}")
    return pcc


def main():
    print(f"\nVAE Decoder TTNN Refactored Model Test")
    print("-" * 60)

    # ── 1. Load CPU golden ───────────────────────────────────────────────────
    print("\n[1/4] Loading CPU golden model ...")
    t0 = time.time()
    pt_model = VaeDecoderPT()
    raw_latent = get_input()
    cpu_out = pt_model.forward(raw_latent)
    cpu_ms = (time.time() - t0) * 1000
    print(f"  CPU forward: {cpu_ms:.0f} ms   output={tuple(cpu_out.shape)}")

    # ── 2. Open TTNN mesh device ─────────────────────────────────────────────
    print("\n[2/4] Opening TTNN (1,4) mesh device ...")
    mesh_device = utils.DeviceGetter.get_device((1, 4))
    print(f"  Device: {mesh_device}")

    # ── 3. Initialize TTNN model ─────────────────────────────────────────────
    print("\n[3/4] Initializing TTNN model ...")
    t0 = time.time()
    state_dict = pt_model.state_dict
    ttnn_model = VaeDecoderTTNN(mesh_device, state_dict)
    init_ms = (time.time() - t0) * 1000
    print(f"  Init time: {init_ms:.0f} ms")

    # ── 4. Run inference ─────────────────────────────────────────────────────
    print("\n[4/4] Running TTNN inference ...")

    # Denormalize the latent (same as CPU)
    z_pt = (raw_latent.float() / SCALING_FACTOR) + SHIFT_FACTOR

    for i in range(3):
        # Upload latent to mesh device (replicated)
        z_tt = ttnn.from_torch(
            z_pt.bfloat16(),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        t0 = time.time()
        outputs = ttnn_model(z_tt)
        ttnn.synchronize_device(mesh_device)
        tt_ms = (time.time() - t0) * 1000

        if outputs is None:
            print(f"  Iter {i}: forward returned None (not fully implemented)")
            continue

        # Gather output from mesh (forward returns a list)
        out_device = outputs[0] if isinstance(outputs, list) else outputs
        out_host = ttnn.to_torch(
            ttnn.from_device(out_device),
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        # Take first replica
        n_replicas = out_host.shape[0]
        tt_out = out_host[:n_replicas // 4].float()

        pcc = compare_pcc(tt_out, cpu_out, f"Iteration {i}")
        print(f"  Iter {i}: {tt_ms:.1f} ms, PCC={pcc:.6f}")


if __name__ == "__main__":
    main()
