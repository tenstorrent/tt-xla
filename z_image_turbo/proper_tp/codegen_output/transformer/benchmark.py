# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark: baseline model_ttnn.py vs. optimized model_ttnn_opt.py.

Runs each configuration for N_WARMUP + N_MEASURE iterations, measures median
wall-clock latency (ms), throughput (iter/s), and PCC against the PyTorch CPU
reference.

Tested configurations (can be set via env vars or edited below):
  BASELINE         — original model (all flags False)
  OPT_NORMS        — fast RMS norms only (USE_FAST_NORMS=True, USE_FAST_QK_NORM=True)
  OPT_FINAL_NORM   — fast norms + fast final LayerNorm
  OPT_DIT_NORM     — ttnn.experimental.dit_rms_norm_unary_fused instead of rms_norm
  OPT_MM           — minimal_matmul only (norms baseline)
  OPT_ALL          — all flags on (norms + final_norm + minimal_matmul)

Usage:
    python benchmark.py
    python benchmark.py --configs BASELINE OPT_NORMS OPT_ALL
"""

import argparse
import os
import statistics
import sys
import time

import torch
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import utils
import model_pt
from model_ttnn     import ZImageTransformerTTNN
from model_ttnn_opt import ZImageTransformerTTNNOpt

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
N_WARMUP  = 2
N_MEASURE = 5

# ── Available benchmark configurations ────────────────────────────────────────

CONFIGS = {
    "BASELINE": dict(
        cls=ZImageTransformerTTNN,
        flags={},
        label="Baseline (original model_ttnn.py)",
    ),
    "OPT_NORMS": dict(
        cls=ZImageTransformerTTNNOpt,
        flags=dict(USE_FAST_NORMS=True, USE_FAST_QK_NORM=True,
                   USE_FAST_FINAL_NORM=False, USE_DIT_NORM=False, USE_MINIMAL_MATMUL=False),
        label="Opt: ttnn.rms_norm (hidden + QK norms)",
    ),
    "OPT_FINAL_NORM": dict(
        cls=ZImageTransformerTTNNOpt,
        flags=dict(USE_FAST_NORMS=True, USE_FAST_QK_NORM=True,
                   USE_FAST_FINAL_NORM=True, USE_DIT_NORM=False, USE_MINIMAL_MATMUL=False),
        label="Opt: ttnn.rms_norm + ttnn.layer_norm (final)",
    ),
    "OPT_DIT_NORM": dict(
        cls=ZImageTransformerTTNNOpt,
        flags=dict(USE_FAST_NORMS=True, USE_FAST_QK_NORM=True,
                   USE_FAST_FINAL_NORM=True, USE_DIT_NORM=True, USE_MINIMAL_MATMUL=False),
        label="Opt: dit_rms_norm_unary_fused + ttnn.layer_norm",
    ),
    "OPT_MM": dict(
        cls=ZImageTransformerTTNNOpt,
        flags=dict(USE_FAST_NORMS=False, USE_FAST_QK_NORM=False,
                   USE_FAST_FINAL_NORM=False, USE_DIT_NORM=False, USE_MINIMAL_MATMUL=True),
        label="Opt: minimal_matmul only",
    ),
    "OPT_ALL": dict(
        cls=ZImageTransformerTTNNOpt,
        flags=dict(USE_FAST_NORMS=True, USE_FAST_QK_NORM=True,
                   USE_FAST_FINAL_NORM=True, USE_DIT_NORM=False, USE_MINIMAL_MATMUL=True,
                   USE_FUSED_QKV=False),
        label="Opt: norms + minimal_matmul (separate Q/K/V)",
    ),
    "OPT_FUSED_QKV": dict(
        cls=ZImageTransformerTTNNOpt,
        flags=dict(USE_FAST_NORMS=True, USE_FAST_QK_NORM=True,
                   USE_FAST_FINAL_NORM=True, USE_DIT_NORM=False, USE_MINIMAL_MATMUL=True,
                   USE_FUSED_QKV=True),
        label="Opt: norms + minimal_matmul_split (fused QKV) + nlp_create_qkv_heads",
    ),
}

DEFAULT_RUN_ORDER = [
    "BASELINE", "OPT_NORMS", "OPT_FINAL_NORM", "OPT_DIT_NORM",
    "OPT_MM", "OPT_ALL", "OPT_FUSED_QKV",
]


def _to_device(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def pcc_score(tt_tensor, pt_tensor, mesh_device):
    tt_host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    n = tt_host.shape[0] // 4
    tt_cpu = tt_host[:n].float()
    pt_cpu = pt_tensor.float()
    return torch.corrcoef(torch.stack([tt_cpu.flatten(), pt_cpu.flatten()]))[0, 1].item()


def run_config(cfg_name, cfg, mesh_device, transformer, inputs, pt_ref):
    """Run warmup + measure for one configuration, return stats dict."""
    label = cfg["label"]
    print(f"\n{'='*70}")
    print(f"  Config: {cfg_name}  —  {label}")
    print(f"{'='*70}")

    # Apply feature flags on the class
    model_cls = cfg["cls"]
    for flag, val in cfg["flags"].items():
        setattr(model_cls, flag, val)

    # Build model
    build_start = time.time()
    model = model_cls(mesh_device, transformer)
    build_ms = (time.time() - build_start) * 1000
    print(f"  Model build: {build_ms:.0f} ms")

    tt_latent, tt_timestep, tt_cap_feats = inputs

    # Warmup
    print(f"  Warming up ({N_WARMUP} iters) ...")
    for _ in range(N_WARMUP):
        tt_out = model([tt_latent], tt_timestep, tt_cap_feats)[0]
        _ = ttnn.from_device(tt_out)

    # Measure
    times_ms = []
    pcc_vals = []
    print(f"  Measuring ({N_MEASURE} iters) ...")
    for i in range(N_MEASURE):
        t0 = time.time()
        tt_out = model([tt_latent], tt_timestep, tt_cap_feats)[0]
        _ = ttnn.from_device(tt_out)
        t1 = time.time()
        elapsed_ms = (t1 - t0) * 1000
        pcc = pcc_score(tt_out, pt_ref, mesh_device)
        times_ms.append(elapsed_ms)
        pcc_vals.append(pcc)
        print(f"    iter {i}: {elapsed_ms:.1f} ms  PCC={pcc:.6f}")

    median_ms   = statistics.median(times_ms)
    mean_ms     = statistics.mean(times_ms)
    min_ms      = min(times_ms)
    fps         = 1000.0 / median_ms
    mean_pcc    = statistics.mean(pcc_vals)
    ok          = mean_pcc >= 0.995

    print(f"\n  ── Results ──")
    print(f"  Median: {median_ms:.1f} ms  |  Mean: {mean_ms:.1f} ms  |  Min: {min_ms:.1f} ms")
    print(f"  Throughput: {fps:.3f} iter/s")
    print(f"  Mean PCC: {mean_pcc:.6f}  {'✓' if ok else '✗ BELOW 0.995'}")

    # Reset flags to default
    for flag, val in cfg["flags"].items():
        setattr(model_cls, flag, getattr(ZImageTransformerTTNNOpt, flag, val))

    del model

    return {
        "name": cfg_name,
        "label": label,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "fps": fps,
        "mean_pcc": mean_pcc,
        "ok": ok,
        "times_ms": times_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", nargs="+", default=DEFAULT_RUN_ORDER,
        choices=list(CONFIGS.keys()),
        help="Which configs to benchmark (default: all)",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading transformer from {model_pt.MODEL_ID}/transformer ...")
    transformer = model_pt.load_model()

    # ── CPU reference inputs + outputs ───────────────────────────────────────
    latents   = [torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)]
    timestep  = torch.tensor([0.5], dtype=torch.bfloat16)
    cap_feats = torch.randn(32, 2560, dtype=torch.bfloat16)

    print("\nRunning CPU reference (1 iteration) ...")
    pt_outputs = model_pt.forward(transformer, latents, timestep, cap_feats)
    pt_ref = pt_outputs[0]
    print(f"  CPU output: shape={tuple(pt_ref.shape)}")

    # ── Apply head padding before TTNN model creation ────────────────────────
    model_pt.pad_heads(transformer)

    # ── Open mesh device ──────────────────────────────────────────────────────
    mesh_device = utils.DeviceGetter.get_device((1, 4))
    print(f"\nMesh device: {mesh_device}")

    # ── Upload runtime inputs ─────────────────────────────────────────────────
    cap_3d       = cap_feats.unsqueeze(0)
    tt_timestep  = _to_device(timestep.reshape(1), mesh_device)
    tt_cap_feats = _to_device(cap_3d, mesh_device)
    tt_latent    = _to_device(latents[0], mesh_device)
    inputs = (tt_latent, tt_timestep, tt_cap_feats)

    # ── Run benchmarks ────────────────────────────────────────────────────────
    results = []
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        r = run_config(cfg_name, cfg, mesh_device, transformer, inputs, pt_ref)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*90)
    print("  BENCHMARK SUMMARY")
    print("="*90)
    baseline = next((r for r in results if r["name"] == "BASELINE"), results[0])
    hdr = f"  {'Config':<20} {'Median ms':>10} {'Mean ms':>10} {'fps':>8} {'PCC':>10} {'vs baseline':>12}  Status"
    print(hdr)
    print("-"*90)
    for r in results:
        speedup = baseline["median_ms"] / r["median_ms"] if r["name"] != "BASELINE" else 1.0
        sign    = "+" if speedup > 1.0 else ""
        cmp     = f"{sign}{(speedup-1)*100:.1f}%" if r["name"] != "BASELINE" else "baseline"
        status  = "✓" if r["ok"] else "✗"
        print(f"  {r['name']:<20} {r['median_ms']:>10.1f} {r['mean_ms']:>10.1f} "
              f"{r['fps']:>8.3f} {r['mean_pcc']:>10.6f} {cmp:>12}  {status}")
    print("="*90)

    # ── Final verdict ─────────────────────────────────────────────────────────
    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n✗  {len(failed)} config(s) below PCC 0.995: {[r['name'] for r in failed]}")
        sys.exit(1)
    else:
        print(f"\n✓  All {len(results)} configurations pass PCC ≥ 0.995")


if __name__ == "__main__":
    main()
