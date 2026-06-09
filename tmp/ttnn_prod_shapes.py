# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Batch-position-invariance probe at the EXACT production shapes that the
1-layer Llama-3.2-3B prefill graph emits (extracted from the lowered TTNN MLIR
for b32, K=1, token-padding=32).

Each op gets N=32 IDENTICAL batch rows; we check the per-row outputs are
bit-identical. Divergence here => the tt-metal kernel is not row-invariant at
THIS shape (a tt-metal issue). All-identical => the full-model divergence is
compositional / from an op not covered here.

Shapes (from ttnn_b32k1_g2):
  qkv matmul : [1024,3072] x [5120,3072]^T -> [1024,5120]
  sdpa       : Q[32,24,32,128] K[32,8,32,128] V[32,8,32,128] is_causal scale=0.0883883461
  o_proj     : [1024,3072] x [3072,3072]^T
  gate_up    : [1024,3072] x [16384,3072]^T -> [1024,16384]
  down       : [1024,8192] x [3072,8192]^T  -> [1024,3072]
  rms_norm   : [1024,3072]
"""
import os

import torch
import ttnn

FP32 = os.environ.get("PS_FP32_ACC", "0") == "1"
# PS_MLIR_CFG=1: pass the EXACT compute config tt-mlir emits for the prefill
# graph (math_fidelity=hifi4, fp32_dest_acc_en=False) instead of ttnn's default
# (cfg=None). This is what distinguishes the divergent graph from the
# bit-identical default-config standalone run.
MLIR_CFG = os.environ.get("PS_MLIR_CFG", "0") == "1"
SCALE = 0.0883883461
NB = 32  # batch (users)
TP = 32  # token padding (tile)


def _bf16(t):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _cfg(device):
    if MLIR_CFG:
        # Exactly what the lowered TTNN MLIR specifies for the matmuls.
        return ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
    if not FP32:
        return None
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _report(name, t):
    """t: [NB, TP, ...] -> compare each user's [TP, ...] block to user 0."""
    row0 = t[0]
    diffs = []
    for i in range(1, t.shape[0]):
        if not torch.equal(t[i], row0):
            d = (t[i].float() - row0.float()).abs().max().item()
            diffs.append((i, d))
    tag = (
        "BIT-IDENTICAL"
        if not diffs
        else f"DIVERGE ({len(diffs)} rows, max|d|={max(d for _,d in diffs):.3e})"
    )
    print(f"[{name:18s}] fp32={int(FP32)} N={t.shape[0]} -> {tag}")
    if diffs[:6]:
        print("    first:", diffs[:6])


def _mm_test(device, name, K, Nout):
    # M = NB*TP rows; row block of TP per user, all users identical.
    torch.manual_seed(0)
    one = torch.randn(TP, K)
    A = one.unsqueeze(0).repeat(NB, 1, 1).reshape(NB * TP, K)
    B = torch.randn(Nout, K)  # transpose_b=true
    At, Bt = _bf16(A).to(device), _bf16(B).to(device)
    out = ttnn.matmul(At, Bt, transpose_b=True, compute_kernel_config=_cfg(device))
    ot = ttnn.to_torch(out).reshape(NB, TP, Nout)
    _report(name, ot)


def test_qkv(d):
    _mm_test(d, "qkv_matmul", 3072, 5120)


def test_oproj(d):
    _mm_test(d, "o_proj", 3072, 3072)


def test_gateup(d):
    _mm_test(d, "gate_up", 3072, 16384)


def test_down(d):
    _mm_test(d, "down", 8192, 3072)


def test_sdpa(device):
    H, Hkv, S, D = 24, 8, TP, 128
    torch.manual_seed(0)
    q1 = torch.randn(H, S, D)
    k1 = torch.randn(Hkv, S, D)
    v1 = torch.randn(Hkv, S, D)
    Q = q1.unsqueeze(0).repeat(NB, 1, 1, 1)
    K = k1.unsqueeze(0).repeat(NB, 1, 1, 1)
    V = v1.unsqueeze(0).repeat(NB, 1, 1, 1)
    Qt, Kt, Vt = _bf16(Q).to(device), _bf16(K).to(device), _bf16(V).to(device)
    out = ttnn.transformer.scaled_dot_product_attention(
        Qt, Kt, Vt, is_causal=True, scale=SCALE, compute_kernel_config=_cfg(device)
    )
    ot = ttnn.to_torch(out)  # [NB, H, S, D]
    _report("sdpa", ot)


def test_rmsnorm(device):
    torch.manual_seed(0)
    one = torch.randn(TP, 3072)
    X = one.unsqueeze(0).repeat(NB, 1, 1).reshape(NB * TP, 3072)
    w = torch.randn(3072)
    Xt, wt = _bf16(X).to(device), _bf16(w).to(device)
    out = ttnn.rms_norm(Xt, weight=wt, epsilon=1e-5, compute_kernel_config=_cfg(device))
    ot = ttnn.to_torch(out).reshape(NB, TP, 3072)
    _report("rms_norm", ot)


def main():
    device = ttnn.open_device(device_id=0)
    try:
        for name, fn in (
            ("qkv", test_qkv),
            ("sdpa", test_sdpa),
            ("o_proj", test_oproj),
            ("gate_up", test_gateup),
            ("down", test_down),
            ("rms_norm", test_rmsnorm),
        ):
            try:
                fn(device)
            except Exception:
                import traceback

                print(f"{name} ERROR:")
                traceback.print_exc()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
