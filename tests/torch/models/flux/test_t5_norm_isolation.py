# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""T5 encoder PCC debug — Sanity 2b: isolate the final T5LayerNorm (RMSNorm).

Capture the REAL pre-final-norm hidden state (huge ~1e5 outliers) on CPU, then
run ONLY the final T5LayerNorm on device vs CPU with that identical input:
  - low PCC  -> the RMSNorm kernel itself is the structural culprit
  - high PCC -> norm is faithful; it only EXPOSES upstream accumulated divergence
"""

import os

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    load_t5_text_encoder,
    tokenize_t5,
)

_DTYPE = getattr(torch, os.environ.get("TT_T5_DTYPE", "bfloat16"))


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    return float("nan") if denom == 0 else float((va @ vb) / denom)


def _report(tt_res, cpu_res, args, kwargs):
    tt = tt_res.to("cpu").to(torch.float64)
    cpu = cpu_res.to(torch.float64)
    print("\n==== FINAL T5LayerNorm ISOLATION (identical input) ====", flush=True)
    print(f"input  maxabs={float(args[0].abs().max()):.2f}", flush=True)
    print(f"out    pcc={_pcc(tt, cpu):.6f}", flush=True)
    print(f"out    cpu_maxabs={float(cpu.abs().max()):.4f} "
          f"dev_maxabs={float(tt.abs().max()):.4f}", flush=True)
    print("==== END ====\n", flush=True)


@pytest.mark.single_device
def test_t5_final_norm_isolation():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    enc = load_t5_text_encoder(_DTYPE).eval()

    # Capture the input to the final RMSNorm during a normal CPU forward.
    captured = {}

    def pre_hook(_mod, inp):
        captured["x"] = inp[0].detach().clone()

    handle = enc.encoder.final_layer_norm.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        enc(tokenize_t5())
    handle.remove()

    x = captured["x"]  # [1, 512, 4096], magnitudes ~1e5
    print(f"\n[norm-iso] dtype={_DTYPE} captured pre-norm input {tuple(x.shape)} "
          f"maxabs={float(x.abs().max()):.1f}\n", flush=True)

    # Isolate ONLY the final layer norm; same input on device and CPU.
    norm = enc.encoder.final_layer_norm
    run_graph_test(
        norm,
        [x],
        framework=Framework.TORCH,
        custom_comparator=_report,
    )
