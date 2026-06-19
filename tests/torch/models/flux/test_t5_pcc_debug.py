# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""T5 encoder PCC debug — Sanity 1: per-layer PCC curve.

Returns ALL hidden states (embeddings + 24 block outputs) stacked, runs the
SAME graph on TT device and CPU, and prints device-vs-CPU PCC at every depth.

Shape of the curve disambiguates:
  - smooth monotonic decline  -> accumulative bf16 drift (no single culprit)
  - sharp cliff at block j     -> a mis-lowered op inside block j
"""

import os

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

# Toggle dtype for the precision-confirmation run: TT_T5_DTYPE=float32|bfloat16
_DTYPE = getattr(torch, os.environ.get("TT_T5_DTYPE", "bfloat16"))

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.flux.pytorch.src.model_utils import (
    load_t5_text_encoder,
    tokenize_t5,
)


class T5AllHiddenStates(torch.nn.Module):
    """Return every hidden state stacked: [num_layers+1, B, S, H]."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        out = self.text_encoder(input_ids, output_hidden_states=True)
        # out.hidden_states: tuple of (embeddings, block_1, ..., block_24)
        return torch.stack(out.hidden_states, dim=0)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    if denom == 0:
        return float("nan")
    return float((va @ vb) / denom)


def _rmsnorm(x):
    # per-token RMS normalization (mimics T5LayerNorm, no weight), reveals the
    # signal subspace hidden under the huge outlier dims.
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _per_layer_report(tt_res, cpu_res, args, kwargs):
    # tt_res / cpu_res: [25, 1, 512, 4096]
    tt_res = tt_res.to("cpu")
    n = tt_res.shape[0]
    print("\n==== T5 PER-LAYER PCC + MAGNITUDE (device vs CPU) ====", flush=True)
    print(
        f"{'depth':>7} {'raw_pcc':>10} {'norm_pcc':>10} "
        f"{'cpu_maxabs':>12} {'dev_maxabs':>12} {'cpu_meanabs':>12}",
        flush=True,
    )
    for i in range(n):
        c = cpu_res[i].to(torch.float64)
        dv = tt_res[i].to(torch.float64)
        raw = _pcc(dv, c)
        norm = _pcc(_rmsnorm(dv), _rmsnorm(c))
        label = "embed" if i == 0 else f"block{i:02d}"
        print(
            f"{label:>7} {raw:>10.5f} {norm:>10.5f} "
            f"{float(c.abs().max()):>12.2f} {float(dv.abs().max()):>12.2f} "
            f"{float(c.abs().mean()):>12.4f}",
            flush=True,
        )
    print("==== END ====\n", flush=True)


@pytest.mark.single_device
def test_t5_per_layer_pcc():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    print(f"\n[t5-pcc-debug] dtype = {_DTYPE}\n", flush=True)
    encoder = T5AllHiddenStates(load_t5_text_encoder(_DTYPE)).eval()
    inputs = [tokenize_t5()]

    run_graph_test(
        encoder,
        inputs,
        framework=Framework.TORCH,
        custom_comparator=_per_layer_report,
    )
