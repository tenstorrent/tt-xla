# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Decide how to generate the recurrent-decode golden.

Evidence so far: FLA's ``fused_recurrent_gated_delta_rule`` is unreliable on
these shapes (data-dependent PCC, NaNs, numerical blow-ups) — the fingerprint of
uninitialized output / autotune garbage. This script checks two things on the
EXACT ``gen_goldens`` inputs:

  1. DETERMINISM: call the recurrent kernel twice; if the two outputs disagree,
     the kernel is returning garbage and cannot be a golden source.
  2. CHUNK-as-golden: the chunk kernel computes the SAME math and already
     validates at >0.99. If it reproduces the reference for the recurrent
     inputs, generate the recurrent golden via chunk (still ``source=fla``).

Run on the GPU box (same dir as ``_reference.py``/``gen_goldens.py``)::

    python diag_recurrent.py
"""

import torch

from _reference import ref_delta_rule
import gen_goldens

from fla.ops.gated_delta_rule import (
    fused_recurrent_gated_delta_rule as fla_rec,
    chunk_gated_delta_rule as fla_chunk,
)


def pcc(a, b):
    a = a.flatten().double().cpu()
    b = b.flatten().double().cpu()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def per_token(ref_o, o):
    if o.shape != ref_o.shape or o.dim() != 4 or o.shape[1] <= 1:
        return None
    return [round(pcc(ref_o[:, t], o[:, t]), 3) for t in range(o.shape[1])]


def run(o_out):
    return (o_out[0] if isinstance(o_out, (tuple, list)) else o_out).float()


def main():
    dev = "cuda"
    cases = [c for c in gen_goldens.build_cases(dev) if c["op"] == "recurrent"]
    print(f"{len(cases)} recurrent cases\n")

    def to(t):
        return t.to(dev) if isinstance(t, torch.Tensor) else t

    for c in cases:
        inp, prm = c["inputs"], c["params"]
        scale, l2 = prm["scale"], prm["l2norm"]

        ref_o, _ = ref_delta_rule(
            inp["q"].float(), inp["k"].float(), inp["v"].float(),
            inp["g"], inp["beta"], scale,
            initial_state=inp["initial_state"],
            cu_seqlens=inp.get("cu_seqlens"), l2norm=l2)

        q, k, v = to(inp["q"]), to(inp["k"]), to(inp["v"])
        g, beta = to(inp["g"]), to(inp["beta"])
        cu = to(inp.get("cu_seqlens"))
        idx = to(inp.get("ssm_state_indices"))
        ist0 = to(inp["initial_state"])

        print(f"=== {c['id']}  q{tuple(inp['q'].shape)} state{tuple(inp['initial_state'].shape)} ===")

        # (1) DETERMINISM of the recurrent kernel (transpose=True layout).
        def rec_once():
            ist = ist0.transpose(-1, -2).contiguous()
            return run(fla_rec(q, k, v, g, beta, scale=scale, initial_state=ist.clone(),
                               cu_seqlens=cu, ssm_state_indices=idx,
                               use_qk_l2norm_in_kernel=l2))
        o1, o2 = rec_once(), rec_once()
        same = pcc(o1, o2)
        print(f"    RECURRENT determinism: PCC(run1,run2)={same:.4f}"
              f"  (1.0 == deterministic)")

        # (2) CHUNK kernel on the same inputs, both state layouts.
        for transpose in (False, True):
            ist = ist0.transpose(-1, -2).contiguous() if transpose else ist0
            try:
                o = run(fla_chunk(q, k, v, g, beta, scale=scale,
                                  initial_state=ist.clone(), output_final_state=True,
                                  cu_seqlens=cu, use_qk_l2norm_in_kernel=l2))
                print(f"    CHUNK   transpose={transpose!s:5s}  PCC={pcc(ref_o, o):.4f}"
                      f"  per-token={per_token(ref_o, o)}")
            except Exception as e:  # noqa: BLE001
                print(f"    CHUNK   transpose={transpose!s:5s}  RAISED "
                      f"{type(e).__name__}: {str(e)[:80]}")
        print()


if __name__ == "__main__":
    main()
