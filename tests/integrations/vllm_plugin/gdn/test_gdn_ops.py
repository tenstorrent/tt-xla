# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Validate the TT GDN ops against FLA goldens.

Loads the golden bundle produced on the GPU box by ``gen_goldens.py``
(``golden/gdn_golden.pt``). If the bundle is absent, regenerates reference
goldens in-process on CPU (no FLA needed) so the math is still exercised — the
``source`` of each golden (``fla`` vs ``ref``) is reported in the test id.

Set ``GDN_TEST_DEVICE=tt`` (or ``cuda``/``cpu``) to run the TT ops on a specific
device. Default is CPU, which validates the pure-PyTorch math; the TT-device run
additionally confirms the ops lower and execute on hardware.
"""

import os
import sys

import pytest
import torch

_GDN_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_GDN_DIR, "..", "..", "..", ".."))
_PLUGIN_DIR = os.path.join(_REPO_ROOT, "integrations", "vllm_plugin")
for _p in (_GDN_DIR, _PLUGIN_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vllm_tt.layers.gdn import (  # noqa: E402
    tt_causal_conv1d_fn,
    tt_causal_conv1d_update,
    tt_chunk_gated_delta_rule,
    tt_fused_gdn_gating,
    tt_fused_recurrent_gated_delta_rule,
    tt_l2norm_fwd,
)

try:
    from tests.benchmark.utils import compute_pcc  # noqa: E402
except Exception:  # pragma: no cover - fallback if path layout differs
    def compute_pcc(golden_output, device_output):
        a = golden_output.flatten().double()
        b = device_output.flatten().double()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

GOLDEN_PATH = os.path.join(_GDN_DIR, "golden", "gdn_golden.pt")
PCC_FP32 = 0.999
PCC_BF16 = 0.99


def _load_cases():
    if os.path.exists(GOLDEN_PATH):
        return torch.load(GOLDEN_PATH, weights_only=False)["cases"]
    # No committed bundle: regenerate reference goldens on CPU.
    import gen_goldens

    cases = gen_goldens.build_cases("cpu")
    out = []
    for c in cases:
        golden, src = gen_goldens.compute_golden(c)
        out.append({
            "op": c["op"], "id": c["id"], "source": src,
            "inputs": c["inputs"], "params": c["params"], "golden": golden,
        })
    return out


_CASES = _load_cases()
_DEVICE = os.environ.get("GDN_TEST_DEVICE", "cpu")


def _dev(t):
    return t.to(_DEVICE) if isinstance(t, torch.Tensor) else t


def _threshold(case) -> float:
    return PCC_BF16 if "bf16" in case["id"] else PCC_FP32


def _run_op(case):
    """Run the TT op for a case; return ``{name: tensor}`` of produced outputs."""
    op = case["op"]
    inp = {k: _dev(v) for k, v in case["inputs"].items()}
    prm = case["params"]

    if op == "l2norm":
        return {"y": tt_l2norm_fwd(inp["x"])}
    if op == "gating":
        g, beta = tt_fused_gdn_gating(inp["A_log"], inp["a"], inp["b"], inp["dt_bias"])
        return {"g": g, "beta": beta}
    if op == "conv1d_prefill":
        conv_state = inp["conv_state"].clone()
        y = tt_causal_conv1d_fn(
            inp["x"], inp["weight"], inp["bias"], "silu",
            conv_state=conv_state, has_initial_state=inp["has_init"],
            cache_indices=inp["indices"], query_start_loc=inp["qsl"])
        return {"y": y, "conv_state_out": conv_state}
    if op == "conv1d_update":
        conv_state = inp["conv_state"].clone()
        y = tt_causal_conv1d_update(
            inp["x"], conv_state, inp["weight"], inp["bias"], "silu",
            conv_state_indices=inp["indices"])
        return {"y": y, "conv_state_out": conv_state}
    if op == "chunk":
        o, final_state = tt_chunk_gated_delta_rule(
            inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
            scale=prm["scale"], initial_state=inp["initial_state"],
            output_final_state=True, cu_seqlens=inp["cu_seqlens"],
            use_qk_l2norm_in_kernel=prm["l2norm"])
        return {"o": o, "final_state": final_state}
    if op == "recurrent":
        o, final_state = tt_fused_recurrent_gated_delta_rule(
            inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"],
            scale=prm["scale"], initial_state=inp["initial_state"].clone(),
            inplace_final_state=True, cu_seqlens=inp["cu_seqlens"],
            ssm_state_indices=inp["ssm_state_indices"],
            use_qk_l2norm_in_kernel=prm["l2norm"])
        return {"o": o, "final_state": final_state}
    raise ValueError(op)


@pytest.mark.push
@pytest.mark.parametrize(
    "case", _CASES, ids=[f"{c['id']}[{c['source']}]" for c in _CASES]
)
def test_gdn_op_matches_golden(case):
    produced = _run_op(case)
    threshold = _threshold(case)
    for name, golden in case["golden"].items():
        out = produced[name].float().cpu()
        pcc = compute_pcc(golden.float().cpu(), out)
        assert pcc >= threshold, (
            f"{case['id']} output '{name}' PCC {pcc:.5f} < {threshold} "
            f"(source={case['source']}, device={_DEVICE})"
        )
