# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Memory-proof H2 unit test: FSDP-style batch-axis weight reshard vs CPU golden.

Context: the gemma-4-31B DP+TP accuracy bug (mesh [8,4], TP=4) only appears with
``shard_weights_on_batch_axis=True`` (FSDP-style). Under that flag the "batch"
mesh axis does double duty: it is the DP row-split for activations AND it shards
the *feature/contraction* dim of the weights, so the hidden state's feature dim
rides the "batch" axis and every linear inserts a batch-axis reshard collective.
``model_runner`` then constrains the selected hidden to ``(None, None, "batch")``
and warns that "a mismatched axis silently corrupts the select output on a 2D
mesh". H2 = that reshard/select is mis-lowered.

This test reproduces exactly that op chain on tiny synthetic tensors — a
column-parallel matmul -> activation -> row-parallel matmul (feature lands on
"batch") -> select — and compares the device result to a CPU golden. No 31B, no
KV cache, no OOM. It runs on the galaxy in seconds.

Discriminator (same golden, two sharding layouts):
  fsdp=True   weights ("model","batch")/("batch","model"), hidden (None,None,"batch")
              == the suspect FSDP layout the model uses.
  fsdp=False  weights ("model",None)/(None,"model"),      hidden ("batch",None,None)
              == coherent pure-TP + DP-row-split layout.

Reading the result:
  fsdp=True FAILS, fsdp=False PASSES   -> H2 CONFIRMED: FSDP batch-axis reshard is
                                          mis-lowered (the model-level bug).
  both PASS                            -> the reshard math is fine; look elsewhere
                                          (H1/H3/H4/H5). The full-model bug is not
                                          in this op chain.
  both FAIL                            -> a more basic TP/reshard problem.

Run:
  pytest -svv tests/torch/test_dp_tp_fsdp_reshard.py
  pytest -svv tests/torch/test_dp_tp_fsdp_reshard.py -k "fsdp_True and 8_4"
"""
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from infra.utilities.torch_multichip_utils import get_mesh
from tt_torch.sharding import sharding_constraint_hook

from tests.infra import ComparisonConfig

# Tiny dims, all divisible by every mesh-axis size we sweep (2, 4, 8, 16) so
# safe_mark_sharding never silently falls back to replication.
BATCH = 32  # sequences (DP row-split dim); divisible by max "batch" axis (8)
SEQ = 8
HIDDEN = 128  # divisible by 16
INTER = 256  # divisible by 16
DTYPE = torch.bfloat16


class FSDPBlock(torch.nn.Module):
    """gate/up (column-parallel) -> GELU -> down (row-parallel) -> select.

    Mirrors the real linear chain: down's output feature dim is sharded on the
    "batch" axis under FSDP, then a per-sequence select — the flagged path.
    """

    def __init__(self):
        super().__init__()
        # Weights are [out, in]. Column: out=INTER; Row: out=HIDDEN.
        self.gate_up = torch.nn.Linear(HIDDEN, INTER, bias=False, dtype=DTYPE)
        self.down = torch.nn.Linear(INTER, HIDDEN, bias=False, dtype=DTYPE)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.gate_up(x))  # [B, S, INTER]
        y = self.down(h)  # [B, S, HIDDEN] -- feature lands on "batch" under FSDP
        # Emulate select_hidden_states: last token per sequence.
        return y[:, -1, :]  # [B, HIDDEN]


@pytest.mark.nightly
@pytest.mark.bh_galaxy
@pytest.mark.parametrize(
    "mesh_shape",
    [(8, 4), (4, 8), (2, 16)],
    ids=lambda s: f"{s[0]}_{s[1]}",
)
@pytest.mark.parametrize("fsdp", [True, False], ids=lambda b: f"fsdp_{b}")
def test_fsdp_batch_axis_reshard(mesh_shape, fsdp):
    """FSDP batch-axis weight reshard + select vs CPU golden (H2)."""
    num_devices = xr.global_runtime_device_count()
    assert num_devices == mesh_shape[0] * mesh_shape[1], (
        f"mesh {mesh_shape} needs {mesh_shape[0] * mesh_shape[1]} devices, "
        f"have {num_devices}"
    )

    mesh = get_mesh(mesh_shape, ("batch", "model"))
    batch_axis = "batch" if fsdp else None
    # hidden layout: FSDP puts the feature dim on "batch"; the coherent
    # non-FSDP layout puts the sequence (row) dim on "batch" instead.
    hidden_spec = (None, None, "batch") if fsdp else ("batch", None, None)

    model = FSDPBlock()
    # Constrain the row-parallel (down) output exactly as model_runner does for
    # select_hidden_states — the "silently corrupts the select output" path.
    hook = sharding_constraint_hook(model.down, mesh, hidden_spec)
    model.down.register_forward_hook(hook)

    def shard_spec_fn(model, args, kwargs):
        return {
            # DP: split the activation batch dim on "batch" (always).
            args[0]: ("batch", None, None),
            # Column-parallel (+FSDP batch-axis on contraction dim).
            model.gate_up.weight: ("model", batch_axis),
            # Row-parallel (+FSDP batch-axis on output/feature dim).
            model.down.weight: (batch_axis, "model"),
        }

    activation = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE)

    run_op_test(
        model,
        [activation],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
