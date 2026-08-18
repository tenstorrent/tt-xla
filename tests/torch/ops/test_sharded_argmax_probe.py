# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Probes for the vocab-sharded greedy sampling path (issue #4494 debug).

test_sharded_argmax_probe: composite_argmax on a pre-sharded (None,"model")
    tensor (clean input) -> confirms the distributed argmax op itself.

test_sharded_compute_logits_argmax: FAITHFUL repro of compute_logits+sample:
    hidden @ vocab-sharded weight -> sharding_constraint(None,"model") ->
    composite_argmax. Isolates whether the sharded model forward feeds correct
    per-shard logits into the (working) argmax.

  MESH_SHAPE=2,4 pytest -svv tests/torch/ops/test_sharded_argmax_probe.py
"""

import os

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from tt_torch.composite_ops import composite_argmax
from tt_torch.sharding import sharding_constraint_tensor
from utils import Category


def _mesh_shape():
    num_devices = xr.global_runtime_device_count()
    env_mesh = os.environ.get("MESH_SHAPE")
    return (
        tuple(int(x) for x in env_mesh.split(",")) if env_mesh else (1, num_devices)
    ), num_devices


class _ArgmaxProbe(torch.nn.Module):
    def forward(self, logits):
        return composite_argmax(logits, dim=-1, keepdim=True)


def _make_cmp(expected):
    def _cmp(device_output, golden_output, args, kwargs):
        dev = int(device_output.reshape(-1)[0].cpu().item())
        gold = int(golden_output.reshape(-1)[0].cpu().item())
        print(f"\n  expected={expected} golden(cpu)={gold} device={dev}")
        assert gold == expected, f"golden wrong? {gold} != {expected}"
        assert dev == gold, f"DEVICE ARGMAX WRONG: device={dev} vs global={gold}"
        print("  OK: device global argmax correct")

    return _cmp


def _make_value_cmp(logits_full):
    """Correctness with ties: the device index must hit the global MAX value."""
    logits_full = logits_full.float().reshape(-1)
    gmax = logits_full.max().item()

    def _cmp(device_output, golden_output, args, kwargs):
        dev = int(device_output.reshape(-1)[0].cpu().item())
        dev_val = logits_full[dev].item()
        gold = int(golden_output.reshape(-1)[0].cpu().item())
        print(
            f"\n  global_max_val={gmax:.5f}  device_idx={dev} device_val={dev_val:.5f}"
            f"  golden_idx={gold} golden_val={logits_full[gold].item():.5f}"
        )
        assert dev_val == gmax, (
            f"DEVICE ARGMAX WRONG: device_idx={dev} has val {dev_val:.5f} "
            f"!= global max {gmax:.5f}"
        )
        print("  OK: device index hits the global max value")

    return _cmp


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.record_test_properties(
    category=Category.OP_TEST, torch_op_name="torch.argmax"
)
def test_sharded_argmax_probe():
    vocab = int(os.environ.get("PROBE_VOCAB", "32000"))
    mesh_shape, num_devices = _mesh_shape()
    model_axis = mesh_shape[1]
    shard_w = vocab // model_axis
    spike_shard = min(2, model_axis - 1)
    spike_idx = spike_shard * shard_w + 123
    dtype = torch.bfloat16 if os.environ.get("PROBE_DTYPE") == "bf16" else torch.float32
    logits = torch.zeros(1, vocab, dtype=dtype)
    logits[0, spike_idx] = 100.0
    mesh = Mesh(list(range(num_devices)), mesh_shape, ("batch", "model"))
    print(
        f"\n=== argmax probe mesh={mesh_shape} dtype={dtype} spike_idx={spike_idx} ==="
    )
    run_graph_test(
        _ArgmaxProbe(),
        [logits],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=lambda m, a, k: {a[0]: (None, "model")},
        custom_comparator=_make_cmp(spike_idx),
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.record_test_properties(
    category=Category.OP_TEST, torch_op_name="torch.argmax"
)
def test_sharded_argmax_random():
    """Realistic distribution: random bf16 logits (tie-prone) sharded
    (None,"model"), distributed argmax must still hit the global max value."""
    import torch as _t

    vocab = int(os.environ.get("PROBE_VOCAB", "32000"))
    mesh_shape, num_devices = _mesh_shape()
    dtype = torch.bfloat16 if os.environ.get("PROBE_DTYPE") == "bf16" else torch.float32
    _t.manual_seed(0)
    logits = _t.randn(1, vocab, dtype=torch.float32).to(dtype)
    mesh = Mesh(list(range(num_devices)), mesh_shape, ("batch", "model"))
    print(f"\n=== random argmax probe mesh={mesh_shape} dtype={dtype} ===")

    def shard_spec_fn(model, args, kwargs):
        return {args[0]: (None, "model")}

    run_graph_test(
        _ArgmaxProbe(),
        [logits],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        custom_comparator=_make_value_cmp(logits),
    )


class _ComputeLogitsArgmax(torch.nn.Module):
    """hidden @ vocab-sharded weight -> constrain (None,'model') -> argmax."""

    def __init__(self, mesh, weight):
        super().__init__()
        self.mesh = mesh
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, hidden):
        logits = hidden @ self.weight  # [1, hidden] @ [hidden, vocab] -> [1, vocab]
        logits = sharding_constraint_tensor(logits, self.mesh, (None, "model"))
        return composite_argmax(logits, dim=-1, keepdim=True)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.record_test_properties(
    category=Category.OP_TEST, torch_op_name="torch.argmax"
)
def test_sharded_compute_logits_argmax():
    vocab = int(os.environ.get("PROBE_VOCAB", "32000"))
    hidden_dim = 2048
    mesh_shape, num_devices = _mesh_shape()
    model_axis = mesh_shape[1]
    shard_w = vocab // model_axis
    spike_shard = min(2, model_axis - 1)
    spike_idx = spike_shard * shard_w + 123

    # weight[:, spike_idx] large so logits peak at spike_idx; hidden = ones.
    dtype = torch.bfloat16 if os.environ.get("PROBE_DTYPE") == "bf16" else torch.float32
    weight = torch.zeros(hidden_dim, vocab, dtype=dtype)
    weight[:, spike_idx] = 1.0
    hidden = torch.ones(1, hidden_dim, dtype=dtype)
    print(f"  dtype={dtype}")
    mesh = Mesh(list(range(num_devices)), mesh_shape, ("batch", "model"))
    print(
        f"\n=== compute_logits+argmax probe mesh={mesh_shape} spike_idx={spike_idx} ==="
    )

    # Match the real sampler graph: weight vocab-sharded on "model", and hidden
    # feature-dim sharded on "batch" (_axis_0) so the graph must all_gather it
    # before the dot_general (as gdump_fullmodel_1 does).
    hidden_spec = (
        (None, "batch") if os.environ.get("PROBE_SHARD_HIDDEN") else (None, None)
    )

    def shard_spec_fn(model, args, kwargs):
        return {model.weight: (None, "model"), args[0]: hidden_spec}

    run_graph_test(
        _ComputeLogitsArgmax(mesh, weight),
        [hidden],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        custom_comparator=_make_cmp(spike_idx),
    )
