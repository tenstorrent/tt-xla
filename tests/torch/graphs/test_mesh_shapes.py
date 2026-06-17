# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test_with_random_inputs
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from utils import Category


class _WithInput(torch.nn.Module):
    def forward(self, x):
        return x @ x.T + 1.0


class _NoInput(torch.nn.Module):
    def forward(self):
        # All operands are constants -> StableHLO module has no func args.
        return torch.ones(4, device=xm.xla_device(), dtype=torch.float32) + 1.0


# Case 1: single-device target, module has args.
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_with_input_single_device():
    run_graph_test_with_random_inputs(
        _WithInput(), [(3, 3)], dtype=torch.float32, framework=Framework.TORCH
    )


# Case 2: single-device target, module has no args.
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_no_input_single_device():
    xr.set_device_type("TT")
    device = torch_xla.device()

    model = torch.compile(_NoInput().to(device), backend="tt")
    out = model()
    torch_xla.sync(wait=True)
    assert torch.allclose(out.cpu(), torch.full((4,), 2.0))


# Case 3: multichip target, sharded inputs.
@pytest.mark.push
@pytest.mark.multi_chip
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_with_input_multichip_sharded():
    enable_spmd()
    xr.set_device_type("TT")
    device = torch_xla.device()
    n = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(n)), (1, n), ("batch", "model"))

    a = torch.randn(32, 32 * n, dtype=torch.float32)
    a_dev = a.to(device)
    xs.mark_sharding(a_dev, mesh, (None, "model"))

    model = torch.compile(_WithInput(), backend="tt").to(device)
    out = model(a_dev)
    torch_xla.sync(wait=True)
    expected = a @ a.T + 1.0
    assert torch.allclose(out.cpu(), expected, rtol=5e-2, atol=1e-1)


# Case 4: multichip target, module has no args.
@pytest.mark.push
@pytest.mark.multi_chip
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_no_input_multichip_sharded_output():
    enable_spmd()
    xr.set_device_type("TT")
    n = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(n)), (1, n), ("batch", "model"))

    # Constant tensor with a sharded output, HLO module has no func args.
    t = torch.ones(8, 32 * n, device=xm.xla_device(), dtype=torch.float32)
    xs.mark_sharding(t, mesh, (None, "model"))
    u = t + 1.0
    torch_xla.sync(wait=True)
    assert torch.allclose(u.cpu(), torch.full((8, 32 * n), 2.0))


# Case 5: sequential single-device compiles, no-input first.
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_single_device_sequential_no_input_then_with_input():
    xr.set_device_type("TT")
    device = torch_xla.device()

    no_input = torch.compile(_NoInput().to(device), backend="tt")
    out_a = no_input()
    torch_xla.sync(wait=True)
    assert torch.allclose(out_a.cpu(), torch.full((4,), 2.0))

    with_input = torch.compile(_WithInput().to(device), backend="tt")
    x = torch.randn(3, 3).to(device)
    out_b = with_input(x)
    torch_xla.sync(wait=True)
    assert out_b.cpu().shape == (3, 3)


# Case 6: sequential single-device compiles, with-input first.
@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_single_device_sequential_with_input_then_no_input():
    xr.set_device_type("TT")
    device = torch_xla.device()

    with_input = torch.compile(_WithInput().to(device), backend="tt")
    x = torch.randn(3, 3).to(device)
    out_a = with_input(x)
    torch_xla.sync(wait=True)
    assert out_a.cpu().shape == (3, 3)

    no_input = torch.compile(_NoInput().to(device), backend="tt")
    out_b = no_input()
    torch_xla.sync(wait=True)
    assert torch.allclose(out_b.cpu(), torch.full((4,), 2.0))


# Case 7: SPMD enabled but input fully replicated.
@pytest.mark.push
@pytest.mark.multi_chip
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_with_input_multichip_replicated():
    enable_spmd()
    xr.set_device_type("TT")
    device = torch_xla.device()
    n = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(n)), (1, n), ("batch", "model"))

    a = torch.randn(32, 32, dtype=torch.float32)
    a_dev = a.to(device)
    xs.mark_sharding(a_dev, mesh, (None, None))  # fully replicated

    model = torch.compile(_WithInput(), backend="tt").to(device)
    out = model(a_dev)
    torch_xla.sync(wait=True)
    expected = a @ a.T + 1.0
    assert torch.allclose(out.cpu(), expected, rtol=1e-2, atol=1e-2)


# Case 8: Multichip no mesh.
@pytest.mark.push
@pytest.mark.multi_chip
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_with_input_multichip_no_mesh():
    enable_spmd()
    xr.set_device_type("TT")
    device = torch_xla.device()

    a = torch.randn(32, 32, dtype=torch.float32)
    a_dev = a.to(device)

    model = torch.compile(_WithInput(), backend="tt").to(device)
    out = model(a_dev)
    torch_xla.sync(wait=True)
    expected = a @ a.T + 1.0
    assert torch.allclose(out.cpu(), expected, rtol=1e-2, atol=1e-2)
