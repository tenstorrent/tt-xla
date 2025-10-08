# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from tests.infra import TorchDeviceConnector
from tests.infra import TorchModelTester, RunMode
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np

class TorchMultichipUnaryModelTester(TorchModelTester):
    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        mesh: Mesh,
        shard_spec_fn=None,
    ) -> None:
        self._this_model = model
        self._input_shape = input_shape
        self._mesh = mesh
        self._shard_spec_fn = shard_spec_fn
        super().__init__()

    def _get_input_activations(self):
        if self._input_activations is None:
            self._input_activations = {"input": torch.randn(self._input_shape)}
        return self._input_activations

    def _get_model(self):
        return self._this_model

    def _get_forward_method_args(self):
        return (self._get_input_activations()["input"],)

    def _get_forward_method_kwargs(self):
        return {}

    def _get_mesh(self):
        return self._mesh

    def _get_shard_specs_function(self):
        return self._shard_spec_fn


def setup_mesh(mesh_shape, axis_names):
    device_ids = np.arange(np.prod(mesh_shape))
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    return mesh


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
@pytest.mark.parametrize("axis_names", [("x", "y")])
@pytest.mark.parametrize("input_shape", [(32, 32)])
@pytest.mark.parametrize("sharding_mode", ["fully_replicated", "partially_sharded"])
def test_linear(mesh_shape, axis_names, input_shape, sharding_mode):
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 256, bias=False)

        def forward(self, x):
            return self.linear(x)

    def shard_spec_function(model):
        if sharding_mode == "partially_sharded":
            # Shard weight matrix along output dimension (dim 0)
            return {model.linear.weight: ("y", None)}
        else:
            # Do not shard anything, fully replicated
            return {}
        
    mesh = setup_mesh(mesh_shape, axis_names)
    model_tester = TorchMultichipUnaryModelTester(
        model=LinearModel(),
        input_shape=input_shape,
        mesh=mesh,
        shard_spec_fn=shard_spec_function,
    )

    model_tester.test()
