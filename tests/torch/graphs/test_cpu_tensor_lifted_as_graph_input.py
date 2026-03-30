# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reproduces graph breaks caused by non-buffer tensor attributes (like YOLOP's
self.grid and self.stride) that stay on CPU when a model is moved to XLA.
AOTAutograd lifts these as CPU graph inputs; extract_compiled_graph then
partitions around them.

The TT backend fixes this by force-moving CPU args to XLA before
extract_compiled_graph. These tests verify the openxla path breaks and the
TT backend path does not.
"""

from unittest.mock import patch

import pytest
import torch
import torch_xla.core.dynamo_bridge as dynamo_bridge
import tt_torch.backend.backend as backend_module
from tt_torch.backend.backend import aot_backend, fw_compiler
from utils import Category

# Vibed tests


def _count_optimized_mod_calls(graph_module):
    return sum(
        1
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and getattr(node.target, "__name__", "") == "optimized_mod"
    )


class DetectLikeModel(torch.nn.Module):
    """Mimics YOLOP's Detect: 3 parallel layers with non-buffer CPU attributes.

    self.grids (plain list) and self.stride (plain tensor) stay on CPU after
    .to("xla"). self.stride[i] is used without .to(device), so the mul ops
    get CPU args and are marked unsupported by the partitioner.
    """

    def __init__(self, nl=3, features=4):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(features, features) for _ in range(nl)]
        )
        self.grids = [torch.zeros(1, features) for _ in range(nl)]
        self.stride = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, x0, x1, x2):
        inputs = [x0, x1, x2]
        outputs = []
        for i, layer in enumerate(self.layers):
            out = layer(inputs[i])
            out = out + self.grids[i].to(inputs[i].device)
            out = out * self.stride[i]  # no .to() — like YOLOP
            outputs.append(out)
        return torch.cat(outputs, dim=0)


def _make_model():
    model = DetectLikeModel().to(torch.bfloat16).to("xla")
    assert model.stride.device.type == "cpu"
    return model


def _make_inputs():
    return [torch.randn(2, 4, dtype=torch.bfloat16, device="xla") for _ in range(3)]


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_openxla_graph_breaks_with_cpu_tensor_attributes():
    """openxla backend has no CPU->XLA fix, so CPU attributes cause graph breaks."""
    captured = {}

    def openxla_backend(gm, example_inputs):
        compiled = dynamo_bridge.extract_compiled_graph(gm, example_inputs)
        captured["compiled"] = compiled
        return compiled

    compiled_model = torch.compile(_make_model(), backend=openxla_backend)
    compiled_model(*_make_inputs())

    assert _count_optimized_mod_calls(captured["compiled"]) > 1


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_tt_backend_no_graph_breaks_with_cpu_tensor_attributes():
    """TT backend force-moves CPU args to XLA, preventing graph breaks."""
    captured = {}

    def capturing_fw_compiler(gm, example_inputs, options, aot_graph_signature=None):
        executor = fw_compiler(gm, example_inputs, options, aot_graph_signature)
        captured["executor"] = executor
        return executor

    with patch.object(backend_module, "fw_compiler", capturing_fw_compiler):
        compiled_model = torch.compile(
            _make_model(),
            backend=lambda gm, inputs: aot_backend(gm, inputs, options=None),
        )
        compiled_model(*_make_inputs())

    assert captured["executor"].compiled_graph is not None
    assert _count_optimized_mod_calls(captured["executor"].compiled_graph) == 1
