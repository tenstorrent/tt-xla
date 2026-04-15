# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
StateEncoder (CategorySpecificMLP) isolation tests.

The state_encoder uses CategorySpecificLinear which does self.W[cat_ids]
(fancy indexing on a nn.Parameter). The compiler lowers this to
ttnn::embedding, triggering L1 overflow on large weight tables.

state_encoder = CategorySpecificMLP(num_categories=32, input_dim=64,
                                     hidden_dim=1024, output_dim=1536)
  layer1: CategorySpecificLinear(32, 64, 1024)   W=[32, 64, 1024]
  layer2: CategorySpecificLinear(32, 1024, 1536)  W=[32, 1024, 1536]

Inputs from CPU log:
  state:         [1, 1, 64]  bfloat16
  embodiment_id: [1]         int64

Run in order:
  1. test_1_state_encoder      — full CategorySpecificMLP
  2. test_2_layer1             — layer1 alone: W[cat_ids] on [32, 64, 1024]
  3. test_3_layer2             — layer2 alone: W[cat_ids] on [32, 1024, 1536]
"""

import os
import torch
import numpy as np
import pytest
from infra import Framework, run_op_test
from utils import Category

from tests.runner.requirements import RequirementsManager

LOADER_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "third_party", "tt_forge_models", "issac_groot", "pytorch", "loader.py",
))


def _load_state_encoder_and_inputs():
    """Load the model, extract state_encoder and its real inputs."""
    from third_party.tt_forge_models.issac_groot.pytorch import ModelLoader, ModelVariant

    loader = ModelLoader(ModelVariant("Gr00t_N1.5_3B"))
    model = loader.load_model()
    model.eval()

    inputs = loader.load_inputs()
    cpu_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            cpu_inputs[k] = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            cpu_inputs[k] = v.cpu()
        else:
            cpu_inputs[k] = v

    gr00t = model.model
    backbone_inputs, _ = gr00t.prepare_input(cpu_inputs)

    compute_dtype = next(gr00t.action_head.parameters()).dtype
    state = backbone_inputs["state"].to(dtype=compute_dtype)
    embodiment_id = backbone_inputs["embodiment_id"]

    state_encoder = gr00t.action_head.state_encoder

    return state_encoder, state, embodiment_id


# ── Wrappers ──────────────────────────────────────────────────────────────────


class StateEncoderWrapper(torch.nn.Module):
    """Full CategorySpecificMLP: layer1 + relu + layer2."""

    def __init__(self, state_encoder):
        super().__init__()
        self.state_encoder = state_encoder

    def forward(self, state, embodiment_id):
        return self.state_encoder(state, embodiment_id)


class Layer1Wrapper(torch.nn.Module):
    """layer1 alone: CategorySpecificLinear(32, 64, 1024).
    W=[32, 64, 1024] → W[cat_ids] selects a [1, 64, 1024] slice.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, state, embodiment_id):
        return self.layer(state, embodiment_id)


class Layer2Wrapper(torch.nn.Module):
    """layer2 alone: CategorySpecificLinear(32, 1024, 1536).
    W=[32, 1024, 1536] → W[cat_ids] selects a [1, 1024, 1536] slice.
    This is the prime suspect — 32*1024*1536*2 = ~96 MB weight table.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, embodiment_id):
        return self.layer(x, embodiment_id)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_1_state_encoder():
    """
    Full state_encoder: CategorySpecificMLP(32, 64, 1024, 1536).
    Input: state [1, 1, 64] bfloat16, embodiment_id [1] int64.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        state_encoder, state, embodiment_id = _load_state_encoder_and_inputs()

        print(f"\n[test_1] state_encoder (full):")
        print(f"  state: shape={state.shape}, dtype={state.dtype}")
        print(f"  embodiment_id: shape={embodiment_id.shape}, val={embodiment_id}")
        print(f"  layer1.W: {list(state_encoder.layer1.W.shape)}")
        print(f"  layer2.W: {list(state_encoder.layer2.W.shape)}")

        wrapper = StateEncoderWrapper(state_encoder)
        wrapper.eval()

        run_op_test(
            wrapper, [state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_2_layer1():
    """
    layer1: CategorySpecificLinear(32, 64, 1024).
    W=[32, 64, 1024] — W[cat_ids] → bmm with state [1,1,64].
    Weight size: 32 * 64 * 1024 * 2 = 4,194,304 B (~4 MB).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        state_encoder, state, embodiment_id = _load_state_encoder_and_inputs()

        layer1 = state_encoder.layer1
        print(f"\n[test_2] layer1:")
        print(f"  W: shape={list(layer1.W.shape)}, size={layer1.W.numel() * 2} bytes")
        print(f"  b: shape={list(layer1.b.shape)}")
        print(f"  state: shape={state.shape}, dtype={state.dtype}")

        wrapper = Layer1Wrapper(layer1)
        wrapper.eval()

        run_op_test(
            wrapper, [state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_3_layer2():
    """
    layer2: CategorySpecificLinear(32, 1024, 1536).
    W=[32, 1024, 1536] — W[cat_ids] → bmm with layer1 output [1,1,1024].
    Weight size: 32 * 1024 * 1536 * 2 = 100,663,296 B (~96 MB).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        state_encoder, state, embodiment_id = _load_state_encoder_and_inputs()

        layer1 = state_encoder.layer1
        layer2 = state_encoder.layer2

        with torch.no_grad():
            layer1_out = torch.relu(layer1(state, embodiment_id))

        print(f"\n[test_3] layer2:")
        print(f"  W: shape={list(layer2.W.shape)}, size={layer2.W.numel() * 2} bytes")
        print(f"  b: shape={list(layer2.b.shape)}")
        print(f"  input (layer1 output): shape={layer1_out.shape}, dtype={layer1_out.dtype}")

        wrapper = Layer2Wrapper(layer2)
        wrapper.eval()

        run_op_test(
            wrapper, [layer1_out, embodiment_id],
            framework=Framework.TORCH,
        )


# ── layer2 sub-op isolation ──────────────────────────────────────────────────


class SelectWWrapper(torch.nn.Module):
    """Isolates self.W[cat_ids] — fancy indexing on W=[32, 1024, 1536]."""

    def __init__(self, W):
        super().__init__()
        self.W = W

    def forward(self, cat_ids):
        return self.W[cat_ids]


class SelectBWrapper(torch.nn.Module):
    """Isolates self.b[cat_ids] — fancy indexing on b=[32, 1536]."""

    def __init__(self, b):
        super().__init__()
        self.b = b

    def forward(self, cat_ids):
        return self.b[cat_ids]


class BmmAddWrapper(torch.nn.Module):
    """Isolates torch.bmm(x, selected_W) + selected_b.unsqueeze(1)."""

    def __init__(self):
        super().__init__()

    def forward(self, x, selected_W, selected_b):
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_4_layer2_select_W():
    """
    layer2 W[cat_ids]: fancy index on W=[32, 1024, 1536].
    This is lowered to ttnn::embedding — the L1 overflow suspect.
    W size: 32 * 1024 * 1536 * 2 = 100,663,296 B (~96 MB).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        state_encoder, _, embodiment_id = _load_state_encoder_and_inputs()
        layer2 = state_encoder.layer2

        print(f"\n[test_4] layer2 W[cat_ids]:")
        print(f"  W: shape={list(layer2.W.shape)}, size={layer2.W.numel() * 2} bytes")
        print(f"  cat_ids (embodiment_id): {embodiment_id}")

        wrapper = SelectWWrapper(layer2.W)
        wrapper.eval()

        run_op_test(
            wrapper, [embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_5_layer2_select_b():
    """
    layer2 b[cat_ids]: fancy index on b=[32, 1536].
    b size: 32 * 1536 * 2 = 98,304 B (~96 KB) — should pass.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        state_encoder, _, embodiment_id = _load_state_encoder_and_inputs()
        layer2 = state_encoder.layer2

        print(f"\n[test_5] layer2 b[cat_ids]:")
        print(f"  b: shape={list(layer2.b.shape)}, size={layer2.b.numel() * 2} bytes")

        wrapper = SelectBWrapper(layer2.b)
        wrapper.eval()

        run_op_test(
            wrapper, [embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_6_layer2_bmm_add():
    """
    torch.bmm(x, selected_W) + selected_b.unsqueeze(1).
    x=[1,1,1024], selected_W=[1,1024,1536], selected_b=[1,1536].
    Pure matmul + add — no embedding, should pass.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        state_encoder, state, embodiment_id = _load_state_encoder_and_inputs()
        layer1 = state_encoder.layer1
        layer2 = state_encoder.layer2

        with torch.no_grad():
            layer1_out = torch.relu(layer1(state, embodiment_id))
            selected_W = layer2.W[embodiment_id]
            selected_b = layer2.b[embodiment_id]

        print(f"\n[test_6] layer2 bmm+add:")
        print(f"  x: shape={layer1_out.shape}")
        print(f"  selected_W: shape={selected_W.shape}")
        print(f"  selected_b: shape={selected_b.shape}")

        wrapper = BmmAddWrapper()
        wrapper.eval()

        run_op_test(
            wrapper, [layer1_out, selected_W, selected_b],
            framework=Framework.TORCH,
        )
