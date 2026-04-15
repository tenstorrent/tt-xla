# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
FlowmatchingActionHead incremental block-level sanity tests.

Tests each stage of the action head forward incrementally to pinpoint
exactly which combination of ops triggers the L1 overflow.

Inputs from CPU log:
  backbone_features: [1, 0, 2048] bfloat16  (zero-length from backbone)
  state:             [1, 1, 64]   bfloat16
  embodiment_id:     [1]          int64
  action_horizon:    16
  action_dim:        32
  num_inference_timesteps: 4

Run in order:
  1. test_1_process_backbone_output  — vlln + vl_self_attention
  2. test_2_process_and_state_encoder — process_backbone_output + state_encoder
  (next batch based on results:)
  3. test_3_till_action_encoder      — + for-loop body through action_encoder
  4. test_4_till_position_embedding  — + position_embedding
  5. test_5_till_future_tokens       — + future_tokens cat
  6. test_6_till_model               — + DiT model forward
  7. test_7_till_actions             — + action_decoder + euler update (full action head, 1 step)
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


def _load_action_head_and_inputs():
    """Load the full model, run backbone on CPU, return action_head + its inputs."""
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

    with torch.no_grad():
        backbone_outputs = gr00t.backbone(backbone_inputs)

    backbone_features = backbone_outputs["backbone_features"]

    return gr00t.action_head, backbone_features, state, embodiment_id


# ── Wrappers ──────────────────────────────────────────────────────────────────


class ProcessBackboneOutputWrapper(torch.nn.Module):
    """Stage 1: vlln(LayerNorm) + vl_self_attention(SelfAttentionTransformer)."""

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention

    def forward(self, backbone_features):
        x = self.vlln(backbone_features)
        x = self.vl_self_attention(x)
        return x


class ProcessAndStateEncoderWrapper(torch.nn.Module):
    """Stage 2: process_backbone_output + state_encoder.
    Returns (vl_embs, state_features) concatenated along dim=1 for a single output tensor.
    """

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.state_encoder = action_head.state_encoder

    def forward(self, backbone_features, state, embodiment_id):
        vl_embs = self.vlln(backbone_features)
        vl_embs = self.vl_self_attention(vl_embs)
        state_features = self.state_encoder(state, embodiment_id)
        return state_features


class TillActionEncoderWrapper(torch.nn.Module):
    """Stage 3: process_backbone_output + state_encoder + 1 loop iteration through action_encoder."""

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
        self.action_horizon = action_head.action_horizon
        self.action_dim = action_head.action_dim

    def forward(self, backbone_features, state, embodiment_id):
        vl_embs = self.vlln(backbone_features)
        vl_embs = self.vl_self_attention(vl_embs)
        state_features = self.state_encoder(state, embodiment_id)

        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
        )

        t_discretized = 0
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        return action_features


class TillPositionEmbeddingWrapper(torch.nn.Module):
    """Stage 4: ... + position_embedding (Embedding(1024, 1536))."""

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
        self.position_embedding = action_head.position_embedding
        self.action_horizon = action_head.action_horizon
        self.action_dim = action_head.action_dim

    def forward(self, backbone_features, state, embodiment_id):
        vl_embs = self.vlln(backbone_features)
        vl_embs = self.vl_self_attention(vl_embs)
        state_features = self.state_encoder(state, embodiment_id)

        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
        )

        t_discretized = 0
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs
        return action_features


class TillFutureTokensWrapper(torch.nn.Module):
    """Stage 5: ... + future_tokens cat (sa_embs construction)."""

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
        self.position_embedding = action_head.position_embedding
        self.future_tokens = action_head.future_tokens
        self.action_horizon = action_head.action_horizon
        self.action_dim = action_head.action_dim

    def forward(self, backbone_features, state, embodiment_id):
        vl_embs = self.vlln(backbone_features)
        vl_embs = self.vl_self_attention(vl_embs)
        state_features = self.state_encoder(state, embodiment_id)

        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
        )

        t_discretized = 0
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        return sa_embs


class TillModelWrapper(torch.nn.Module):
    """Stage 6: ... + DiT model forward."""

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
        self.position_embedding = action_head.position_embedding
        self.future_tokens = action_head.future_tokens
        self.dit_model = action_head.model
        self.action_horizon = action_head.action_horizon
        self.action_dim = action_head.action_dim

    def forward(self, backbone_features, state, embodiment_id):
        vl_embs = self.vlln(backbone_features)
        vl_embs = self.vl_self_attention(vl_embs)
        state_features = self.state_encoder(state, embodiment_id)

        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
        )

        t_discretized = 0
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.dit_model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        return model_output


class TillActionsWrapper(torch.nn.Module):
    """Stage 7: full action head forward for 1 denoising step (no loop)."""

    def __init__(self, action_head):
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.state_encoder = action_head.state_encoder
        self.action_encoder = action_head.action_encoder
        self.position_embedding = action_head.position_embedding
        self.future_tokens = action_head.future_tokens
        self.dit_model = action_head.model
        self.action_decoder = action_head.action_decoder
        self.action_horizon = action_head.action_horizon
        self.action_dim = action_head.action_dim
        self.num_inference_timesteps = action_head.num_inference_timesteps

    def forward(self, backbone_features, state, embodiment_id):
        vl_embs = self.vlln(backbone_features)
        vl_embs = self.vl_self_attention(vl_embs)
        state_features = self.state_encoder(state, embodiment_id)

        batch_size = vl_embs.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
        )
        dt = 1.0 / self.num_inference_timesteps

        t_discretized = 0
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized)
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.dit_model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -self.action_horizon:]
        actions = actions + dt * pred_velocity
        return actions


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_1_process_backbone_output():
    """
    1. vlln (LayerNorm(2048)) + vl_self_attention (SelfAttentionTransformer).
    Input: backbone_features [1, 0, 2048] bfloat16 (zero-length).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, _, _ = _load_action_head_and_inputs()

        print(f"\n[test_1] process_backbone_output:")
        print(f"  backbone_features: shape={backbone_features.shape}, dtype={backbone_features.dtype}")

        wrapper = ProcessBackboneOutputWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_2_process_and_state_encoder():
    """
    2. process_backbone_output + state_encoder(CategorySpecificMLP).
    Inputs: backbone_features [1, 0, 2048], state [1, 1, 64] bfloat16, embodiment_id [1] int64.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, state, embodiment_id = _load_action_head_and_inputs()

        print(f"\n[test_2] process + state_encoder:")
        print(f"  backbone_features: shape={backbone_features.shape}, dtype={backbone_features.dtype}")
        print(f"  state: shape={state.shape}, dtype={state.dtype}")
        print(f"  embodiment_id: shape={embodiment_id.shape}, dtype={embodiment_id.dtype}")

        wrapper = ProcessAndStateEncoderWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_3_till_action_encoder():
    """
    3. ... + 1 denoising step through action_encoder (MultiEmbodimentActionEncoder).
    Adds: torch.randn [1,16,32] noise + action_encoder (CategorySpecificLinear + SinusoidalPositionalEncoding).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, state, embodiment_id = _load_action_head_and_inputs()

        print(f"\n[test_3] till action_encoder:")
        print(f"  action_horizon={action_head.action_horizon}, action_dim={action_head.action_dim}")

        torch.manual_seed(42)
        wrapper = TillActionEncoderWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_4_till_position_embedding():
    """
    4. ... + position_embedding (Embedding(1024, 1536)).
    This is the suspected L1 overflow culprit — weight table is 3.0 MB > 1.4 MB L1.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, state, embodiment_id = _load_action_head_and_inputs()

        print(f"\n[test_4] till position_embedding:")
        pe = action_head.position_embedding
        print(f"  position_embedding: Embedding({pe.num_embeddings}, {pe.embedding_dim})")
        print(f"  weight size: {pe.num_embeddings * pe.embedding_dim * 2} bytes")

        torch.manual_seed(42)
        wrapper = TillPositionEmbeddingWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_5_till_future_tokens():
    """
    5. ... + future_tokens (Embedding(32, 1536)) cat into sa_embs.
    Adds: future_tokens.weight expand + torch.cat(state_features, future_tokens, action_features).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, state, embodiment_id = _load_action_head_and_inputs()

        print(f"\n[test_5] till future_tokens:")
        ft = action_head.future_tokens
        print(f"  future_tokens: Embedding({ft.num_embeddings}, {ft.embedding_dim})")

        torch.manual_seed(42)
        wrapper = TillFutureTokensWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_6_till_model():
    """
    6. ... + DiT model forward (12-layer transformer).
    Adds: DiT(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=t).
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, state, embodiment_id = _load_action_head_and_inputs()

        print(f"\n[test_6] till model (DiT):")
        print(f"  DiT params: {sum(p.numel() for p in action_head.model.parameters()):,}")

        torch.manual_seed(42)
        wrapper = TillModelWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_7_till_actions():
    """
    7. ... + action_decoder + euler update (full action head, 1 denoising step).
    Adds: action_decoder(CategorySpecificMLP) + pred_velocity slice + actions update.
    """
    with RequirementsManager.for_loader(LOADER_PATH):
        action_head, backbone_features, state, embodiment_id = _load_action_head_and_inputs()

        print(f"\n[test_7] till actions (full 1-step):")
        print(f"  num_inference_timesteps={action_head.num_inference_timesteps}")

        torch.manual_seed(42)
        wrapper = TillActionsWrapper(action_head)
        wrapper.eval()

        run_op_test(
            wrapper, [backbone_features, state, embodiment_id],
            framework=Framework.TORCH,
        )
