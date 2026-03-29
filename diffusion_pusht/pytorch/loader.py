# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diffusion Policy PushT model loader implementation for action prediction.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _setup_policies_namespace() -> None:
    """Register lerobot.policies in sys.modules so subpackage imports work when this loader
    is imported outside the normal lerobot package context (e.g. via tt-forge-models dynamic
    import). Without this, 'from lerobot.policies.diffusion...' can fail with import errors.
    """
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        return
    policies_path = Path(spec.origin).resolve().parent / "policies"
    if not policies_path.exists():
        return
    if "lerobot.policies" in sys.modules:
        return
    policies_module = types.ModuleType("lerobot.policies")
    policies_module.__path__ = [str(policies_path)]
    sys.modules["lerobot.policies"] = policies_module


class DiffusionPolicyInferenceWrapper(torch.nn.Module):
    """Wraps DiffusionPolicy to use predict_action_chunk (inference) instead of forward (training).

    DiffusionPolicy.forward() computes training loss; for inference we use predict_action_chunk.
    See: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/modeling_diffusion.py
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict) -> torch.Tensor:
        """Run inference via predict_action_chunk. Returns action tensor."""
        return self.policy.predict_action_chunk(batch)


class ModelVariant(StrEnum):
    """Available Diffusion Policy PushT model variants."""

    DIFFUSION_PUSHT = "diffusion_pusht"


class ModelLoader(ForgeModel):
    """Diffusion Policy PushT model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.DIFFUSION_PUSHT: ModelConfig(
            pretrained_model_name="lerobot/diffusion_pusht",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIFFUSION_PUSHT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DiffusionPushT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        _setup_policies_namespace()
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        model = DiffusionPolicy.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.to(device)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        else:
            model = model.to(dtype=torch.float32)
        model.eval()
        self.config = model.config
        return DiffusionPolicyInferenceWrapper(model)

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        _setup_policies_namespace()
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        if self.config is None:
            self.config = DiffusionConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        n_obs_steps = self.config.n_obs_steps
        dtype = dtype_override or torch.float32

        # Build dummy observation batch matching the model's expected input format.
        # The diffusion policy expects n_obs_steps frames stacked along dim 1.
        batch = {}
        for key, feature in (self.config.input_features or {}).items():
            if feature.type.name == "VISUAL":
                # Image observations: (batch, n_obs_steps, C, H, W)
                c, h, w = feature.shape
                batch[key] = torch.zeros(
                    (batch_size, n_obs_steps, c, h, w), dtype=dtype, device=device
                )
            elif feature.type.name in ("STATE", "ENV"):
                # State observations: (batch, n_obs_steps, state_dim)
                batch[key] = torch.zeros(
                    (batch_size, n_obs_steps, *feature.shape),
                    dtype=dtype,
                    device=device,
                )

        return {"batch": batch}

    def unpack_forward_output(self, fwd_output):
        """predict_action_chunk returns action tensor directly."""
        return fwd_output
