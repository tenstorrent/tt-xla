# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PPO Seals CartPole model loader implementation
"""

from typing import Optional
from dataclasses import dataclass

import torch
from stable_baselines3 import PPO

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


@dataclass
class PPOCartPoleConfig(ModelConfig):
    """Configuration specific to PPO CartPole models"""

    source: ModelSource = ModelSource.HUGGING_FACE


class ModelVariant(StrEnum):
    """Available PPO CartPole model variants."""

    PPO_SEALS_CARTPOLE_V0 = "ppo_seals_cartpole_v0"


class ModelLoader(ForgeModel):
    """PPO Seals CartPole model loader implementation."""

    _VARIANTS = {
        ModelVariant.PPO_SEALS_CARTPOLE_V0: PPOCartPoleConfig(
            pretrained_model_name="HumanCompatibleAI/ppo-seals-CartPole-v0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PPO_SEALS_CARTPOLE_V0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PPOSealsCartPole",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PPO policy network.

        Returns:
            torch.nn.Module: The PPO MLP policy network in eval mode.
        """
        model_name = self._variant_config.pretrained_model_name
        ppo_model = PPO.load(model_name)
        policy = ppo_model.policy
        policy.eval()

        if dtype_override is not None:
            policy = policy.to(dtype_override)

        return policy

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample CartPole observations as model inputs.

        CartPole observations are 4-dimensional vectors:
        [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Returns:
            torch.Tensor: Sample observation tensor of shape (batch_size, 4).
        """
        observation = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).repeat(batch_size, 1)

        if dtype_override is not None:
            observation = observation.to(dtype_override)

        return observation
