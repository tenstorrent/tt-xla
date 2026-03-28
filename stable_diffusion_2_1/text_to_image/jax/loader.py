# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 2.1 (Flax) model loader implementation for text-to-image generation.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from diffusers import FlaxStableDiffusionPipeline

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion 2.1 Flax model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion 2.1 (Flax) model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="flax/stable-diffusion-2-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None
        self.params = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 2.1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Flax Stable Diffusion 2.1 pipeline.

        Args:
            dtype_override: Optional JAX dtype to override the model's default dtype.
                           If not provided, defaults to jnp.bfloat16.

        Returns:
            FlaxStableDiffusionPipeline: The loaded pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else jnp.bfloat16

        self.pipeline, self.params = FlaxStableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            dtype=dtype,
            **kwargs,
        )

        return self.pipeline

    def load_inputs(self, prompt=None, prng_seed=0):
        """Load and return sample inputs for the Stable Diffusion 2.1 model.

        Args:
            prompt: Optional text prompt. If None, uses DEFAULT_PROMPT.
            prng_seed: Integer seed for JAX PRNG key (default: 0).

        Returns:
            dict: Input dictionary with prompt_ids and prng key.
        """
        if self.pipeline is None:
            self.load_model()

        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT

        prompt_ids = self.pipeline.prepare_inputs(prompt_value)
        prng = jax.random.PRNGKey(prng_seed)

        return {
            "prompt_ids": prompt_ids,
            "params": self.params,
            "prng_seed": prng,
        }
