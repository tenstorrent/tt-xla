# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD VAE ft-mse (Flax) model loader implementation.

Loads the enterprise-explorers/sd-vae-ft-mse-flax Flax VAE decoder, a
fine-tuned kl-f8 autoencoder for Stable Diffusion optimized with MSE loss
for smoother reconstructions.

Available variants:
- FT_MSE: Default MSE-fine-tuned VAE decoder
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

REPO_ID = "enterprise-explorers/sd-vae-ft-mse-flax"

# Standard SD VAE uses 4 latent channels
LATENT_CHANNELS = 4

# Small test dimensions for VAE decoder inputs
# SD VAE compression: 8x spatial
LATENT_HEIGHT = 8
LATENT_WIDTH = 8


class ModelVariant(StrEnum):
    """Available SD VAE ft-mse Flax model variants."""

    FT_MSE = "ft_mse"


class ModelLoader(ForgeModel):
    """SD VAE ft-mse Flax model loader."""

    _VARIANTS = {
        ModelVariant.FT_MSE: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FT_MSE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._vae = None
        self._params = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD_VAE_FT_MSE_FLAX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Flax VAE model.

        Returns:
            FlaxAutoencoderKL instance.
        """
        from diffusers import FlaxAutoencoderKL

        if self._vae is None:
            dtype = dtype_override if dtype_override is not None else jnp.float32
            self._vae, self._params = FlaxAutoencoderKL.from_pretrained(
                REPO_ID,
                dtype=dtype,
            )
        return self._vae

    def load_inputs(self, **kwargs) -> Any:
        """Prepare latent inputs for the VAE decoder.

        Returns:
            Latent array of shape [batch, height, width, channels] (channels-last for Flax).
        """
        dtype = kwargs.get("dtype_override", jnp.float32)
        key = jax.random.PRNGKey(0)
        # Flax uses channels-last format: [batch, height, width, channels]
        return jax.random.normal(
            key,
            (1, LATENT_HEIGHT, LATENT_WIDTH, LATENT_CHANNELS),
            dtype=dtype,
        )
