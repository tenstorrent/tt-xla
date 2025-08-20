# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OFT Stable Diffusion model loader implementation
"""

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from .src.model_utils import prepare_oft_pipeline, generate_sample_inputs
from typing import Optional


class ModelVariant(StrEnum):
    """Available OFT Stable Diffusion model variants."""

    OFT_STABLE_DIFFUSION_V1_5 = "runwayml/stable-diffusion-v1-5"


class ModelLoader(ForgeModel):
    """OFT Stable Diffusion model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OFT_STABLE_DIFFUSION_V1_5: ModelConfig(
            pretrained_model_name="runwayml/stable-diffusion-v1-5",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OFT_STABLE_DIFFUSION_V1_5

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="oft_stable_diffusion",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.pipeline = None
        self.prompt = "A beautiful mountain landscape during sunset"
        self.num_inference_steps = 30

    def load_model(self):
        """Load and return the OFT Stable Diffusion pipeline.

        Returns:
            StableDiffusionPipeline: The OFT-enabled Stable Diffusion pipeline
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load and prepare pipeline with OFT adapters
        self.pipeline = prepare_oft_pipeline(pretrained_model_name)

        return self.pipeline

    def load_inputs(self):
        """Generate sample inputs for the OFT Stable Diffusion model.

        Returns:
            tuple: Sample inputs that can be fed to the model
                - latents (torch.Tensor): Latent noise tensor [B, C, H, W]
                - timestep (torch.Tensor): Timestep tensor [B]
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings [B, seq_len, dim]
        """
        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.load_model()

        # Generate sample inputs
        inputs = generate_sample_inputs(
            self.pipeline,
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
        )

        return inputs
