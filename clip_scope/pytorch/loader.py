# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLIP-ViT-L-Scope sparse autoencoder model loader implementation.

Reference: https://huggingface.co/lewington/CLIP-ViT-L-scope
"""
import torch
from clipscope import ConfiguredViT, TopKSAE
from datasets import load_dataset
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available CLIP-ViT-L-Scope SAE layer variants."""

    LAYER_22_RESID = "Layer_22_Resid"


class CLIPScopeModel(torch.nn.Module):
    """Wraps ConfiguredViT + TopKSAE into a single nn.Module for forge compatibility."""

    def __init__(self, sae, transformer, location):
        super().__init__()
        self.sae = sae
        self.transformer = transformer
        self.location = location

    def forward(self, pixel_values):
        activations = self.transformer.all_activations(pixel_values)[self.location]
        # Use the CLS token activation
        cls_activation = activations[:, 0]
        output = self.sae(cls_activation)
        return output


class ModelLoader(ForgeModel):
    """CLIP-ViT-L-Scope sparse autoencoder model loader implementation."""

    _VARIANTS = {
        ModelVariant.LAYER_22_RESID: ModelConfig(
            pretrained_model_name="lewington/CLIP-ViT-L-scope",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LAYER_22_RESID

    # Mapping from variant to checkpoint path and layer config
    _LAYER_CONFIGS = {
        ModelVariant.LAYER_22_RESID: {
            "checkpoint": "22_resid/1200013184.pt",
            "location": (22, "resid"),
        },
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CLIP-ViT-L-Scope",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CLIP-ViT-L-Scope SAE model.

        Returns:
            torch.nn.Module: A wrapped module combining ConfiguredViT and TopKSAE.
        """
        layer_config = self._LAYER_CONFIGS[self._variant]
        device = "cpu"

        sae = TopKSAE.from_pretrained(
            checkpoint=layer_config["checkpoint"], device=device
        )
        location = layer_config["location"]
        transformer = ConfiguredViT([location], device=device)

        model = CLIPScopeModel(sae, transformer, location)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return a sample input image for the model.

        Returns:
            PIL.Image.Image: A sample 224x224 RGB image from the cats dataset.
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].resize((224, 224))
        return image
