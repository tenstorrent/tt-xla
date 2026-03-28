# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prithvi EO V2 model loader implementation for feature extraction.

Prithvi-EO-2.0 is a geospatial foundation model based on a Vision Transformer
with Masked Autoencoder (MAE) pretraining. It processes multi-temporal,
multi-spectral satellite imagery using 3D patch embeddings.
"""

import importlib.util
import json
import sys

import torch
from huggingface_hub import hf_hub_download
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Prithvi EO V2 feature extraction model variants."""

    V2_300M = "V2_300M"


class ModelLoader(ForgeModel):
    """Prithvi EO V2 model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.V2_300M: ModelConfig(
            pretrained_model_name="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_300M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Prithvi EO V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_prithvi_mae_class(self, repo_id):
        """Download and dynamically import the PrithviMAE class from HuggingFace."""
        mae_path = hf_hub_download(repo_id=repo_id, filename="prithvi_mae.py")

        spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["prithvi_mae"] = module
        spec.loader.exec_module(module)

        return module.PrithviMAE

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Prithvi EO V2 model instance.

        Downloads the model definition, config, and weights from HuggingFace Hub,
        then constructs and loads the PrithviMAE model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Prithvi EO V2 model instance.
        """
        repo_id = self._variant_config.pretrained_model_name

        PrithviMAE = self._load_prithvi_mae_class(repo_id)

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)["pretrained_cfg"]

        model = PrithviMAE(**config)

        weights_path = hf_hub_download(
            repo_id=repo_id, filename="Prithvi_EO_V2_300M.pt"
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Discard fixed positional embedding weights (recomputed from config)
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Prithvi EO V2 model.

        Generates synthetic multi-temporal satellite imagery matching the expected
        input format: (batch, channels, time_steps, height, width) where channels=6
        spectral bands and time_steps=4.

        Args:
            dtype_override: Optional torch.dtype to override the input tensor's dtype.
            batch_size: Batch size for the inputs.

        Returns:
            tuple: (input_tensor, temporal_coords, location_coords) matching the
                   model's forward signature.
        """
        # Model expects input shape: (B, C, T, H, W)
        # C=6 bands (B02, B03, B04, B05, B06, B07)
        # T=4 time steps, H=W=224
        in_chans = 6
        num_frames = 4
        img_size = 224

        input_tensor = torch.randn(batch_size, in_chans, num_frames, img_size, img_size)

        if dtype_override is not None:
            input_tensor = input_tensor.to(dtype_override)

        # Temporal coordinates: (year, julian_day) per time step
        temporal_coords = torch.tensor(
            [[2023, 26], [2023, 106], [2023, 201], [2023, 266]], dtype=torch.float
        ).unsqueeze(0)

        # Location coordinates: (longitude, latitude)
        location_coords = torch.tensor([[-103.0, 25.0]], dtype=torch.float).unsqueeze(0)

        return input_tensor, temporal_coords, location_coords
