# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prithvi-EO 2.0 model loader implementation for Earth observation tasks.

Prithvi-EO is a Vision Transformer pretrained with Masked Autoencoder (MAE)
objective on multi-spectral satellite imagery. It uses 3D patch embeddings
and temporal/location encodings.
"""

import importlib.util
import json
import torch
from typing import Optional

from huggingface_hub import hf_hub_download

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

HF_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"


def _load_prithvi_mae_module():
    """Download and dynamically import the PrithviMAE module from HuggingFace."""
    mae_path = hf_hub_download(repo_id=HF_REPO_ID, filename="prithvi_mae.py")
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelVariant(StrEnum):
    """Available Prithvi-EO model variants."""

    V2_300M_TL = "V2_300M_TL"


class ModelLoader(ForgeModel):
    """Prithvi-EO model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2_300M_TL: ModelConfig(
            pretrained_model_name=HF_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_300M_TL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Prithvi-EO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        repo_id = self._variant_config.pretrained_model_name

        # Download config and model architecture from HuggingFace
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        prithvi_mae = _load_prithvi_mae_module()
        model = prithvi_mae.PrithviMAE(**config["pretrained_cfg"])

        # Load pretrained weights
        weights_path = hf_hub_download(
            repo_id=repo_id, filename="Prithvi_EO_V2_300M_TL.pt"
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Position embeddings are recomputed from config, so remove from checkpoint
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # Input shape: (B, C, T, H, W)
        # 6 bands (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2), 4 time steps, 224x224
        num_channels = 6
        num_frames = 4
        img_size = 224

        # Synthetic normalized input (zero-mean, unit-variance)
        pixel_values = torch.randn(
            batch_size, num_channels, num_frames, img_size, img_size
        )

        # Temporal coordinates: (B, T, 2) with [year, julian_day]
        temporal_coords = torch.tensor(
            [[2023, 1], [2023, 91], [2023, 182], [2023, 274]], dtype=torch.float32
        )
        temporal_coords = temporal_coords.unsqueeze(0).expand(batch_size, -1, -1)

        # Location coordinates: (B, 2) with [lon, lat]
        location_coords = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        location_coords = location_coords.expand(batch_size, -1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)
            temporal_coords = temporal_coords.to(dtype_override)
            location_coords = location_coords.to(dtype_override)

        return (pixel_values, temporal_coords, location_coords)
