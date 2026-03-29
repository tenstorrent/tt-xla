# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prithvi-EO-2.0 model loader implementation for image feature extraction.

Prithvi-EO-2.0 is a Vision Transformer pretrained as a Masked Autoencoder (MAE)
for Earth Observation. It uses 3D patch embeddings to handle spatiotemporal
multispectral satellite imagery with temporal and location encodings.
"""
import importlib.util
import json

import torch
from huggingface_hub import hf_hub_download
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
    """Available Prithvi-EO model variants."""

    TINY_TL = "Tiny_TL"


class ModelLoader(ForgeModel):
    """Prithvi-EO-2.0 model loader for Earth Observation feature extraction."""

    _VARIANTS = {
        ModelVariant.TINY_TL: ModelConfig(
            pretrained_model_name="ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_TL

    _WEIGHTS_FILENAME = "Prithvi_EO_V2_tiny_TL.pt"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Prithvi_EO_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Prithvi-EO MAE model.

        Downloads the model definition and weights from HuggingFace Hub,
        constructs the PrithviMAE model from config, and loads pretrained weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Prithvi-EO MAE model instance.
        """
        repo_id = self._variant_config.pretrained_model_name

        # Download model definition and config
        model_file = hf_hub_download(repo_id=repo_id, filename="prithvi_mae.py")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

        # Dynamically import PrithviMAE from the downloaded model file
        spec = importlib.util.spec_from_file_location("prithvi_mae", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PrithviMAE = module.PrithviMAE

        # Load config
        with open(config_path) as f:
            config = json.load(f)["pretrained_cfg"]

        model = PrithviMAE(
            img_size=config["img_size"],
            num_frames=config["num_frames"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            decoder_embed_dim=config["decoder_embed_dim"],
            decoder_depth=config["decoder_depth"],
            decoder_num_heads=config["decoder_num_heads"],
            mlp_ratio=config["mlp_ratio"],
            coords_encoding=config.get("coords_encoding", []),
            coords_scale_learn=config.get("coords_scale_learn", True),
            mask_ratio=config["mask_ratio"],
            norm_pix_loss=config["norm_pix_loss"],
        )

        # Load pretrained weights
        weights_path = hf_hub_download(repo_id=repo_id, filename=self._WEIGHTS_FILENAME)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Prithvi-EO model.

        The model expects:
        - pixel_values: (B, C, T, H, W) multispectral satellite imagery
        - temporal_coords: (B, T, 2) with [year, day_of_year] per timestep
        - location_coords: (B, 2) with [longitude, latitude]

        Args:
            dtype_override: Optional torch.dtype to override input tensor dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # 6-channel multispectral imagery, 4 timesteps, 224x224 spatial
        pixel_values = torch.randn(batch_size, 6, 4, 224, 224, dtype=dtype)

        # Temporal coordinates: [year, day_of_year] for each timestep
        temporal_coords = (
            torch.tensor([[2018, 26], [2018, 106], [2018, 201], [2018, 266]])
            .float()
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        # Location coordinates: [longitude, latitude]
        location_coords = torch.tensor([[-103.0, 25.0]]).float().expand(batch_size, -1)

        if dtype_override is not None:
            temporal_coords = temporal_coords.to(dtype_override)
            location_coords = location_coords.to(dtype_override)

        return {
            "pixel_values": pixel_values,
            "temporal_coords": temporal_coords,
            "location_coords": location_coords,
        }
