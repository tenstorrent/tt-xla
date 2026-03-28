# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos Transfer2.5-2B model loader for tt_forge_models.

Cosmos Transfer2.5 is a 2.36B-parameter diffusion transformer by NVIDIA for
controlled video generation (video-to-video). It takes control signals (Canny
edge, depth maps, segmentation masks, or blurred RGB) plus a text prompt and
generates 1280x720 video at 16 FPS.

Repository:
- https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B

Available subfolders:
- transformer: CosmosTransformer3DModel
- controlnet: CosmosControlNetModel
- vae: AutoencoderKLWan
"""

from typing import Any, Optional

import torch
from diffusers import AutoModel, Cosmos2_5_TransferPipeline

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

COSMOS_TRANSFER_REPO = "nvidia/Cosmos-Transfer2.5-2B"

SUPPORTED_SUBFOLDERS = {"transformer", "controlnet", "vae"}


class ModelVariant(StrEnum):
    """Available Cosmos Transfer2.5 variants."""

    EDGE = "edge"
    DEPTH = "depth"
    SEG = "seg"
    BLUR = "blur"


# Maps each variant to the HF revision for the controlnet and pipeline.
_VARIANT_REVISIONS = {
    ModelVariant.EDGE: {
        "controlnet": "diffusers/controlnet/general/edge",
        "pipeline": "diffusers/general",
    },
    ModelVariant.DEPTH: {
        "controlnet": "diffusers/controlnet/general/depth",
        "pipeline": "diffusers/general",
    },
    ModelVariant.SEG: {
        "controlnet": "diffusers/controlnet/general/seg",
        "pipeline": "diffusers/general",
    },
    ModelVariant.BLUR: {
        "controlnet": "diffusers/controlnet/general/blur",
        "pipeline": "diffusers/general",
    },
}


class ModelLoader(ForgeModel):
    """
    Loader for Cosmos Transfer2.5-2B controlled video generation model.

    Supports loading the full pipeline or individual components via subfolder:
    - 'transformer': CosmosTransformer3DModel
    - 'controlnet': CosmosControlNetModel
    - 'vae': AutoencoderKLWan

    Variants correspond to different control signal types:
    - EDGE: Canny edge control
    - DEPTH: Depth map control
    - SEG: Segmentation mask control
    - BLUR: Blurred RGB control
    """

    _VARIANTS = {
        ModelVariant.EDGE: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
        ModelVariant.DEPTH: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
        ModelVariant.SEG: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
        ModelVariant.BLUR: ModelConfig(
            pretrained_model_name=COSMOS_TRANSFER_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EDGE

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        subfolder: Optional[str] = None,
    ):
        super().__init__(variant)
        if subfolder is not None and subfolder not in SUPPORTED_SUBFOLDERS:
            raise ValueError(
                f"Unknown subfolder: {subfolder}. Supported: {SUPPORTED_SUBFOLDERS}"
            )
        self._subfolder = subfolder
        self.pipeline: Optional[Cosmos2_5_TransferPipeline] = None
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Cosmos-Transfer2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_revisions(self) -> dict:
        return _VARIANT_REVISIONS[self._variant]

    def _load_controlnet(self, dtype: torch.dtype):
        revisions = self._get_revisions()
        self._controlnet = AutoModel.from_pretrained(
            COSMOS_TRANSFER_REPO,
            revision=revisions["controlnet"],
            torch_dtype=dtype,
        )
        return self._controlnet

    def _load_pipeline(self, dtype: torch.dtype) -> Cosmos2_5_TransferPipeline:
        if self._controlnet is None:
            self._load_controlnet(dtype)
        revisions = self._get_revisions()
        self.pipeline = Cosmos2_5_TransferPipeline.from_pretrained(
            COSMOS_TRANSFER_REPO,
            controlnet=self._controlnet,
            revision=revisions["pipeline"],
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "controlnet":
            if self._controlnet is None:
                self._load_controlnet(dtype)
            return self._controlnet

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            return self.pipeline.vae
        elif self._subfolder == "transformer":
            return self.pipeline.transformer
        else:
            return self.pipeline.transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._subfolder == "controlnet":
            return self._load_controlnet_inputs(dtype)

        if self.pipeline is None:
            self._load_pipeline(dtype)

        if self._subfolder == "vae":
            vae_type = kwargs.get("vae_type", "decoder")
            if vae_type == "decoder":
                return self._load_vae_decoder_inputs(dtype)
            else:
                return self._load_vae_encoder_inputs(dtype)
        else:
            return self._load_transformer_inputs(dtype)

    def _load_transformer_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the Cosmos transformer forward pass."""
        batch_size = 1
        config = self.pipeline.transformer.config

        latent_num_frames = 2
        latent_height = 2
        latent_width = 2

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.text_embed_dim, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def _load_controlnet_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic inputs for the CosmosControlNetModel forward pass."""
        if self._controlnet is None:
            self._load_controlnet(dtype)

        config = self._controlnet.config
        batch_size = 1
        latent_num_frames = 2
        latent_height = 2
        latent_width = 2

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.text_embed_dim, dtype=dtype
        )

        controlnet_condition = torch.randn(
            batch_size,
            config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_condition": controlnet_condition,
            "timestep": timestep,
            "return_dict": False,
        }

    def _load_vae_decoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic decoder inputs for the Wan VAE."""
        latent_channels = self.pipeline.vae.config.latent_channels
        return {
            "sample": torch.randn(1, latent_channels, 2, 2, 2, dtype=dtype),
        }

    def _load_vae_encoder_inputs(self, dtype: torch.dtype) -> dict:
        """Prepare synthetic encoder inputs for the Wan VAE."""
        return {
            "sample": torch.randn(1, 3, 9, 64, 64, dtype=dtype),
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
