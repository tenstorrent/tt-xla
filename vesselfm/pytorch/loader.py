# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
vesselFM model loader implementation for 3D blood vessel segmentation.

vesselFM is a foundation model for universal zero-shot 3D blood vessel segmentation
based on MONAI's DynUNet architecture. It segments blood vessels across arbitrary
3D imaging domains.
"""
import torch
from huggingface_hub import hf_hub_download
from typing import Optional

from .src.model import DynUNet

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
    """Available vesselFM model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """vesselFM model loader for 3D blood vessel segmentation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="bwittmann/vesselFM",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    _WEIGHTS_FILENAME = "vesselFM_base.pt"

    # DynUNet architecture configuration from the vesselFM model card
    _KERNEL_SIZE = [[3, 3, 3]] * 6
    _STRIDES = [[1, 1, 1]] + [[2, 2, 2]] * 5
    _FILTERS = [32, 64, 128, 256, 320, 320]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="vesselFM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the vesselFM DynUNet model.

        Downloads pretrained weights from HuggingFace Hub and constructs
        a MONAI DynUNet model for 3D blood vessel segmentation.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The vesselFM DynUNet model instance.
        """
        repo_id = self._variant_config.pretrained_model_name

        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=self._KERNEL_SIZE,
            strides=self._STRIDES,
            upsample_kernel_size=self._STRIDES[1:],
            filters=self._FILTERS,
            res_block=True,
        )

        weights_path = hf_hub_download(repo_id=repo_id, filename=self._WEIGHTS_FILENAME)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample 3D volume input for vesselFM model.

        The model expects single-channel 3D volumes. Uses 128x128x128 patch size
        consistent with the model's sliding window inference configuration.

        Args:
            dtype_override: Optional torch.dtype to override input tensor dtype.
            batch_size: Batch size for the inputs.

        Returns:
            torch.Tensor: Input tensor of shape (batch_size, 1, 128, 128, 128).
        """
        dtype = dtype_override or torch.float32
        input_tensor = torch.randn(batch_size, 1, 128, 128, 128, dtype=dtype)
        return input_tensor
