# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PUMA Model loader implementation for medical image segmentation.

Supports multiple segmentation architectures (UNet, UNet++, Attention, DeepLab,
MultiRes, Sharp) trained on nuclei and tissue segmentation tasks.
"""

import numpy as np
from dataclasses import dataclass
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


@dataclass
class PumaModelConfig(ModelConfig):
    """Configuration for PUMA model variants with specific checkpoint filenames."""

    filename: str = ""


class ModelVariant(StrEnum):
    """Available PUMA model variants."""

    UNET_NUCLEI = "UNet_Nuclei"
    UNETPP_NUCLEI = "UNetPP_Nuclei"
    ATTENTION_NUCLEI = "Attention_Nuclei"
    DEEPLAB_NUCLEI = "DeepLab_Nuclei"
    MULTIRES_NUCLEI = "MultiRes_Nuclei"
    SHARP_NUCLEI = "Sharp_Nuclei"


class ModelLoader(ForgeModel):
    """PUMA Model loader for medical image segmentation tasks."""

    _VARIANTS = {
        ModelVariant.UNET_NUCLEI: PumaModelConfig(
            pretrained_model_name="tanjina284/puma_model",
            filename="unet_nuclei_epoch225_datasetsize16000_fullmodel.keras",
        ),
        ModelVariant.UNETPP_NUCLEI: PumaModelConfig(
            pretrained_model_name="tanjina284/puma_model",
            filename="unetpp_nuclei_epoch230_datasetsize16000_fullmodel.keras",
        ),
        ModelVariant.ATTENTION_NUCLEI: PumaModelConfig(
            pretrained_model_name="tanjina284/puma_model",
            filename="attention_nuclei_epoch100_datasetsize16000_fullmodel.keras",
        ),
        ModelVariant.DEEPLAB_NUCLEI: PumaModelConfig(
            pretrained_model_name="tanjina284/puma_model",
            filename="deeplab_nuclei_epoch210_datasetsize16000_fullmodel.keras",
        ),
        ModelVariant.MULTIRES_NUCLEI: PumaModelConfig(
            pretrained_model_name="tanjina284/puma_model",
            filename="multires_nuclei_epoch185_datasetsize16000_fullmodel.keras",
        ),
        ModelVariant.SHARP_NUCLEI: PumaModelConfig(
            pretrained_model_name="tanjina284/puma_model",
            filename="sharp_nuclei_epoch240_datasetsize16000_fullmodel.keras",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNET_NUCLEI

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
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
            model="PUMA Model",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.KERAS,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PUMA segmentation model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            keras.Model: The loaded Keras segmentation model.
        """
        import keras

        config = self._variant_config
        model_path = hf_hub_download(
            repo_id=config.pretrained_model_name,
            filename=config.filename,
        )
        model = keras.models.load_model(model_path)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for the PUMA segmentation model.

        Args:
            dtype_override: Optional numpy dtype to override the input's default dtype.
            batch_size: Number of samples in the batch. Default is 1.

        Returns:
            numpy.ndarray: Sample input tensor of shape (batch_size, 256, 256, 3)
        """
        # Keras uses channels-last format (NHWC)
        inputs = np.random.randn(batch_size, 256, 256, 3).astype(np.float32)

        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs
