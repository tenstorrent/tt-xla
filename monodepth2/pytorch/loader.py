# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Monodepth2 model loader implementation
"""
import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from typing import Optional
from ...base import ForgeModel
from .src.utils import load_model, load_input


class ModelVariant(StrEnum):
    """Available Monodepth2 model variants."""

    MONO_640X192 = "mono_640x192"
    STEREO_640X192 = "stereo_640x192"
    MONO_STEREO_640X192 = "mono+stereo_640x192"
    MONO_NO_PT_640X192 = "mono_no_pt_640x192"
    STEREO_NO_PT_640X192 = "stereo_no_pt_640x192"
    MONO_STEREO_NO_PT_640X192 = "mono+stereo_no_pt_640x192"
    MONO_1024X320 = "mono_1024x320"
    STEREO_1024X320 = "stereo_1024x320"
    MONO_STEREO_1024X320 = "mono+stereo_1024x320"


class ModelLoader(ForgeModel):
    """Monodepth2 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MONO_640X192: ModelConfig(
            pretrained_model_name="mono_640x192",
        ),
        ModelVariant.STEREO_640X192: ModelConfig(
            pretrained_model_name="stereo_640x192",
        ),
        ModelVariant.MONO_STEREO_640X192: ModelConfig(
            pretrained_model_name="mono+stereo_640x192",
        ),
        ModelVariant.MONO_NO_PT_640X192: ModelConfig(
            pretrained_model_name="mono_no_pt_640x192",
        ),
        ModelVariant.STEREO_NO_PT_640X192: ModelConfig(
            pretrained_model_name="stereo_no_pt_640x192",
        ),
        ModelVariant.MONO_STEREO_NO_PT_640X192: ModelConfig(
            pretrained_model_name="mono+stereo_no_pt_640x192",
        ),
        ModelVariant.MONO_1024X320: ModelConfig(
            pretrained_model_name="mono_1024x320",
        ),
        ModelVariant.STEREO_1024X320: ModelConfig(
            pretrained_model_name="stereo_1024x320",
        ),
        ModelVariant.MONO_STEREO_1024X320: ModelConfig(
            pretrained_model_name="mono+stereo_1024x320",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MONO_640X192

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
            model="monodepth2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_DEPTH_EST,
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
        self._height = None
        self._width = None

    def load_model(self, dtype_override=None):
        """Load pretrained Monodepth2 model."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        model, height, width = load_model(pretrained_model_name)
        model.eval()

        self._height = height
        self._width = width

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Monodepth2 model"""

        inputs, original_width, original_height = load_input(self._height, self._width)

        self.original_width = original_width
        self.original_height = original_height

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def postprocess_and_save_disparity_map(self, co_out, save_path):
        disp = co_out[0].to(torch.float32)
        disp_resized = torch.nn.functional.interpolate(
            disp,
            (self.original_height, self.original_width),
            mode="bilinear",
            align_corners=False,
        )

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(
            np.uint8
        )
        im = pil.fromarray(colormapped_im)

        os.makedirs(save_path, exist_ok=True)
        name_dest_im = f"{save_path}/{self._variant_config.pretrained_model_name}_pred_disp_vis.png"
        im.save(name_dest_im)
