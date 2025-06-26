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
)
from ...base import ForgeModel
from .src.utils import load_model, load_input


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="monodepth2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    """Loads Monodepth2 model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "mono_640x192"
        self._height = None
        self._width = None

    def load_model(self, dtype_override=None):
        """Load pretrained Monodepth2 model."""
        model, height, width = load_model(self.model_name)
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
        name_dest_im = f"{save_path}/{self.model_name}_pred_disp_vis.png"
        im.save(name_dest_im)
