# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SHARP model loader implementation for image-to-3D view synthesis.

Loads the RGBGaussianPredictor from Apple's ml-sharp repository, which predicts
3D Gaussian splats from a single RGB image in a single feedforward pass.

Requires the ml-sharp repository to be cloned at /tmp/ml_sharp_repo.
"""
import os
import subprocess
import sys

import torch
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

ML_SHARP_REPO_PATH = "/tmp/ml_sharp_repo"


def _ensure_sharp_importable():
    """Ensure the ml-sharp repo is cloned and importable."""
    if not os.path.isdir(ML_SHARP_REPO_PATH):
        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/apple/ml-sharp.git",
                ML_SHARP_REPO_PATH,
            ]
        )

    src_path = os.path.join(ML_SHARP_REPO_PATH, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


class ModelVariant(StrEnum):
    """Available SHARP model variants."""

    RGB_GAUSSIAN_PREDICTOR = "RGB_Gaussian_Predictor"


class ModelLoader(ForgeModel):
    """SHARP model loader for the RGBGaussianPredictor."""

    _VARIANTS = {
        ModelVariant.RGB_GAUSSIAN_PREDICTOR: ModelConfig(
            pretrained_model_name="apple/Sharp",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RGB_GAUSSIAN_PREDICTOR

    _DEFAULT_MODEL_URL = (
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    )
    _INPUT_RESOLUTION = 1536

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SHARP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SHARP RGBGaussianPredictor.

        Returns:
            torch.nn.Module: The RGB Gaussian predictor model.
        """
        _ensure_sharp_importable()
        from sharp.models import PredictorParams, create_predictor

        model = create_predictor(PredictorParams())

        state_dict = torch.hub.load_state_dict_from_url(
            self._DEFAULT_MODEL_URL, progress=True
        )
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the RGBGaussianPredictor.

        Returns:
            dict: Input tensors (image, disparity_factor) for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # image: RGB input [B, 3, H, W] normalized to [0, 1]
        image = torch.rand(
            batch_size,
            3,
            self._INPUT_RESOLUTION,
            self._INPUT_RESOLUTION,
            dtype=dtype,
        )

        # disparity_factor: focal_length / image_width ratio [B]
        disparity_factor = torch.ones(batch_size, dtype=dtype)

        return {"image": image, "disparity_factor": disparity_factor}
