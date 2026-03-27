# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TripoSR model loader implementation for single-image 3D reconstruction.

Loads the TSR (Triplane Scene Representation) model from the TripoSR pipeline,
which performs feed-forward 3D reconstruction from a single image using a
transformer backbone with triplane representation.

Requires the TripoSR repository to be cloned at /tmp/triposr_repo.
"""
import os
import sys

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

TRIPOSR_REPO_PATH = "/tmp/triposr_repo"


def _ensure_triposr_importable():
    """Ensure the TripoSR repo is cloned and importable."""
    if not os.path.isdir(TRIPOSR_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/VAST-AI-Research/TripoSR.git",
                TRIPOSR_REPO_PATH,
            ]
        )

    if TRIPOSR_REPO_PATH not in sys.path:
        sys.path.insert(0, TRIPOSR_REPO_PATH)


class ModelVariant(StrEnum):
    """Available TripoSR model variants."""

    V1 = "V1"


class ModelLoader(ForgeModel):
    """TripoSR model loader for single-image 3D reconstruction."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="stabilityai/TripoSR",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TripoSR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TripoSR TSR model.

        Returns:
            torch.nn.Module: The TSR model for single-image 3D reconstruction.
        """
        _ensure_triposr_importable()
        from tsr.system import TSR

        model = TSR.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the TripoSR model.

        The TSR model forward pass expects a list of PIL images and a device string.

        Returns:
            dict: Input dict with 'image' (list of PIL Images) and 'device' keys.
        """
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        images = [image] * batch_size

        return {"image": images, "device": "cpu"}
