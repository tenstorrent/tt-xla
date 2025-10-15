# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Transfuser model loader implementation
"""
import torch
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Transfuser model variants for autonomous driving."""

    TRANSFUSER = "transfuser"


class ModelLoader(ForgeModel):
    """Transfuser model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TRANSFUSER

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="transfuser",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Transfuser model instance with default settings.

        Returns:
            Torch model: The Transfuser model instance.
        """
        # Load model with defaults
        from third_party.tt_forge_models.transfuser.pytorch.src.model import (
            LidarCenterNet,
        )

        model = LidarCenterNet()
        checkpoint_path = get_file(
            "test_files/pytorch/transfuser/transfuser_checkpoint.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the Transfuser model with default settings.

        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """

        rgb = torch.randint(0, 256, (1, 3, 160, 704)).to(dtype=torch.float32)
        lidar_bev = torch.rand((1, 2, 256, 256), dtype=torch.float32) * 0.2
        target_point = torch.tensor([[0.2033, -23.1296]], dtype=torch.float32)
        target_point_image = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        ego_vel = torch.tensor([[1.2922e-09]])

        kwargs = {
            "rgb": rgb,
            "lidar_bev": lidar_bev,
            "target_point": target_point,
            "target_point_image": target_point_image,
            "ego_vel": ego_vel,
        }

        return kwargs
