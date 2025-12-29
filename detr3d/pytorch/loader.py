# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Detr3d model loader implementation
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from third_party.tt_forge_models.detr3d.pytorch.src.detr import Detr3D
from third_party.tt_forge_models.detr3d.pytorch.src.dataset import LiDARInstance3DBoxes
from third_party.tt_forge_models.tools.utils import get_file, extract_tensors_recursive


class ModelVariant(StrEnum):
    """Available Detr3d model variants for autonomous driving."""

    DETR3D_RESNET101 = "detr3d_resnet101"


class ModelLoader(ForgeModel):
    """Detr3d model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DETR3D_RESNET101

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        # Configuration parameters
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
            model="detr3d",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Detr3d model instance with default settings.
        Returns:
            Torch model: The Detr3d model instance.
        """
        # Load model with defaults
        model = Detr3D()
        checkpoint_path = get_file("test_files/pytorch/detr3d/detr3d_resnet101.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint,
            strict=False,
        )
        model = model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the Detr3d model with default settings.
        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        input_dict = {
            "img_metas": [
                [
                    [
                        {
                            "img_shape": [(928, 1600, 3)] * 6,
                            "lidar2img": [torch.rand(4, 4) * 100 for _ in range(6)],
                            "box_type_3d": LiDARInstance3DBoxes,
                        }
                    ]
                ]
            ],
        }
        tensor = torch.randn(1, 6, 3, 928, 1600)
        img = []
        img.append(tensor)
        kwargs = {
            "img": img,
            "img_metas": input_dict["img_metas"],
        }

        return kwargs

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The DETR3D model returns bbox_list which is a list of dictionaries:
        [{"pts_bbox": {"boxes_3d": tensor, "scores_3d": tensor,
                       "labels_3d": tensor}}, ...]

        For training, we extract all tensor outputs and concatenate them
        to create a single differentiable tensor for backpropagation.

        Args:
            fwd_output: Output from the model's forward pass

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        tensors = []
        extract_tensors_recursive(fwd_output, tensors)

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
