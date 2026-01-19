# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vadv2 model loader implementation
"""
import torch
import numpy as np
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from ...tools.utils import get_file
from third_party.tt_forge_models.vadv2.pytorch.src.vad import VAD
from third_party.tt_forge_models.vadv2.pytorch.src.dataset import LiDARInstance3DBoxes


class ModelVariant(StrEnum):
    """Available Vadv2 model variants for autonomous driving."""

    VADV2_TINY = "vadv2_tiny"


class ModelLoader(ForgeModel):
    """Vadv2 model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VADV2_TINY

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
            model="vadv2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Vadv2 model instance with default settings.

        Returns:
            Torch model: The Vadv2 model instance.
        """
        # Load model with defaults
        model = VAD()
        checkpoint_path = get_file("test_files/pytorch/vadv2/vadv2_tiny.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint,
            strict=False,
        )
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the Vadv2 model with default settings.

        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        input_dict = {
            "img_metas": [
                [
                    [
                        {
                            "ori_shape": [(360, 640, 3)] * 6,
                            "img_shape": [(384, 640, 3)] * 6,
                            "lidar2img": [
                                torch.rand(4, 4) * 6,
                            ],
                            "pad_shape": [(384, 640, 3)] * 6,
                            "scale_factor": 1.0,
                            "flip": False,
                            "pcd_horizontal_flip": False,
                            "pcd_vertical_flip": False,
                            "box_type_3d": LiDARInstance3DBoxes,
                            "img_norm_cfg": {
                                "mean": np.array(
                                    [123.675, 116.28, 103.53], dtype=np.float32
                                ),
                                "std": np.array(
                                    [58.395, 57.12, 57.375], dtype=np.float32
                                ),
                                "to_rgb": True,
                            },
                            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                            "prev_idx": "",
                            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
                            "pcd_scale_factor": 1.0,
                            "pts_filename": "data/pcd.bin",
                            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                            "can_bus": torch.rand(18),
                        }
                    ]
                ]
            ],
            "gt_bboxes_3d": [
                [
                    [
                        LiDARInstance3DBoxes(
                            torch.rand(11, 9),
                            box_dim=9,
                        )
                    ]
                ]
            ],
            "gt_labels_3d": [[[torch.tensor([8, 8, 8, 8, 8, 8, 0, 8, 8, 0, 8])]]],
            "fut_valid_flag": [torch.tensor([True])],
            "ego_his_trajs": [[torch.tensor([[[[0.0757, 4.2529], [0.0757, 4.2529]]]])]],
            "ego_fut_trajs": [[torch.zeros((1, 1, 6, 2))]],
            "ego_fut_masks": [[torch.ones((1, 1, 6))]],
            "ego_fut_cmd": [[torch.tensor([[[[1.0, 0.0, 0.0]]]])]],
            "ego_lcf_feat": [[torch.zeros((1, 1, 1, 9))]],
            "gt_attr_labels": [[[torch.rand(11, 34)]]],
            "map_gt_labels_3d": [[torch.zeros((7,))]],
            "map_gt_bboxes_3d": [[None]],
        }
        tensor = torch.randn(1, 6, 3, 384, 640)
        img1 = []
        img1.append(tensor)
        kwargs = {
            "img": img1,
            "img_metas": input_dict["img_metas"],
            "gt_bboxes_3d": input_dict["gt_bboxes_3d"],
            "gt_labels_3d": input_dict["gt_labels_3d"],
            "fut_valid_flag": input_dict["fut_valid_flag"],
            "ego_his_trajs": input_dict["ego_his_trajs"],
            "ego_fut_trajs": input_dict["ego_fut_trajs"],
            "ego_fut_cmd": input_dict["ego_fut_cmd"],
            "ego_lcf_feat": input_dict["ego_lcf_feat"],
            "gt_attr_labels": input_dict["gt_attr_labels"],
        }

        return kwargs
