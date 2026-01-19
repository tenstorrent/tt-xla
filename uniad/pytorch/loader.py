# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UniAD model loader implementation
"""
import torch
from typing import Optional
from ...base import ForgeModel
from ...config import ModelGroup, ModelTask, ModelSource, Framework, StrEnum, ModelInfo
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available UniAD model variants for autonomous driving."""

    UNIAD_E2E = "uniad_e2e"


class ModelLoader(ForgeModel):
    """UniAD model loader implementation for autonomous driving tasks."""

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.UNIAD_E2E

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
            model="uniad",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the UniAD model instance with default settings.

        Returns:
            Torch model: The UniAD model instance.
        """
        # Load model with defaults
        from third_party.tt_forge_models.uniad.pytorch.src.uniad_e2e import UniAD

        model = UniAD()
        checkpoint_path = get_file("test_files/pytorch/uniad/uniad_e2e.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint,
            strict=False,
        )
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the UniAD model with default settings.

        Returns:
            dict: A dictionary of input tensors and metadata suitable for the model.
        """
        from third_party.tt_forge_models.uniad.pytorch.src.track_utils import (
            LiDARInstance3DBoxes,
        )

        img_tensor = img_tensor = [
            torch.randn(1, 6, 3, 928, 1600, dtype=torch.float32) * 50
        ]
        img_metas = [
            [
                {
                    "ori_shape": [(900, 1600, 3)] * 6,
                    "img_shape": [(928, 1600, 3)] * 6,
                    "lidar2img": [
                        torch.randn(4, 4, dtype=torch.float32) * 500 for _ in range(6)
                    ],
                    "box_type_3d": LiDARInstance3DBoxes,
                    "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
                    "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
                    "can_bus": torch.randn(18, dtype=torch.float32) * 300,
                }
            ]
        ]

        timestamp = [torch.tensor([1533151603.5476])]
        l2g_r_mat = torch.randn(1, 3, 3)
        l2g_t = torch.randn(1, 3) * 800

        gt_lane_labels = [torch.randint(0, 4, (1, 12))]
        gt_lane_bboxes = [torch.randint(0, 200, (1, 12, 4))]
        gt_segmentation = [torch.randint(0, 256, (1, 7, 200, 200))]
        gt_lane_masks = [torch.zeros((1, 12, 200, 200), dtype=torch.uint8)]

        gt_instance = [torch.randint(0, 318, (1, 7, 200, 200))]
        gt_centerness = [torch.randint(0, 256, (1, 7, 1, 200, 200))]
        gt_offset = [torch.randint(-5, 256, (1, 7, 2, 200, 200))]
        gt_flow = [torch.full((1, 7, 2, 200, 200), 255)]
        gt_backward_flow = [torch.full((1, 7, 2, 200, 200), 255)]
        gt_occ_has_invalid_frame = [torch.ones((1,), dtype=torch.long)]
        gt_occ_img_is_valid = [torch.randint(0, 2, (1, 9))]

        sdc_planning = [torch.rand(1, 1, 6, 3) * 25]
        sdc_planning_mask = [torch.ones(1, 1, 6, 2)]
        command = [torch.tensor([0])]

        kwargs = {
            "command": command,
            "gt_backward_flow": gt_backward_flow,
            "gt_centerness": gt_centerness,
            "gt_flow": gt_flow,
            "gt_instance": gt_instance,
            "gt_lane_bboxes": gt_lane_bboxes,
            "gt_lane_labels": gt_lane_labels,
            "gt_lane_masks": gt_lane_masks,
            "gt_occ_has_invalid_frame": gt_occ_has_invalid_frame,
            "gt_occ_img_is_valid": gt_occ_img_is_valid,
            "gt_offset": gt_offset,
            "gt_segmentation": gt_segmentation,
            "img": [img_tensor],
            "img_metas": img_metas,
            "l2g_r_mat": l2g_r_mat,
            "l2g_t": l2g_t,
            "rescale": True,
            "sdc_planning": sdc_planning,
            "sdc_planning_mask": sdc_planning_mask,
            "timestamp": timestamp,
        }

        return kwargs
