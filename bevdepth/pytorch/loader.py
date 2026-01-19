# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVDepth model loader implementation
"""
import torch
from typing import Optional
from ...base import ForgeModel
from ...config import (
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelInfo,
    ModelConfig,
)
from third_party.tt_forge_models.bevdepth.pytorch.src.model import (
    load_checkpoint,
    create_bevdepth_model,
)
from ...tools.utils import get_file, extract_tensors_recursive


class ModelVariant(StrEnum):
    """Available BEVDepth model variants."""

    BEVDEPTH_LSS_R50_256X704_128X128_24E_2KEY = (
        "bev_depth_lss_r50_256x704_128x128_24e_2key"
    )
    BEVDEPTH_LSS_R50_256X704_128X128_24E_2KEY_EMA = (
        "bev_depth_lss_r50_256x704_128x128_24e_2key_ema"
    )
    BEVDEPTH_LSS_R50_256X704_128X128_20E_CBGS_2KEY_DA = (
        "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da"
    )
    BEVDEPTH_LSS_R50_256X704_128X128_20E_CBGS_2KEY_DA_EMA = (
        "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema"
    )


class ModelLoader(ForgeModel):
    """BEVDepth model loader implementation for autonomous driving tasks."""

    _VARIANTS = {
        ModelVariant.BEVDEPTH_LSS_R50_256X704_128X128_24E_2KEY: ModelConfig(
            pretrained_model_name="bev_depth_lss_r50_256x704_128x128_24e_2key"
        ),
        ModelVariant.BEVDEPTH_LSS_R50_256X704_128X128_24E_2KEY_EMA: ModelConfig(
            pretrained_model_name="bev_depth_lss_r50_256x704_128x128_24e_2key_ema"
        ),
        ModelVariant.BEVDEPTH_LSS_R50_256X704_128X128_20E_CBGS_2KEY_DA: ModelConfig(
            pretrained_model_name="bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da"
        ),
        ModelVariant.BEVDEPTH_LSS_R50_256X704_128X128_20E_CBGS_2KEY_DA_EMA: ModelConfig(
            pretrained_model_name="bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema"
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BEVDEPTH_LSS_R50_256X704_128X128_24E_2KEY

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="bevdepth",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.BEVDEPTH_LSS_R50_256X704_128X128_24E_2KEY
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def _get_checkpoint_path(self, variant: str) -> str:
        """Get the checkpoint path for a specific variant."""
        # Map variants to their checkpoint files
        checkpoint_map = {
            "bev_depth_lss_r50_256x704_128x128_24e_2key": "test_files/pytorch/bevdepth/bev_depth_lss_r50_256x704_128x128_24e_2key.pth",
            "bev_depth_lss_r50_256x704_128x128_24e_2key_ema": "test_files/pytorch/bevdepth/bev_depth_lss_r50_256x704_128x128_24e_2key_ema.pth",
            "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da": "test_files/pytorch/bevdepth/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da.pth",
            "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema": "test_files/pytorch/bevdepth/bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.pth",
        }

        checkpoint_file = checkpoint_map[variant]
        return str(get_file(checkpoint_file))

    def load_model(self, **kwargs):
        """Load and return the BEVDepth model instance with variant-specific settings.
        Returns:
            Torch model: The BEVDepth model instance configured for the specified variant.
        """
        variant_str = str(self._variant) if self._variant else str(self.DEFAULT_VARIANT)

        model = create_bevdepth_model(variant_str, is_train_depth=False)
        # Load checkpoint
        checkpoint_path = self._get_checkpoint_path(variant_str)
        load_checkpoint(model, checkpoint_path)

        model.eval()
        return model

    def load_inputs(self, **kwargs):
        """Return sample inputs for the BEVDepth model with default settings."""
        sweep_imgs = torch.randn(1, 2, 6, 3, 256, 704)
        mats = {
            "sensor2ego_mats": torch.randn(1, 2, 6, 4, 4),
            "intrin_mats": torch.randn(1, 2, 6, 4, 4),
            "ida_mats": torch.randn(1, 2, 6, 4, 4),
            "bda_mat": torch.randn(1, 4, 4),
        }
        return [sweep_imgs, mats]

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The BEVDepth model returns a nested structure from multi_apply:
        tuple of lists of dicts, where each dict contains detection head outputs
        like 'heatmap', 'reg', 'height', 'dim', 'rot', 'vel', etc.

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
