# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/openpose/pytorch_lwopenpose_2d_osmr.py
"""
Openpose V2 model loader implementation
"""
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision import transforms
from ....tools.utils import get_file
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel


class ModelLoader(ForgeModel):
    """Openpose V2 model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "lwopenpose2d_mobilenet_cmupan_coco"

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
            model="openpose_v2",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Openpose V2 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Openpose V2 model instance.

        """
        model = ptcv_get_model(self.model_name, pretrained=True)
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Openpose V2 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        image_file = get_file(
            "https://raw.githubusercontent.com/axinc-ai/ailia-models/master/pose_estimation_3d/blazepose-fullbody/girl-5204299_640.jpg"
        )
        input_image = Image.open(str(image_file))
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(
            0
        )  # create a mini-batch as expected by the model

        batch_input = input_batch.repeat_interleave(batch_size, dim=0)
        if dtype_override is not None:
            batch_input = batch_input.to(dtype_override)

        return batch_input
