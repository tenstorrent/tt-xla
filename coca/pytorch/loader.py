# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CoCa (Contrastive Captioners) model loader implementation for image captioning using OpenCLIP.
"""
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


class ModelVariant(StrEnum):
    """Available CoCa model variants."""

    VIT_L_14_MSCOCO = "ViT_L_14_mscoco"


class ModelLoader(ForgeModel):
    """CoCa model loader using OpenCLIP for image captioning tasks."""

    _VARIANTS = {
        ModelVariant.VIT_L_14_MSCOCO: ModelConfig(
            pretrained_model_name="coca_ViT-L-14",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIT_L_14_MSCOCO

    # OpenCLIP pretrained checkpoint name for each variant
    _PRETRAINED_TAG = {
        ModelVariant.VIT_L_14_MSCOCO: "mscoco_finetuned_laion2B-s13B-b90k",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="CoCa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_CAPT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CoCa model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The CoCa model instance.
        """
        import open_clip

        model, _, self.preprocess = open_clip.create_model_and_transforms(
            self._variant_config.pretrained_model_name,
            pretrained=self._PRETRAINED_TAG[self._variant],
        )
        self.tokenizer = open_clip.get_tokenizer(
            self._variant_config.pretrained_model_name
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the CoCa model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing the preprocessed image.
        """
        import open_clip

        if self.preprocess is None:
            _, _, self.preprocess = open_clip.create_model_and_transforms(
                self._variant_config.pretrained_model_name,
                pretrained=self._PRETRAINED_TAG[self._variant],
            )

        from datasets import load_dataset

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"]

        pixel_values = self.preprocess(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"image": pixel_values}

    def post_process(self, outputs):
        """Post-process CoCa model outputs to decode generated captions.

        Args:
            outputs: Raw model output containing generated token ids.
        """
        import open_clip

        if self.tokenizer is None:
            self.tokenizer = open_clip.get_tokenizer(
                self._variant_config.pretrained_model_name
            )

        generated = outputs
        if isinstance(generated, torch.Tensor):
            text = open_clip.decode(generated[0])
            print(f"Generated caption: {text}")

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass (tuple of tensors)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
