# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GenD CLIP L/14 model loader for deepfake detection (binary image classification).
"""
import importlib.util
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor
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
from PIL import Image


class ModelVariant(StrEnum):
    """Available GenD CLIP model variants."""

    LARGE_PATCH14 = "Large_Patch14"


class ModelLoader(ForgeModel):
    """GenD CLIP L/14 model loader for deepfake detection."""

    _VARIANTS = {
        ModelVariant.LARGE_PATCH14: ModelConfig(
            pretrained_model_name="yermandy/GenD_CLIP_L_14",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_PATCH14

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GenD_CLIP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _load_remote_module(repo_id, filename="modeling_gend.py"):
        """Download and import the custom modeling code from the HuggingFace repo.

        The GenD model uses a custom architecture not registered in transformers'
        auto_map, so we load the modeling code directly.
        """
        module_name = "modeling_gend"
        if module_name in sys.modules:
            return sys.modules[module_name]
        modeling_path = hf_hub_download(repo_id, filename)
        spec = importlib.util.spec_from_file_location(module_name, modeling_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the GenD CLIP model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The GenD model instance for deepfake detection.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        modeling_gend = self._load_remote_module(pretrained_model_name)
        config = modeling_gend.GenDConfig.from_pretrained(pretrained_model_name)
        model = modeling_gend.GenD(config)

        weights_path = hf_hub_download(pretrained_model_name, "model.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GenD CLIP model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size (default 1).

        Returns:
            dict: Input tensors containing pixel values.
        """
        if self.processor is None:
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        image = Image.new("RGB", (224, 224))

        inputs = self.processor(images=image, return_tensors="pt")

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def post_process(self, outputs):
        """Post-process model outputs to extract classification probabilities.

        Args:
            outputs: Raw model output containing logits.
        """
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]

        probs = logits.softmax(dim=-1)
        labels = ["fake", "real"]
        for i, label in enumerate(labels):
            print(f"Probability of '{label}': {probs[0, i].item():.4f}")

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass.

        Returns:
            torch.Tensor: Flattened output tensor for backward pass.
        """
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
