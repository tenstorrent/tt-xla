# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YuE-s2-1B-general model loader implementation
"""
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """YuE-s2-1B-general model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = "m-a-p/YuE-s2-1B-general"
        self.tokenizer = None

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
            model="YuE-s2-1B-general",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, **kwargs):
        """Load tokenizer for the model.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the YuE-s2-1B-general model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The YuE-s2-1B-general model instance.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LlamaForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YuE-s2-1B-general model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        prompt = (
            "[verse]\n"
            "Standing on the edge of tomorrow\n"
            "Chasing all the light through the sorrow\n"
        )

        tokenized_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        generation_config = GenerationConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
        )

        inputs = {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask,
            "generation_config": generation_config,
        }

        for key in inputs:
            if key == "generation_config":
                continue
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
