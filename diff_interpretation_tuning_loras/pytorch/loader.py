#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diff Interpretation Tuning (DIT) adapter model loader implementation.

Loads a base Qwen3 causal language model and applies a DIT adapter from
diff-interpretation-tuning/loras for interpreting weight differences
between language model checkpoints.

Available variants:
- QWEN3_4B: DIT adapter on Qwen/Qwen3-4B (default)
- QWEN3_8B: DIT adapter on Qwen/Qwen3-8B
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import (
    cast_input_to_type,
    pad_inputs,
)

ADAPTER_REPO = "diff-interpretation-tuning/loras"

_BASE_MODELS = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}

_ADAPTER_PATHS = {
    "qwen3_4b": "hidden-topic/qwen3-4b/dit-adapter.pt",
    "qwen3_8b": "hidden-topic/qwen3-8b/dit-adapter.pt",
}


class ModelVariant(StrEnum):
    """Available DIT adapter variants."""

    QWEN3_4B = "qwen3_4b"
    QWEN3_8B = "qwen3_8b"


class ModelLoader(ForgeModel):
    """Diff Interpretation Tuning LoRA model loader."""

    _VARIANTS = {
        ModelVariant.QWEN3_4B: ModelConfig(
            pretrained_model_name=_BASE_MODELS["qwen3_4b"],
        ),
        ModelVariant.QWEN3_8B: ModelConfig(
            pretrained_model_name=_BASE_MODELS["qwen3_8b"],
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN3_4B

    sample_text = "The weight difference between these two models indicates that"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DIFF_INTERPRETATION_TUNING_LORAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the base Qwen3 model with a DIT adapter applied.

        Returns:
            AutoModelForCausalLM with DIT adapter weights loaded.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Download and apply the DIT adapter
        variant_key = str(self._variant)
        adapter_path = hf_hub_download(
            repo_id=ADAPTER_REPO,
            filename=_ADAPTER_PATHS[variant_key],
        )
        adapter_weights = torch.load(
            adapter_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(adapter_weights, strict=False)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], 128)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], 128)

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs
