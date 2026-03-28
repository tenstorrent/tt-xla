# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NuExtract-tiny model loader implementation for causal language modeling.

NuExtract-tiny is a structured information extraction model fine-tuned from
Qwen/Qwen1.5-0.5B. It extracts structured JSON data from unstructured text
using a template-based prompt format.
"""
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available NuExtract model variants."""

    NUEXTRACT_TINY = "NuExtract_tiny"


class ModelLoader(ForgeModel):
    """NuExtract-tiny model loader implementation for structured extraction via causal LM."""

    _VARIANTS = {
        ModelVariant.NUEXTRACT_TINY: LLMModelConfig(
            pretrained_model_name="numind/NuExtract-tiny",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NUEXTRACT_TINY

    sample_text = (
        "We introduce NuExtract -- a foundation model for structured information extraction "
        "from textual sources, fine-tuned on a private high-quality synthetic dataset."
    )

    sample_template = json.dumps(
        {
            "Model": {"Name": "", "Based_on": ""},
            "Task": "",
        }
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Return model info for reporting."""
        return ModelInfo(
            model="NuExtract",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NuExtract model instance."""
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
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs using the NuExtract prompt format."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        schema = json.dumps(json.loads(self.sample_template), indent=4)
        input_llm = (
            "<|input|>\n### Template:\n"
            + schema
            + "\n### Text:\n"
            + self.sample_text
            + "\n<|output|>\n"
        )

        inputs = self.tokenizer(
            input_llm,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )
        return inputs
