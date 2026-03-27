# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EAGLE-LLaMA model loader implementation for causal language modeling.

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) is a
speculative decoding framework that accelerates LLM inference. This loader
handles the EAGLE draft model built on top of LLaMA 3 Instruct 8B.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available EAGLE-LLaMA model variants."""

    EAGLE_LLAMA3_INSTRUCT_8B = "EAGLE-LLaMA3-Instruct-8B"


class ModelLoader(ForgeModel):
    """EAGLE-LLaMA model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.EAGLE_LLAMA3_INSTRUCT_8B: LLMModelConfig(
            pretrained_model_name="yuhuili/EAGLE-LLaMA3-Instruct-8B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EAGLE_LLAMA3_INSTRUCT_8B

    sample_text = "Hey how are you doing today?"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="EAGLE-LLaMA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        if self.tokenizer is None:
            self.load_model()

        if hasattr(co_out, "logits"):
            output_ids = co_out.logits.argmax(-1)
        elif isinstance(co_out, (list, tuple)):
            output_ids = co_out[0].argmax(-1)
        else:
            output_ids = co_out.argmax(-1)

        decoded_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Input: {self.sample_text}")
        print(f"Output: {decoded_text}")
        return decoded_text
