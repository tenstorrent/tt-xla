# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-2 model loader implementations for text generation and sequence classification.
"""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)
from typing import Optional

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
    """Available GPT-2 model variants."""

    GPT2_BASE = "Default"
    GPT2_LARGE = "Large"
    GPT2_SMALL_DUTCH = "Small_Dutch"
    GPT2_SEQUENCE_CLASSIFICATION = "Sequence_Classification"


class ModelLoader(ForgeModel):
    """GPT-2 loader for causal language modeling and sequence classification."""

    _VARIANTS = {
        ModelVariant.GPT2_BASE: LLMModelConfig(
            pretrained_model_name="gpt2",
            max_length=256,
        ),
        ModelVariant.GPT2_LARGE: LLMModelConfig(
            pretrained_model_name="openai-community/gpt2-large",
            max_length=256,
        ),
        ModelVariant.GPT2_SMALL_DUTCH: LLMModelConfig(
            pretrained_model_name="GroNLP/gpt2-small-dutch",
            max_length=256,
        ),
        ModelVariant.GPT2_SEQUENCE_CLASSIFICATION: LLMModelConfig(
            pretrained_model_name="mnoukhov/gpt2-imdb-sentiment-classifier",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT2_BASE

    sample_text = "This is a sample text from "

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant == ModelVariant.GPT2_SEQUENCE_CLASSIFICATION:
            task = ModelTask.NLP_TEXT_CLS
        else:
            task = ModelTask.NLP_CAUSAL_LM

        group = (
            ModelGroup.VULCAN
            if variant in (ModelVariant.GPT2_LARGE, ModelVariant.GPT2_SMALL_DUTCH)
            else ModelGroup.GENERALITY
        )

        return ModelInfo(
            model="GPT-2",
            variant=variant,
            group=group,
            task=task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        # Set padding side to left for classification variants
        if self._variant == ModelVariant.GPT2_SEQUENCE_CLASSIFICATION:
            self.tokenizer.padding_side = "left"

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        if self._variant in (
            ModelVariant.GPT2_BASE,
            ModelVariant.GPT2_LARGE,
            ModelVariant.GPT2_SMALL_DUTCH,
        ):
            config = GPT2Config.from_pretrained(model_name)
            config_dict = config.to_dict()
            config_dict["use_cache"] = True
            if dtype_override is not None:
                config_dict["torch_dtype"] = dtype_override
            if self.num_layers is not None:
                config_dict["num_hidden_layers"] = self.num_layers
            config = GPT2Config(**config_dict)
            model = GPT2LMHeadModel.from_pretrained(model_name, config=config, **kwargs)
        else:
            model_kwargs = {
                "trust_remote_code": True,
                "use_cache": False,
            }
            if dtype_override is not None:
                model_kwargs["torch_dtype"] = dtype_override
            model_kwargs |= kwargs
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **model_kwargs
            )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if self._variant in (
            ModelVariant.GPT2_BASE,
            ModelVariant.GPT2_LARGE,
            ModelVariant.GPT2_SMALL_DUTCH,
        ):
            # Use random input for text generation
            vocab_size = GPT2Config.from_pretrained(
                self._variant_config.pretrained_model_name
            ).vocab_size

            input_ids = torch.cat(
                [
                    torch.randint(1, vocab_size, (1, 255)),
                    torch.zeros(1, 1, dtype=torch.int64),
                ],
                dim=-1,
            ).to(torch.int64)

            return {"input_ids": input_ids}

        else:
            test_input = self.sample_text
            tokenized = self.tokenizer(
                test_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._variant_config.max_length,
            )
            return {"input_ids": tokenized["input_ids"]}

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        if self._variant == ModelVariant.GPT2_SEQUENCE_CLASSIFICATION:
            # For classification: map class index to label
            predicted_value = logits.argmax(-1).item()
            model = self.load_model()
            return model.config.id2label[predicted_value]
        else:
            # For generation: decode tokens
            generated_ids = logits.argmax(-1)
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
