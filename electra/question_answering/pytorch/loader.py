# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ELECTRA model loader implementation for question answering task.
"""

from transformers import ElectraForQuestionAnswering, ElectraTokenizerFast
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available ELECTRA question answering model variants."""

    LARGE_DISCRIMINATOR_SQUAD2_512 = "Large_Discriminator_Squad2_512"
    GELECTRA_LARGE_GERMANQUAD = "GElectra_Large_GermanQuAD"


class ModelLoader(ForgeModel):
    """ELECTRA model loader implementation for question answering task."""

    _VARIANTS = {
        ModelVariant.LARGE_DISCRIMINATOR_SQUAD2_512: LLMModelConfig(
            pretrained_model_name="ahotrod/electra_large_discriminator_squad2_512",
            max_length=512,
        ),
        ModelVariant.GELECTRA_LARGE_GERMANQUAD: LLMModelConfig(
            pretrained_model_name="deepset/gelectra-large-germanquad",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_DISCRIMINATOR_SQUAD2_512

    _SAMPLE_DATA = {
        ModelVariant.LARGE_DISCRIMINATOR_SQUAD2_512: {
            "context": (
                "Super Bowl 50 was an American football game to determine the champion of the National Football League "
                "(NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the "
                "National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. "
                "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
            ),
            "question": "Which NFL team represented the AFC at Super Bowl 50?",
        },
        ModelVariant.GELECTRA_LARGE_GERMANQUAD: {
            "context": "Mein Name ist Wolfgang und ich lebe in Berlin.",
            "question": "Wo wohne ich?",
        },
    }

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

        sample = self._SAMPLE_DATA[self._variant]
        self.context = sample["context"]
        self.question = sample["question"]

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ELECTRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ElectraForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        start_logits = co_out[0]
        end_logits = co_out[1]

        answer_start_index = start_logits.argmax()
        answer_end_index = end_logits.argmax()

        input_ids = inputs["input_ids"]
        predict_answer_tokens = input_ids[0, answer_start_index : answer_end_index + 1]

        predicted_answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )
        print("Predicted answer:", predicted_answer)
