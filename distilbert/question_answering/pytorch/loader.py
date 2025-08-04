# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT model loader implementation for question answering.
"""

from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
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
    """Available DistilBERT model variants for question answering."""

    DISTILBERT_BASE_CASED_DISTILLED_SQUAD = "distilbert-base-cased-distilled-squad"


class ModelLoader(ForgeModel):
    """DistilBERT model loader implementation for question answering."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DISTILBERT_BASE_CASED_DISTILLED_SQUAD: LLMModelConfig(
            pretrained_model_name="distilbert-base-cased-distilled-squad",
            max_length=384,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DISTILBERT_BASE_CASED_DISTILLED_SQUAD

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = 384
        self.tokenizer = None

        # Sample data from SQuADv1.1
        self.context = """Super Bowl 50 was an American football game to determine the champion of the National Football League
        (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
        National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title.
        The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
        As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed
        initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
        (under which the game would have been known as "Super Bowl L"), so that the logo could prominently
        feature the Arabic numerals 50."""

        self.question = "Which NFL team represented the AFC at Super Bowl 50?"

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
            model="DistilBERT-QuestionAnswering",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load DistilBERT model for question answering from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DistilBERT model instance.
        """

        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DistilBertForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for DistilBERT question answering.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            # Ensure tokenizer is initialized
            self.load_model(dtype_override=dtype_override)

        # Data preprocessing
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
        """Decode the model output for question answering."""
        inputs = self.load_inputs()
        answer_start_index = co_out[0].argmax()
        answer_end_index = co_out[1].argmax()

        predict_answer_tokens = inputs.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]
        predicted_answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )
        print("Predicted answer:", predicted_answer)
