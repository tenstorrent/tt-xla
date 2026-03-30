# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
COVID QA DistilBERT model loader implementation for question answering.
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
    """Available COVID QA DistilBERT model variants for question answering."""

    COVID_QA_DISTILLBERT = "Covid_QA_DistillBert"


class ModelLoader(ForgeModel):
    """COVID QA DistilBERT model loader implementation for question answering."""

    _VARIANTS = {
        ModelVariant.COVID_QA_DISTILLBERT: LLMModelConfig(
            pretrained_model_name="shaina/covid_qa_distillBert",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COVID_QA_DISTILLBERT

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = 384
        self.tokenizer = None

        # Sample COVID-19 QA data
        self.context = (
            "COVID-19 is caused by the SARS-CoV-2 virus. Common symptoms include fever, "
            "cough, and shortness of breath. The virus primarily spreads through respiratory "
            "droplets produced when an infected person coughs, sneezes, or talks. Vaccines "
            "have been developed to help prevent severe illness from COVID-19."
        )
        self.question = "What causes COVID-19?"

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
            model="COVID_QA_DistilBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load COVID QA DistilBERT model for question answering from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The DistilBERT model instance.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = DistilBertForQuestionAnswering.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for COVID QA DistilBERT question answering.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
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
        print(f"Question: {self.question}")
        print(f"Predicted answer: {predicted_answer}")
