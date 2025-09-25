# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model loader implementation for sentence embedding generation.
"""

from transformers import BertModel, BertTokenizer
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
    """Available BERT model variants for sentence embedding generation."""

    EMRECAN_BERT_BASE_TURKISH_CASED_MEAN_NLI_STSB_TR = (
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    )


class ModelLoader(ForgeModel):
    """BERT model loader implementation for sentence embedding generation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.EMRECAN_BERT_BASE_TURKISH_CASED_MEAN_NLI_STSB_TR: LLMModelConfig(
            pretrained_model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            max_length=16,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.EMRECAN_BERT_BASE_TURKISH_CASED_MEAN_NLI_STSB_TR

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
        self.sentence = "Bu örnek bir cümle"
        self.max_length = 16
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
            model="BERT-SentenceEmbeddingGeneration",
            variant=variant_name,
            group=ModelGroup.RED,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load BERT model for sentence embedding generation from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT model instance.
        """

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = BertModel.from_pretrained(
            self.model_name, use_cache=False, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BERT sentence embedding generation.

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
            self.sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for sentence embedding generation."""
        import torch

        inputs = self.load_inputs()
        attention_mask = inputs["attention_mask"]

        # Mean pooling: mask out padding tokens and compute mean
        token_embeddings = co_out[0]  # Last hidden state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        print("Sentence embeddings:", sentence_embeddings)
