# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Microsoft Multilingual-MiniLM model loader for sentence embedding generation.

Note: This model uses BertModel architecture with XLMRobertaTokenizer.
AutoTokenizer will not work with this checkpoint, so XLMRobertaTokenizer
must be loaded explicitly.
"""
import torch
from transformers import BertModel, XLMRobertaTokenizer
from typing import Optional

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
    """Available model variants for Multilingual-MiniLM."""

    MULTILINGUAL_MINILM_L12_H384 = "microsoft/Multilingual-MiniLM-L12-H384"


class ModelLoader(ForgeModel):
    """Microsoft Multilingual-MiniLM model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.MULTILINGUAL_MINILM_L12_H384: LLMModelConfig(
            pretrained_model_name="microsoft/Multilingual-MiniLM-L12-H384",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL_MINILM_L12_H384

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Multilingual-MiniLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load XLMRobertaTokenizer for the current variant.

        This model requires XLMRobertaTokenizer explicitly as AutoTokenizer
        will not work with this checkpoint.

        Returns:
            The loaded tokenizer instance
        """
        if self.tokenizer is None:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Multilingual-MiniLM model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            **kwargs: Additional model-specific arguments passed to from_pretrained.

        Returns:
            torch.nn.Module: The model instance.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            sentence: Optional sentence string. If None, uses a default sentence.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = "This is an example sentence for embedding generation."

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def output_postprocess(self, output, inputs=None):
        """Post-process model outputs to generate sentence embeddings.

        Args:
            output: Model output tensor, tuple, or BaseModelOutput.
            inputs: Optional input tensors. If None, will call load_inputs().

        Returns:
            torch.Tensor: Sentence embeddings computed using mean pooling.
        """
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        """Decode the model output for sentence embedding generation.

        Args:
            outputs: Model output tuple or BaseModelOutput.
            inputs: Optional input tensors. If None, will call load_inputs().

        Returns:
            torch.Tensor: Sentence embeddings computed using mean pooling.
        """
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        Args:
            fwd_output: Output from the model's forward pass

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
