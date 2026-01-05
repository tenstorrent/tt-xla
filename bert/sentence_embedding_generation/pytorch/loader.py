# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model loader implementation for sentence embedding generation.
"""
import torch
from transformers import BertModel, BertTokenizer
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
            model="bert_sentence_embedding",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load BERT model for sentence embedding generation from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT model instance.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = BertModel.from_pretrained(model_name, use_cache=False, **model_kwargs)
        model.eval()

        # Store model for potential use in decode_output
        self.model = model

        return model

    def input_preprocess(self, dtype_override=None, sentence=None, max_length=None):
        """Preprocess input sentence(s) and return model-ready input tensors.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            sentence: Optional sentence string or list of sentences. If None, uses a default sentence.
            max_length: Optional maximum sequence length. If None, uses config value.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Use provided sentence or default
        if sentence is None:
            sentence = "Bu örnek bir cümle"

        # Get max_length from parameter, config, or default
        if max_length is None:
            max_length = getattr(self._variant_config, "max_length", 128)

        # Data preprocessing
        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def load_inputs(self, dtype_override=None, sentence=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            sentence: Optional sentence string. If None, uses a default sentence.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        return self.input_preprocess(
            dtype_override=dtype_override,
            sentence=sentence,
        )

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

        # Extract token embeddings from outputs
        if isinstance(output, tuple):
            token_embeddings = output[0]  # Last hidden state
        elif hasattr(output, "last_hidden_state"):
            # Handle BaseModelOutput or similar
            token_embeddings = output.last_hidden_state
        else:
            # Assume output is already the last hidden state tensor
            token_embeddings = output

        # Mean pooling: mask out padding tokens and compute mean
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
            outputs: Model output tuple (last_hidden_state, ...) or BaseModelOutput.
            inputs: Optional input tensors. If None, will call load_inputs().

        Returns:
            torch.Tensor: Sentence embeddings computed using mean pooling.
        """
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The BERT model returns a BaseModelOutputWithPoolingAndCrossAttentions
        containing last_hidden_state and pooler_output tensors.

        For training, we extract the main output tensors and concatenate them
        to create a single differentiable tensor for backpropagation.

        Args:
            fwd_output: Output from the model's forward pass

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        tensors = []

        # Handle HuggingFace model output objects
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
