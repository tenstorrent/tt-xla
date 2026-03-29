# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SapBERT model loader implementation for biomedical entity embedding generation.
"""
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
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
    """Available SapBERT model variants for biomedical entity embedding."""

    SAPBERT_FROM_PUBMEDBERT_FULLTEXT_MEAN_TOKEN = (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    )
    BIOSYN_SAPBERT_BC5CDR_DISEASE = "dmis-lab/biosyn-sapbert-bc5cdr-disease"


class ModelLoader(ForgeModel):
    """SapBERT model loader implementation for biomedical entity embedding generation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SAPBERT_FROM_PUBMEDBERT_FULLTEXT_MEAN_TOKEN: LLMModelConfig(
            pretrained_model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
            max_length=25,
        ),
        ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE: LLMModelConfig(
            pretrained_model_name="dmis-lab/biosyn-sapbert-bc5cdr-disease",
            max_length=25,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SAPBERT_FROM_PUBMEDBERT_FULLTEXT_MEAN_TOKEN

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.model = None
        self.tokenizer = None
        self.num_layers = num_layers

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

        variant_groups = {
            ModelVariant.SAPBERT_FROM_PUBMEDBERT_FULLTEXT_MEAN_TOKEN: ModelGroup.VULCAN,
            ModelVariant.BIOSYN_SAPBERT_BC5CDR_DISEASE: ModelGroup.VULCAN,
        }

        return ModelInfo(
            model="SapBERT",
            variant=variant,
            group=variant_groups.get(variant, ModelGroup.VULCAN),
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SapBERT model for biomedical entity embedding from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The SapBERT model instance.
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
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
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

        # Use provided sentence or default biomedical entity name
        if sentence is None:
            sentence = "covid-19"

        # Get max_length from parameter, config, or default
        if max_length is None:
            max_length = getattr(self._variant_config, "max_length", 25)

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
        """Post-process model outputs to generate entity embeddings using mean pooling.

        Args:
            output: Model output tensor, tuple, or BaseModelOutput.
            inputs: Optional input tensors. If None, will call load_inputs().

        Returns:
            torch.Tensor: Entity embeddings computed using mean pooling.
        """
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        # Extract token embeddings from outputs
        if isinstance(output, (tuple, list)):
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
        entity_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return entity_embeddings

    def decode_output(self, outputs, inputs=None):
        """Decode the model output for biomedical entity embedding generation.

        Args:
            outputs: Model output tuple (last_hidden_state, ...) or BaseModelOutput.
            inputs: Optional input tensors. If None, will call load_inputs().

        Returns:
            torch.Tensor: Entity embeddings computed using mean pooling.
        """
        return self.output_postprocess(outputs, inputs=inputs)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The SapBERT model returns a BaseModelOutputWithPoolingAndCrossAttentions
        containing last_hidden_state and pooler_output tensors.

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
