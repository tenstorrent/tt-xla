# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SantaCoder model loader implementation
"""


from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ...base import ForgeModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional


class ModelVariant(StrEnum):
    """Available SantaCoder model variants."""

    SANTACODER_1_1B = "1_1B"


class ModelLoader(ForgeModel):

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.SANTACODER_1_1B: LLMModelConfig(
            pretrained_model_name="bigcode/santacoder",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SANTACODER_1_1B

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = self._variant_config.pretrained_model_name
        self.tokenizer = None
        self.num_layers = num_layers

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
            model="SantaCoder",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SantaCoder model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The SantaCoder model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {"use_cache": False, "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SantaCoder model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors, pixel values and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(
                dtype_override=dtype_override
            )  # This will initialize the tokenizer

        text = "def hello_world():"
        inputs = self.tokenizer(text, return_tensors="pt")

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_outputs(self, outputs):
        """Decode the model outputs to text.

        Args:
            outputs: The model outputs to decode.

        Returns:
            str: The decoded text.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()

        # Handle both structured outputs and raw tensors
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Ensure logits are float type for softmax operation
        if not logits.dtype.is_floating_point:
            logits = logits.float()

        # Get logits for the last token in each batch
        next_token_logits = logits[:, -1]
        next_tokens = next_token_logits.softmax(dim=-1).argmax(dim=-1)

        if next_tokens.dim() == 0:
            # Single token case
            return [self.tokenizer.decode([next_tokens.item()])]
        else:
            # Batch of tokens case
            return [self.tokenizer.decode([token.item()]) for token in next_tokens]
