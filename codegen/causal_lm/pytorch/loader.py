# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Codegen model loader implementation
"""


from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelVariant(StrEnum):
    """Available Codegen model variants."""

    CODEGEN_350M_MONO = "Salesforce/codegen-350M-mono"
    CODEGEN_350M_MULTI = "Salesforce/codegen-350M-multi"
    CODEGEN_350M_NL = "Salesforce/codegen-350M-nl"


class ModelLoader(ForgeModel):

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.CODEGEN_350M_MONO: LLMModelConfig(
            pretrained_model_name="Salesforce/codegen-350M-mono",
            max_length=256,
        ),
        ModelVariant.CODEGEN_350M_MULTI: LLMModelConfig(
            pretrained_model_name="Salesforce/codegen-350M-multi",
            max_length=256,
        ),
        ModelVariant.CODEGEN_350M_NL: LLMModelConfig(
            pretrained_model_name="Salesforce/codegen-350M-nl",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CODEGEN_350M_MONO

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = self._variant_config.pretrained_model_name
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
            model="codegen",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Codegen model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Codegen model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, use_cache=False, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Codegen model with default settings.

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
