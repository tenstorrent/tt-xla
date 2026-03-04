# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi 4 model loader implementation for causal language modeling
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Phi 4 model variants."""

    PHI_4 = "Phi_4"


class ModelLoader(ForgeModel):
    """Phi 4 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PHI_4: ModelConfig(
            pretrained_model_name="microsoft/phi-4",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI_4

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
        return ModelInfo(
            model="Phi-4",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Phi 4 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Phi 4 model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model_kwargs["use_cache"] = False
        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Phi 4 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            List: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Input prompt
        input_prompt = "Africa is an emerging economy because"

        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Return as list of tensors as expected by the test
        sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

        # Add batch dimension if needed
        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    # TODO - Verify this function correct (was AI_GENERATED)
    def decode_output(self, outputs, dtype_override=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass or generated token IDs
            dtype_override: Optional torch.dtype for tokenizer initialization

        Returns:
            str: Decoded output text
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Check if outputs are token IDs (from generation) or logits
        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            decoded_output = self.tokenizer.decode(outputs)
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.o_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.qkv_proj.weight] = ("model", "batch")

            shard_specs[layer.mlp.gate_up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("model", "batch")
        shard_specs[model.lm_head.weight] = ("model", "batch")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the model variant."""
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
