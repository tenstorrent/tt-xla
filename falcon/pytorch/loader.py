# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon model loader implementation for causal language modeling
"""
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Falcon model variants."""

    FALCON_1B = "tiiuae/Falcon3-1B-Base"
    FALCON_3B = "tiiuae/Falcon3-3B-Base"
    FALCON_7B = "tiiuae/Falcon3-7B-Base"
    FALCON_10B = "tiiuae/Falcon3-10B-Base"
    FALCON_MAMBA_7B = "tiiuae/Falcon3-Mamba-7B-Base"
    FALCON_7B_INSTRUCT = "tiiuae/falcon-7b-instruct"


class ModelLoader(ForgeModel):
    """Falcon model loader implementation for causal LM tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.FALCON_1B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-1B-Base",
        ),
        ModelVariant.FALCON_3B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-3B-Base",
        ),
        ModelVariant.FALCON_7B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-7B-Base",
        ),
        ModelVariant.FALCON_10B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-10B-Base",
        ),
        ModelVariant.FALCON_MAMBA_7B: ModelConfig(
            pretrained_model_name="tiiuae/Falcon3-Mamba-7B-Base",
        ),
        ModelVariant.FALCON_7B_INSTRUCT: ModelConfig(
            pretrained_model_name="tiiuae/falcon-7b-instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FALCON_1B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [
            ModelVariant.FALCON_1B,
            ModelVariant.FALCON_3B,
            ModelVariant.FALCON_7B,
            ModelVariant.FALCON_10B,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="falcon",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)

        # Configuration parameters
        self.input_text_1 = "Write a function to calculate the factorial of a number"
        self.max_length = 512
        self.tokenizer = None
        self.input_text_2 = "Hello, my dog is cute"

    def load_model(self, dtype_override=None):
        """Load and return the Falcon model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Falcon model instance for causal LM.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Falcon model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        if self._variant == ModelVariant.FALCON_7B_INSTRUCT:
            inputs = self.tokenizer(self.input_text_2, return_tensors="pt")
        else:
            inputs = self.tokenizer.encode(
                self.input_text_1,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
            )
        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        if inputs is None:
            inputs = self.load_inputs()

        response_start = torch.argmax(outputs.start_logits)
        response_end = torch.argmax(outputs.end_logits) + 1
        response_tokens = inputs.input_ids[0, response_start:response_end]

        return self.tokenizer.decode(response_tokens)

    def get_mesh_config(self, num_devices: int):
        # Single-device: always return data-parallel (1, 1)
        if num_devices == 1:
            return (1, 1), ("batch", "model")

        # Variant-specific mesh decisions for clarity and robustness
        if self._variant == ModelVariant.FALCON_MAMBA_7B:
            assert (
                num_devices % 2 == 0
            ), "Mamba requires an even number of devices for (2, N/2) mesh"
            mesh_shape = (2, num_devices // 2)
            return mesh_shape, ("batch", "model")

        # All other Falcon variants have attention heads in config
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        else:
            assert num_devices % 2 == 0, "Attention heads cannot be evenly distributed"
            mesh_shape = (2, num_devices // 2)

        shard_attention = self._variant in [
            ModelVariant.FALCON_7B,
            ModelVariant.FALCON_10B,
        ]
        if shard_attention:
            assert (
                self.config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        if self._variant in [ModelVariant.FALCON_1B, ModelVariant.FALCON_3B]:
            return None

        shard_specs = {}

        base = (
            getattr(model, "model", None)
            or getattr(model, "backbone", None)
            or getattr(model, "transformer", None)
        )
        if base is None:
            raise AttributeError(f"Unsupported model type: {type(model).__name__}")

        layers_container = getattr(base, "layers", None) or getattr(base, "h", None)
        if layers_container is None:
            raise AttributeError(
                f"Unsupported base container for {type(base).__name__}; expected `layers` or `h`."
            )

        if self._variant in [ModelVariant.FALCON_7B, ModelVariant.FALCON_10B]:
            for layer in layers_container:
                shard_specs[layer.mlp.up_proj.weight] = ("model", None)
                shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
                shard_specs[layer.mlp.down_proj.weight] = (None, "model")

                shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        elif self._variant == ModelVariant.FALCON_7B_INSTRUCT:
            for layer in layers_container:
                shard_specs[layer.mlp.dense_h_to_4h.weight] = ("model", None)
                shard_specs[layer.mlp.dense_4h_to_h.weight] = (None, "model")
        elif self._variant == ModelVariant.FALCON_MAMBA_7B:
            for layer in layers_container:
                shard_specs[layer.mixer.in_proj.weight] = ("model", None)
                shard_specs[layer.mixer.x_proj.weight] = ("model", None)
                shard_specs[layer.mixer.dt_proj.weight] = ("model", None)
                shard_specs[layer.mixer.out_proj.weight] = (None, "model")
        else:
            return None

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Falcon model.

        Returns:
            The configuration object for the Falcon model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
