# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral model loader implementation for causal language modeling
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Mistral model variants."""

    MISTRAL_7B = "7b"
    MISTRAL_7B_INSTRUCT_V03 = "7b_instruct_v03"
    MINISTRAL_3B = "ministral_3b_instruct"
    MINISTRAL_8B = "ministral_8b_instruct"
    MISTRAL_SMALL_24B_INSTRUCT_2501 = "mistral_small_24b_instruct_2501"
    MISTRAL_LARGE_INSTRUCT_2411 = "mistral_large_instruct_2411"
    MISTRAL_NEMO_INSTRUCT_2407 = "mistral_nemo_instruct_2407"
    DEVSTRAL_SMALL_2505 = "devstral_small_2505"
    MAGISTRAL_SMALL_2506 = "magistral_small_2506"


class ModelLoader(ForgeModel):
    """Mistral model loader implementation for causal language modeling tasks."""

    # These models do NOT ship a HF tokenizer. Instead they use tekken.json,
    # which must be loaded via mistral-common, can't use AutoTokenizer.
    _TEKKEN_TOKENIZER_VARIANTS = {
        ModelVariant.DEVSTRAL_SMALL_2505,
        ModelVariant.MAGISTRAL_SMALL_2506,
    }

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MISTRAL_7B: ModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-v0.1",
        ),
        ModelVariant.MISTRAL_7B_INSTRUCT_V03: ModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        ),
        ModelVariant.MINISTRAL_3B: ModelConfig(
            pretrained_model_name="ministral/Ministral-3b-instruct",
        ),
        ModelVariant.MINISTRAL_8B: ModelConfig(
            pretrained_model_name="mistralai/Ministral-8B-Instruct-2410",
        ),
        ModelVariant.MISTRAL_SMALL_24B_INSTRUCT_2501: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Small-24B-Instruct-2501",
        ),
        ModelVariant.MISTRAL_LARGE_INSTRUCT_2411: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Large-Instruct-2411",
        ),
        ModelVariant.MISTRAL_NEMO_INSTRUCT_2407: ModelConfig(
            pretrained_model_name="mistralai/Mistral-Nemo-Instruct-2407",
        ),
        ModelVariant.DEVSTRAL_SMALL_2505: ModelConfig(
            pretrained_model_name="mistralai/Devstral-Small-2505",
        ),
        ModelVariant.MAGISTRAL_SMALL_2506: ModelConfig(
            pretrained_model_name="mistralai/Magistral-Small-2506",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MISTRAL_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant in [
            ModelVariant.MISTRAL_7B_INSTRUCT_V03,
            ModelVariant.MINISTRAL_8B,
            ModelVariant.MISTRAL_SMALL_24B_INSTRUCT_2501,
            ModelVariant.MISTRAL_LARGE_INSTRUCT_2411,
            ModelVariant.MISTRAL_NEMO_INSTRUCT_2407,
            ModelVariant.DEVSTRAL_SMALL_2505,
            ModelVariant.MAGISTRAL_SMALL_2506,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="mistral",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """

        if self._variant in self._TEKKEN_TOKENIZER_VARIANTS:

            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from huggingface_hub import hf_hub_download

            tokenizer_json = hf_hub_download(
                repo_id=self._variant_config.pretrained_model_name,
                filename="tekken.json",
            )
            self.tokenizer = MistralTokenizer.from_file(tokenizer_json)
            return self.tokenizer

        tokenizer_kwargs = {
            "padding_side": "right",
        }
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Mistral model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The Mistral model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Load pre-trained model from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Mistral model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        if self.model is None:
            self.load_model(dtype_override)

        # Set up sample input
        test_input = "How often does the letter r occur in Mistral?"

        # Tokenize input using either Tekken or regular HF tokenizer.
        if self._variant in self._TEKKEN_TOKENIZER_VARIANTS:

            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest

            req = ChatCompletionRequest(
                messages=[UserMessage(content=test_input)],
            )
            encoded = self.tokenizer.encode_chat_completion(req)
            token_ids = encoded.tokens

            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        else:
            inputs = self.tokenizer.encode_plus(test_input, return_tensors="pt")

        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            # if the model uses sliding window attention, match sliding window value to input size so it
            # does not go out of bounds when updating the cache
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    # TODO - Verify this function correct (was AI_GENERATED)
    def decode_output(self, outputs, dtype_override):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded next token text
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Get logits for the last token
        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        if self._variant not in [
            ModelVariant.MINISTRAL_3B,
        ]:
            assert (
                self.config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        if self._variant in [
            ModelVariant.MINISTRAL_3B,
        ]:
            return None

        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Mistral model variant.

        Returns:
            The configuration object for the Mistral model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
