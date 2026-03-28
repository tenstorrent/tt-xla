# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
import torch

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
    get_static_cache_decode_inputs,
)
from .prefill_inputs import get_prefill_texts_for_batch, PREFILL_TEXTS


class ModelVariant(StrEnum):
    """Available Llama model variants for causal LM."""

    # Llama 3 variants
    LLAMA_3_8B = "3.0_8B"
    LLAMA_3_8B_INSTRUCT = "3.0_8B_Instruct"

    # Llama 3.1 variants
    LLAMA_3_1_8B = "3.1_8B"
    LLAMA_3_1_8B_INSTRUCT = "3.1_8B_Instruct"
    LLAMA_3_1_70B = "3.1_70B"
    LLAMA_3_1_70B_INSTRUCT = "3.1_70B_Instruct"
    LLAMA_3_1_405B = "3.1_405B"
    LLAMA_3_1_405B_INSTRUCT = "3.1_405B_Instruct"

    # Llama 3.2 variants
    LLAMA_3_2_1B = "3.2_1B"
    LLAMA_3_2_1B_INSTRUCT = "3.2_1B_Instruct"
    LLAMA_3_2_3B = "3.2_3B"
    LLAMA_3_2_3B_INSTRUCT = "3.2_3B_Instruct"

    # Llama 3.3 variants
    LLAMA_3_3_70B_INSTRUCT = "3.3_70B_Instruct"
    LLAMA_3_3_70B_INSTRUCT_AWQ = "3.3_70B_Instruct_Awq"

    # casperhansen AWQ quantized variants
    LLAMA_3_8B_INSTRUCT_AWQ = "3.0_8B_Instruct_Awq"

    # RedHatAI FP8 quantized variants
    LLAMA_3_2_1B_INSTRUCT_FP8 = "3.2_1B_Instruct_FP8"
    LLAMA_3_2_1B_INSTRUCT_FP8_DYNAMIC = "3.2_1B_Instruct_FP8_Dynamic"

    # hugging-quants AWQ INT4 quantized variants
    LLAMA_3_1_8B_INSTRUCT_AWQ_INT4 = "3.1_8B_Instruct_Awq_Int4"

    # HuggingFace community variants
    HUGGYLLAMA_7B = "Huggyllama_7B"

    # Llama 2 variants
    LLAMA_2_7B = "2_7B"

    # TinyLlama variants
    TINYLLAMA_V1_1 = "Tinyllama_v1.1"

    # JackFram variants
    JACKFRAM_LLAMA_160M = "JackFram_160M"

    # cazzz307 abliterated variants
    ABLITERATED_LLAMA_3_2_1B_INSTRUCT = "Abliterated_3.2_1B_Instruct"


class ModelLoader(ForgeModel):
    """Llama model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Llama 3 variants
        ModelVariant.LLAMA_3_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3-8B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_length=128,
        ),
        # Llama 3.1 variants
        ModelVariant.LLAMA_3_1_8B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_70B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-70B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_70B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_405B: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-405B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_1_405B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Meta-Llama-3.1-405B-Instruct",
            max_length=128,
        ),
        # Llama 3.2 variants
        ModelVariant.LLAMA_3_2_1B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-1B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_1B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-1B-Instruct",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_3B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-3B-Instruct",
            max_length=128,
        ),
        # RedHatAI FP8 quantized variants
        ModelVariant.LLAMA_3_2_1B_INSTRUCT_FP8: LLMModelConfig(
            pretrained_model_name="RedHatAI/Llama-3.2-1B-Instruct-FP8",
            max_length=128,
        ),
        ModelVariant.LLAMA_3_2_1B_INSTRUCT_FP8_DYNAMIC: LLMModelConfig(
            pretrained_model_name="RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic",
            max_length=128,
        ),
        # hugging-quants AWQ INT4 quantized variants
        ModelVariant.LLAMA_3_1_8B_INSTRUCT_AWQ_INT4: LLMModelConfig(
            pretrained_model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            max_length=128,
        ),
        # casperhansen AWQ quantized variants
        ModelVariant.LLAMA_3_8B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="casperhansen/llama-3-8b-instruct-awq",
            max_length=128,
        ),
        # Llama 3.3 variants
        ModelVariant.LLAMA_3_3_70B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-3.3-70B-Instruct",
            max_length=128,
        ),
        # Llama 2 variants
        ModelVariant.LLAMA_2_7B: LLMModelConfig(
            pretrained_model_name="meta-llama/Llama-2-7b-hf",
            max_length=128,
        ),
        # HuggingFace community variants
        ModelVariant.HUGGYLLAMA_7B: LLMModelConfig(
            pretrained_model_name="huggyllama/llama-7b",
            max_length=128,
        ),
        # Yahma variants
        ModelVariant.YAHMA_LLAMA_7B: LLMModelConfig(
            pretrained_model_name="yahma/llama-7b-hf",
            max_length=128,
        ),
        # TinyLlama variants
        ModelVariant.TINYLLAMA_V1_1: LLMModelConfig(
            pretrained_model_name="TinyLlama/TinyLlama_v1.1",
            max_length=128,
        ),
        # JackFram variants
        ModelVariant.JACKFRAM_LLAMA_160M: LLMModelConfig(
            pretrained_model_name="JackFram/llama-160m",
            max_length=128,
        ),
        # cazzz307 abliterated variants
        ModelVariant.ABLITERATED_LLAMA_3_2_1B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="cazzz307/Abliterated-Llama-3.2-1B-Instruct",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_1B_INSTRUCT

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

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
        self.seq_len = None
        self.config = None
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

        # Set group based on variant (instruct variants are RED priority except llama_3_8b_instruct and llama_3_1_405b_instruct variant)
        if variant in [
            ModelVariant.LLAMA_3_2_1B_INSTRUCT_FP8_DYNAMIC,
            ModelVariant.LLAMA_3_3_70B_INSTRUCT_AWQ,
            ModelVariant.LLAMA_3_8B_INSTRUCT_AWQ,
            ModelVariant.ABLITERATED_LLAMA_3_2_1B_INSTRUCT,
        ]:
            group = ModelGroup.VULCAN
        elif (
            (
                "instruct" in variant.value
                and (
                    variant
                    not in [
                        ModelVariant.LLAMA_3_8B_INSTRUCT,
                        ModelVariant.LLAMA_3_1_405B_INSTRUCT,
                    ]
                )
            )
            or "70B" in variant.value
            or variant == ModelVariant.LLAMA_3_1_405B
        ):
            group = ModelGroup.RED
        elif variant in [
            ModelVariant.LLAMA_3_2_1B,
            ModelVariant.LLAMA_3_2_3B,
            ModelVariant.LLAMA_3_1_8B_INSTRUCT,
        ]:
            group = ModelGroup.PRIORITY
        elif variant in [
            ModelVariant.LLAMA_2_7B,
            ModelVariant.LLAMA_3_2_1B_INSTRUCT_FP8_DYNAMIC,
            ModelVariant.JACKFRAM_LLAMA_160M,
        ]:
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="Llama",
            variant=variant,
            group=group,
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
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for Llama models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, num_layers=None, **kwargs):
        """Load and return the Llama model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        Returns:
            torch.nn.Module: The Llama model instance for causal LM.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        # Check if this is an AWQ variant and configure accordingly
        if pretrained_model_name in (
            "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            "casperhansen/llama-3-8b-instruct-awq",
        ):
            model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if num_layers is not None:
            model.model.layers = model.model.layers[:num_layers]

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs (single token + static KV cache).
        Attention mask is intentionally omitted for single-batch decode. Defaults to steady-state decode.
        """

        # Ensure tokenizer and config are initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        max_cache_len = self._variant_config.max_length
        self.seq_len = 1

        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            model=self.model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )

    def load_inputs_prefill(self, dtype_override=None, batch_size=1, seq_len=128):
        """Load prefill-step inputs with texts sized appropriately for the target sequence length.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
            batch_size: Batch size for the inputs.
            seq_len: Target sequence length. Texts are chosen to minimize padding.

        Returns:
            dict: Input tensors (input_ids, attention_mask) padded to seq_len.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get appropriate texts for this seq_len and batch_size
        if seq_len not in PREFILL_TEXTS:
            available = sorted(PREFILL_TEXTS.keys())
            raise ValueError(
                f"seq_len={seq_len} is not supported. Available sequence lengths: {available}"
            )
        texts = get_prefill_texts_for_batch(seq_len, batch_size)

        # Tokenize all texts in the batch with padding to exact seq_len
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        self.seq_len = seq_len
        return inputs

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        """Generates text .
        Args:
            max_new_tokens (int): The maximum number of new tokens to generate.
            model (torch.nn.Module): The language model used for token generation.
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
            tokenizer: The tokenizer used to decode token IDs into text.
        """
        current_pos = self.seq_len

        for _ in range(max_new_tokens):
            logits = model(*inputs)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            next_token_logits = logits[:, current_pos - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Update input_ids and attention_mask
            inputs[0][:, current_pos] = next_token_id
            inputs[1][:, current_pos] = 1

            current_pos += 1

        valid_tokens = inputs[0][:, self.seq_len : current_pos].view(-1).tolist()
        answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        return answer

    def get_mesh_config(self, num_devices: int):
        if self._variant in [
            ModelVariant.LLAMA_3_1_70B,
            ModelVariant.LLAMA_3_1_70B_INSTRUCT,
            ModelVariant.LLAMA_3_3_70B_INSTRUCT,
            ModelVariant.LLAMA_3_3_70B_INSTRUCT_AWQ,
            ModelVariant.LLAMA_3_1_405B,
            ModelVariant.LLAMA_3_1_405B_INSTRUCT,
        ]:
            if num_devices == 32:  # Galaxy
                mesh_shape = (4, 8)
            else:  # wh/bh llmbox
                mesh_shape = (2, num_devices // 2)
        else:
            mesh_shape = (1, num_devices)

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
        """Load weight shard specifications for tensor parallelism.

        Args:
            model: The model whose weights are to be sharded.
            strategy: Sharding strategy — "fsdp" shards across both axes,
                      "megatron" shards on "model" axis only (other axis is None).
            batch_axis: Name of the non-model mesh axis for "fsdp" specs (ignored
                        by "megatron"). Defaults to "batch" for a ("batch", "model")
                        mesh; pass "data" when input sharding is enabled, because
                        load_shard_spec_data_parallel hardcodes "data" as the input
                        sharding axis, forcing the mesh to ("data", "model").

        Returns:
            dict mapping weight tensors to shard spec tuples, or None for small models.
        """
        if self._variant in [
            ModelVariant.LLAMA_3_2_1B,
            ModelVariant.LLAMA_3_2_1B_INSTRUCT,
            ModelVariant.LLAMA_3_2_3B,
            ModelVariant.LLAMA_3_2_3B_INSTRUCT,
            ModelVariant.LLAMA_3_2_1B_INSTRUCT_FP8,
            ModelVariant.LLAMA_3_2_1B_INSTRUCT_FP8_DYNAMIC,
            ModelVariant.HUGGYLLAMA_7B,
            ModelVariant.LLAMA_2_7B,
            ModelVariant.JACKFRAM_LLAMA_160M,
        ]:
            return None

        shard_specs = {}

        if strategy == "fsdp":
            # FSDP: weights sharded across both batch_axis and "model" mesh axes.
            shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
            shard_specs[model.lm_head.weight] = ("model", batch_axis)
            shard_specs[model.model.norm.weight] = (batch_axis,)
            for layer in model.model.layers:
                shard_specs[layer.mlp.up_proj.weight] = ("model", batch_axis)
                shard_specs[layer.mlp.gate_proj.weight] = ("model", batch_axis)
                shard_specs[layer.mlp.down_proj.weight] = (batch_axis, "model")

                shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
                shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
                shard_specs[layer.input_layernorm.weight] = (batch_axis,)
                shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)

        elif strategy == "megatron":
            # Megatron: weights sharded on "model" axis, replicated (None) on the other.
            shard_specs[model.model.embed_tokens.weight] = (None, None)
            shard_specs[model.lm_head.weight] = ("model", None)
            shard_specs[model.model.norm.weight] = (None,)
            for layer in model.model.layers:
                shard_specs[layer.mlp.up_proj.weight] = ("model", None)
                shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
                shard_specs[layer.mlp.down_proj.weight] = (None, "model")

                shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
                shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
                shard_specs[layer.input_layernorm.weight] = (None,)
                shard_specs[layer.post_attention_layernorm.weight] = (None,)

        else:
            raise ValueError(f"Unknown sharding strategy: {strategy!r}")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Llama model variant.

        Returns:
            The configuration object for the Llama model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
