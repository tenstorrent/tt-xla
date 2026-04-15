# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLM-4-9B-Chat Abliterated GGUF model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


def _patch_transformers_chatglm_gguf():
    """Monkey-patch transformers to add chatglm GGUF architecture support.

    The GLM-4-9B model uses the 'chatglm' architecture identifier in its GGUF
    metadata. Transformers 5.x has Glm4ForCausalLM but lacks GGUF loading
    support for the chatglm architecture. We bridge the gap by registering
    the config mapping and remapping model_type to glm4.
    """
    from transformers.modeling_gguf_pytorch_utils import (
        GGUF_SUPPORTED_ARCHITECTURES,
        GGUF_TO_TRANSFORMERS_MAPPING,
    )
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    if "chatglm" in GGUF_SUPPORTED_ARCHITECTURES:
        return  # Already patched

    # 1. Register chatglm as a supported architecture
    GGUF_SUPPORTED_ARCHITECTURES.append("chatglm")

    # 2. Add config mapping for chatglm -> glm4 config fields
    GGUF_TO_TRANSFORMERS_MAPPING["config"]["chatglm"] = {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "vocab_size": "vocab_size",
    }

    # 3. Register chatglm tokenizer converter with merge-fixing logic.
    # ChatGLM vocabularies contain tokens with literal spaces, so the
    # space-delimited GGUF merge strings sometimes split into 3 parts
    # instead of the expected 2. We resolve the ambiguity by checking
    # which split produces tokens that exist in the vocabulary.
    from transformers.integrations.ggml import (
        GGUF_TO_FAST_CONVERTERS,
        GGUFQwen2Converter,
        GGUFTokenizerSkeleton,
    )

    class GGUFChatGLMConverter(GGUFQwen2Converter):
        def converted(self):
            vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
            raw_merges = self.original_tokenizer.merges
            clean_merges = []
            for m in raw_merges:
                if len(m) == 2:
                    clean_merges.append(m)
                elif len(m) == 3:
                    # Try both possible 2-way splits
                    left = m[0] + " " + m[1]
                    if left in vocab:
                        clean_merges.append((left, m[2]))
                    elif m[0] in vocab and (m[1] + " " + m[2]) in vocab:
                        clean_merges.append((m[0], m[1] + " " + m[2]))
                    # else: skip unresolvable merge
            merges = clean_merges

            from transformers.convert_slow_tokenizer import Qwen2Converter

            tokenizer = Qwen2Converter.converted(self, vocab, merges)
            from tokenizers import AddedToken

            tokenizer.add_special_tokens(
                [
                    AddedToken("<|endoftext|>", normalized=False, special=True),
                    AddedToken("<|im_start|>", normalized=False, special=True),
                    AddedToken("<|im_end|>", normalized=False, special=True),
                ]
            )
            return tokenizer

    if "chatglm" not in GGUF_TO_FAST_CONVERTERS:
        GGUF_TO_FAST_CONVERTERS["chatglm"] = GGUFChatGLMConverter

    # 4. Patch load_gguf_checkpoint to remap model_type and compute partial_rotary_factor
    orig_load = gguf_utils.load_gguf_checkpoint

    def patched_load_gguf_checkpoint(*args, **kwargs):
        result = orig_load(*args, **kwargs)
        config = result.get("config", {})
        if config.get("model_type") == "chatglm":
            config["model_type"] = "glm4"
            head_dim = config.get("head_dim", 128)
            if hasattr(args[0] if args else None, "__fspath__") or isinstance(
                args[0] if args else None, str
            ):
                try:
                    from gguf import GGUFReader
                    from transformers.modeling_gguf_pytorch_utils import (
                        _gguf_parse_value,
                    )

                    reader = GGUFReader(args[0])
                    for key, field in reader.fields.items():
                        if "rope.dimension_count" in key:
                            rope_dim = _gguf_parse_value(
                                field.parts[field.data[0]], field.types
                            )
                            config["partial_rotary_factor"] = rope_dim / head_dim
                            break
                except Exception:
                    pass
        return result

    gguf_utils.load_gguf_checkpoint = patched_load_gguf_checkpoint

    # Also patch modules that imported load_gguf_checkpoint directly
    import transformers.models.auto.tokenization_auto as tok_auto
    import transformers.configuration_utils as config_utils
    import transformers.modeling_utils as modeling_utils

    for mod in (tok_auto, config_utils, modeling_utils):
        if hasattr(mod, "load_gguf_checkpoint"):
            mod.load_gguf_checkpoint = patched_load_gguf_checkpoint


# Apply the monkey-patch at import time
_patch_transformers_chatglm_gguf()


class ModelVariant(StrEnum):
    """Available GLM-4-9B-Chat Abliterated GGUF model variants for causal language modeling."""

    GLM_4_9B_CHAT_ABLITERATED_Q4_K_M = "9B_CHAT_ABLITERATED_Q4_K_M"


class ModelLoader(ForgeModel):
    """GLM-4-9B-Chat Abliterated GGUF model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GLM_4_9B_CHAT_ABLITERATED_Q4_K_M: LLMModelConfig(
            pretrained_model_name="bartowski/glm-4-9b-chat-abliterated-GGUF",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLM_4_9B_CHAT_ABLITERATED_Q4_K_M

    GGUF_FILE = "glm-4-9b-chat-abliterated-Q4_K_M.gguf"

    sample_text = "Give me a short introduction to large language models."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GLM-4-9B-Chat Abliterated GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        tokenizer_kwargs["gguf_file"] = self.GGUF_FILE

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, gguf_file=self.GGUF_FILE
            )
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [
            {
                "role": "user",
                "content": self.sample_text,
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )
        return self.config
