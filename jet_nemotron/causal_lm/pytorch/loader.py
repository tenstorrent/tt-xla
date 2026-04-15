# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jet-Nemotron model loader implementation for causal language modeling.
"""

import functools

import torch
from typing import Optional, TypedDict

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _apply_compatibility_patches():
    """Apply runtime patches needed for Jet-Nemotron with transformers 5.x.

    Called lazily at model load time so that optional dependencies (fla)
    don't need to be installed at import/collection time.
    """
    import transformers.utils

    # The Jet-Nemotron HuggingFace model code imports LossKwargs from
    # transformers.utils, but it does not exist in transformers 5.x.
    if not hasattr(transformers.utils, "LossKwargs"):

        class LossKwargs(TypedDict, total=False):
            labels: torch.Tensor | None
            num_items_in_batch: int | None

        transformers.utils.LossKwargs = LossKwargs  # type: ignore[attr-defined]

    # The model's custom HuggingFace code passes autotune_interval to
    # FusedRMSNormGated, but the released fla package doesn't accept it.
    from fla.modules import FusedRMSNormGated

    if not getattr(FusedRMSNormGated.__init__, "_patched", False):
        _orig_init = FusedRMSNormGated.__init__

        @functools.wraps(_orig_init)
        def _patched_init(self, *args, autotune_interval=None, **kwargs):
            return _orig_init(self, *args, **kwargs)

        _patched_init._patched = True  # type: ignore[attr-defined]
        FusedRMSNormGated.__init__ = _patched_init  # type: ignore[method-assign]

    # The model's RoPE code uses rope_type='default', but transformers 5.x
    # doesn't include a 'default' entry in ROPE_INIT_FUNCTIONS.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:

        def _compute_default_rope_parameters(config=None, device=None, **kwargs):
            base = config.rope_theta
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device, dtype=torch.float
                    )
                    / dim
                )
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


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


class ModelVariant(StrEnum):
    """Available Jet-Nemotron model variants for causal language modeling."""

    JET_NEMOTRON_2B = "Jet_Nemotron_2B"
    JET_NEMOTRON_4B = "Jet_Nemotron_4B"


class ModelLoader(ForgeModel):
    """Jet-Nemotron model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.JET_NEMOTRON_2B: LLMModelConfig(
            pretrained_model_name="jet-ai/Jet-Nemotron-2B",
            max_length=128,
        ),
        ModelVariant.JET_NEMOTRON_4B: LLMModelConfig(
            pretrained_model_name="jet-ai/Jet-Nemotron-4B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JET_NEMOTRON_2B

    sample_text = "Give me a short introduction to large language model."

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jet-Nemotron",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        _apply_compatibility_patches()

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # The model config doesn't define pad_token_id, which causes an
        # AttributeError in transformers 5.x strict attribute access.
        # Load the config first and set a default before model construction.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        # The model declares _tied_weights_keys as a list, but transformers
        # 5.x expects a dict.  Disabling tie_word_embeddings avoids the
        # incompatible code path (weights are random in CI anyway).
        config.tie_word_embeddings = False

        model_kwargs = {"trust_remote_code": True, "config": config}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        messages = [{"role": "user", "content": self.sample_text}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
