# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v2 Small EN model loader implementation for sentence embedding generation.
"""
import sys
import types

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from typing import List, Optional, Set, Tuple

# The jinaai/jina-embeddings-v2-small-en custom code depends on APIs removed
# in transformers 5.x. Shim them so trust_remote_code=True works.


def _find_pruneable_heads_and_indices(
    heads: Set[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


if not hasattr(
    __import__("transformers").pytorch_utils, "find_pruneable_heads_and_indices"
):
    import transformers.pytorch_utils

    transformers.pytorch_utils.find_pruneable_heads_and_indices = (
        _find_pruneable_heads_and_indices
    )

if "transformers.onnx" not in sys.modules:
    _onnx_mod = types.ModuleType("transformers.onnx")

    class _OnnxConfig:
        pass

    _onnx_mod.OnnxConfig = _OnnxConfig
    sys.modules["transformers.onnx"] = _onnx_mod

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
    """Available Jina Embeddings v2 Small EN model variants."""

    JINA_EMBEDDINGS_V2_SMALL_EN = "jina-embeddings-v2-small-en"


class ModelLoader(ForgeModel):
    """Jina Embeddings v2 Small EN model loader for sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.JINA_EMBEDDINGS_V2_SMALL_EN: ModelConfig(
            pretrained_model_name="jinaai/jina-embeddings-v2-small-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_EMBEDDINGS_V2_SMALL_EN

    sample_sentences = [
        "Jina Embeddings v2 is a lightweight English embedding model with extended context"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-Embeddings-v2-Small-EN",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
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

        return self.tokenizer

    @staticmethod
    def _get_head_mask(
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> List[Optional[torch.Tensor]]:
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return [head_mask[i] for i in range(num_hidden_layers)]
        return [None] * num_hidden_layers

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        # The custom JinaBERT code is incompatible with transformers 5.x
        # (meta device init + removed APIs), so we load manually on CPU.
        config = AutoConfig.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        if not hasattr(config, "is_decoder"):
            config.is_decoder = False
        if not hasattr(config, "add_cross_attention"):
            config.add_cross_attention = False

        if dtype_override is not None:
            config.torch_dtype = dtype_override

        model_class = get_class_from_dynamic_module(
            "jinaai/jina-bert-implementation--modeling_bert.JinaBertModel",
            pretrained_model_name,
        )
        model = model_class(config)

        # Patch get_head_mask removed in transformers 5.x
        if not hasattr(model, "get_head_mask"):
            model.get_head_mask = self._get_head_mask

        weights_path = hf_hub_download(pretrained_model_name, "model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs

    def output_postprocess(self, output, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(output, (tuple, list)):
            token_embeddings = output[0]
        elif hasattr(output, "last_hidden_state"):
            token_embeddings = output.last_hidden_state
        else:
            token_embeddings = output

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings

    def decode_output(self, outputs, inputs=None):
        return self.output_postprocess(outputs, inputs=inputs)
