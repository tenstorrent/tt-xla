# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v2 Base EN model loader implementation for sentence embedding generation.
"""
import sys
import types

import torch
from transformers import AutoTokenizer
from typing import Optional


def _patch_transformers_compat():
    """Patch missing transformers modules needed by jina-embeddings-v2 custom code.

    The jina-embeddings-v2 model uses custom code written for older transformers
    versions. This patches:
    - transformers.onnx (removed in transformers 5.x)
    - find_pruneable_heads_and_indices (removed in transformers 5.x)
    - PreTrainedModel.get_head_mask (removed in transformers 5.x)
    """
    if "transformers.onnx" not in sys.modules:
        onnx_module = types.ModuleType("transformers.onnx")

        class OnnxConfig:
            pass

        onnx_module.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = onnx_module

    from transformers import pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned):
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                if head not in already_pruned:
                    mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = (
            find_pruneable_heads_and_indices
        )

    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(
            self, head_mask, num_hidden_layers, is_attention_chunked=False
        ):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask


_patch_transformers_compat()


def _load_jina_v2_model(pretrained_model_name, **model_kwargs):
    """Load jina-embeddings-v2 model, working around transformers 5.x incompatibilities.

    The custom JinaBert code computes ALiBi tensors during __init__, which fails
    with transformers 5.x meta-device initialization. We manually instantiate
    the model class on CPU and then load the weights.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    config = AutoConfig.from_pretrained(pretrained_model_name, trust_remote_code=True)
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    if not hasattr(config, "add_cross_attention"):
        config.add_cross_attention = False

    model_kwargs.pop("trust_remote_code", None)
    dtype = model_kwargs.pop("torch_dtype", None)

    cls = get_class_from_dynamic_module(
        config.auto_map["AutoModel"],
        pretrained_model_name,
        trust_remote_code=True,
    )
    model = cls(config)

    weights_file = hf_hub_download(pretrained_model_name, "model.safetensors")
    state_dict = safetensors.torch.load_file(weights_file)
    model.load_state_dict(state_dict, strict=False)

    if dtype is not None:
        model = model.to(dtype)

    return model


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
    """Available Jina Embeddings v2 Base EN model variants."""

    JINA_EMBEDDINGS_V2_BASE_EN = "jina-embeddings-v2-base-en"


class ModelLoader(ForgeModel):
    """Jina Embeddings v2 Base EN model loader for English sentence embedding generation."""

    _VARIANTS = {
        ModelVariant.JINA_EMBEDDINGS_V2_BASE_EN: ModelConfig(
            pretrained_model_name="arkohut/jina-embeddings-v2-base-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.JINA_EMBEDDINGS_V2_BASE_EN

    sample_sentences = [
        "Jina Embeddings v2 is a versatile English embedding model with extended context"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Jina-Embeddings-v2-Base-EN",
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = _load_jina_v2_model(pretrained_model_name, **model_kwargs)
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

    def unpack_forward_output(self, fwd_output):
        tensors = []

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
