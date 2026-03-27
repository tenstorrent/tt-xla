# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Step-3.5-Flash model loader implementation for causal language modeling.

Uses reduced MoE configuration for testing since the full model is too
large to load directly.
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """Step-3.5-Flash model loader for causal language modeling."""

    def __init__(self, variant=None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.model_name = "stepfun-ai/Step-3.5-Flash"
        self.tokenizer = None
        self.text = "What is machine learning?"
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Step-3.5-Flash",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Reduce model dimensions for testing
        num_layers = self.num_layers if self.num_layers is not None else 6
        config.num_hidden_layers = num_layers
        config.num_attention_heads = 16
        config.hidden_size = 1024
        config.num_attention_groups = 4
        config.intermediate_size = 1024 * 4
        config.moe_num_experts = 8
        config.moe_top_k = 2
        config.moe_intermediate_size = 512
        config.num_nextn_predict_layers = 0

        # Trim per-layer config lists to match reduced layer count
        total_layers = num_layers + config.num_nextn_predict_layers
        for attr in [
            "layer_types",
            "rope_theta",
            "partial_rotary_factors",
            "swiglu_limits",
            "swiglu_limits_shared",
        ]:
            if hasattr(config, attr):
                value = getattr(config, attr)
                if isinstance(value, list) and len(value) > total_layers:
                    setattr(config, attr, value[:total_layers])

        # Update MoE layer indices to only include valid layers
        if hasattr(config, "moe_layers_enum") and isinstance(
            config.moe_layers_enum, str
        ):
            original_moe_layers = [
                int(x) for x in config.moe_layers_enum.split(",") if x.strip()
            ]
            valid_moe_layers = [l for l in original_moe_layers if l < num_layers]
            config.moe_layers_enum = ",".join(str(l) for l in valid_moe_layers)

        # Reduce sliding attention head count to match
        if hasattr(config, "attention_other_setting"):
            config.attention_other_setting["num_attention_heads"] = 16
            config.attention_other_setting["num_attention_groups"] = 4

        model_kwargs = {
            "attn_implementation": "eager",
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_config(config, **model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        return model

    def load_inputs(self, batch_size=1):
        if self.tokenizer is None:
            self.load_model()

        inputs = self.tokenizer(self.text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
