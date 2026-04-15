# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ERNIE 4.5 VL model loader implementation for multimodal image-text-to-text generation.
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ERNIE 4.5 VL model variants."""

    ERNIE_4_5_VL_28B_A3B_PT = "28B_A3B_PT"


class ModelLoader(ForgeModel):
    """ERNIE 4.5 VL model loader implementation for multimodal image-text-to-text generation tasks."""

    _VARIANTS = {
        ModelVariant.ERNIE_4_5_VL_28B_A3B_PT: LLMModelConfig(
            pretrained_model_name="baidu/ERNIE-4.5-VL-28B-A3B-PT",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ERNIE_4_5_VL_28B_A3B_PT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                },
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ERNIE 4.5-VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.processor

    @staticmethod
    def _patch_ernie_config(pretrained_model_name):
        """Patch upstream ERNIE 4.5 VL config bugs before model loading.

        The upstream config has two bugs:
        1. Ernie45MoeConfig.__init__ accesses kwargs["num_hidden_layers"] without
           a default, crashing when transformers creates a default config for
           diff-based repr logging.
        2. Ernie4_5_VLMoEConfig.__init__ does not forward moe_use_hard_gate to
           its parent Ernie45MoeConfig, so the parent always overwrites it with
           False, causing MOELayer init to crash with a None gate.
        """
        from transformers import AutoConfig
        from transformers.configuration_utils import PretrainedConfig

        _orig_repr = PretrainedConfig.__repr__

        def _safe_repr(self):
            try:
                return _orig_repr(self)
            except Exception:
                return f"{self.__class__.__name__}(...)"

        PretrainedConfig.__repr__ = _safe_repr
        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name, trust_remote_code=True
            )
        finally:
            PretrainedConfig.__repr__ = _orig_repr

        # Patch the parent MoE config class to use safe defaults
        parent_cls = type(config).__mro__[1]  # Ernie45MoeConfig
        orig_init = parent_cls.__init__

        def _patched_init(self, *args, **kw):
            # Use .get() for num_hidden_layers to avoid KeyError on default init
            if "num_hidden_layers" not in kw:
                kw.setdefault("num_hidden_layers", 0)
            orig_init(self, *args, **kw)

        parent_cls.__init__ = _patched_init

        # Patch VL config to ensure moe_use_hard_gate is True after init.
        # The upstream config JSON has moe_use_hard_gate=null, and the parent
        # class overwrites it with False, so we force it to True after init.
        vl_cls = type(config)
        orig_vl_init = vl_cls.__init__

        def _patched_vl_init(self, *args, moe_use_hard_gate=True, **kw):
            if moe_use_hard_gate is None:
                moe_use_hard_gate = True
            orig_vl_init(self, *args, moe_use_hard_gate=moe_use_hard_gate, **kw)
            self.moe_use_hard_gate = moe_use_hard_gate

        vl_cls.__init__ = _patched_vl_init

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": False,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        self._patch_ernie_config(pretrained_model_name)

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        text = self.processor.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[text],
            images=[dummy_image],
            videos=[],
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.processor.decode(next_token_id)
