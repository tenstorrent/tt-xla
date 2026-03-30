# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MAIRA-2 model loader implementation for radiology report generation from chest X-rays.
"""

from typing import Optional

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
    """Available MAIRA-2 model variants."""

    MAIRA_2 = "maira_2"


class ModelLoader(ForgeModel):
    """MAIRA-2 model loader for radiology report generation from chest X-rays."""

    _VARIANTS = {
        ModelVariant.MAIRA_2: ModelConfig(
            pretrained_model_name="microsoft/maira-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAIRA_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="maira_2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Create a sample frontal chest X-ray image
        frontal_image = Image.new("L", (512, 512), color=128)

        processed_inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=frontal_image,
            current_lateral=None,
            prior_frontal=None,
            indication="Dyspnea.",
            technique="PA and lateral views of the chest.",
            comparison="None.",
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        )

        inputs = dict(processed_inputs)

        if dtype_override is not None:
            for key in inputs:
                if (
                    hasattr(inputs[key], "dtype")
                    and inputs[key].dtype.is_floating_point
                ):
                    inputs[key] = inputs[key].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if hasattr(inputs[key], "repeat_interleave"):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return fwd_output

    def decode_output(self, outputs):
        if self.processor is None:
            self._load_processor()

        if hasattr(outputs, "logits"):
            predicted_ids = outputs.logits.argmax(-1)
        else:
            predicted_ids = outputs[0].argmax(-1)

        generated_text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return generated_text
