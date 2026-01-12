# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BLIP PaddlePaddle model loader implementation
"""

from typing import Optional, List
from dataclasses import dataclass
from PIL import Image
import paddle
from paddlenlp.transformers import (
    BlipProcessor,
    BlipModel,
    BlipTextModel,
    BlipVisionModel,
)

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import get_file


@dataclass
class BlipConfig(ModelConfig):
    """Configuration specific to BLIP models"""

    task: "BlipTask"


class BlipTask(StrEnum):
    """Task types for BLIP models."""

    TEXT = "text"
    VISION = "vision"
    BLIP_IMAGE_CAPTIONING = "blip_image_captioning"


class ModelVariant(StrEnum):
    """Available BLIP model variants (Paddle) by task."""

    BLIP_IMAGE_CAPTIONING = "blip_image_captioning"
    BLIP_TEXT = "blip_text"
    BLIP_VISION = "blip_vision"


class ModelLoader(ForgeModel):
    """BLIP PaddlePaddle model loader implementation."""

    _VARIANTS = {
        ModelVariant.BLIP_IMAGE_CAPTIONING: BlipConfig(
            pretrained_model_name="Salesforce/blip-image-captioning-base",
            task=BlipTask.BLIP_IMAGE_CAPTIONING,
        ),
        ModelVariant.BLIP_TEXT: BlipConfig(
            pretrained_model_name="Salesforce/blip-image-captioning-base",
            task=BlipTask.TEXT,
        ),
        ModelVariant.BLIP_VISION: BlipConfig(
            pretrained_model_name="Salesforce/blip-image-captioning-base",
            task=BlipTask.VISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BLIP_IMAGE_CAPTIONING

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        task = cls._VARIANTS[variant].task
        if task == BlipTask.BLIP_IMAGE_CAPTIONING:
            model_task = ModelTask.NLP_EMBED_GEN
        elif task == BlipTask.VISION:
            model_task = ModelTask.CV_IMAGE_FE
        else:
            model_task = ModelTask.MM_IMAGE_TEXT_SIM

        return ModelInfo(
            model="blip_image_captioning",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=model_task,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained BLIP model for this instance's variant (Paddle)."""
        model_name = self._variant_config.pretrained_model_name
        task = self._variant_config.task

        if task == BlipTask.TEXT:
            model = BlipTextModel.from_pretrained(model_name)
            model.eval()
            return model
        elif task == BlipTask.VISION:
            model = BlipVisionModel.from_pretrained(model_name)
            model.eval()
            return model
        else:
            base_model = BlipModel.from_pretrained(model_name)

            class BlipWrapper(paddle.nn.Layer):
                def __init__(self, model: BlipModel):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids, pixel_values, attention_mask):
                    output = self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                    )
                    return output.text_embeds, output.image_embeds

            wrapped = BlipWrapper(base_model)
            return wrapped

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample inputs for BLIP model with this instance's variant settings (Paddle)."""
        model_name = self._variant_config.pretrained_model_name
        task = self._variant_config.task
        processor = BlipProcessor.from_pretrained(model_name)

        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_path))

        if task == BlipTask.TEXT:
            text = "a photo of cats in bed"
            inputs = processor(text=text, return_tensors="pd", padding=True)
            inputs = [inputs["input_ids"]]
            return inputs

        if task == BlipTask.VISION:
            images = [image] * batch_size
            enc = processor(images=images, return_tensors="pd", padding=True)
            return [enc["pixel_values"]]

        else:
            text = [
                "cats sleeping",
                "snowy weather",
            ]
            inputs = processor(
                images=image, text=text, return_tensors="pd", padding=True
            )
            self.text = text
            return [
                inputs["input_ids"],
                inputs["pixel_values"],
                inputs["attention_mask"],
            ]
