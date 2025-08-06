# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode, random_image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    FlaxPreTrainedModel,
    FlaxVisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTConfig,
)


class VisionTextDualEncoderTester(JaxModelTester):
    """Tester for Vision-Text Dual Encoder (VTDE) model."""

    def __init__(
        self,
        vision_model_path: str,
        text_model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._vision_model_path = vision_model_path
        self._text_model_path = text_model_path
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(
            self._vision_model_path, self._text_model_path
        )

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        model_config = ViTConfig.from_pretrained(self._vision_model_path)
        image_size = model_config.image_size
        random_image = random_image(image_size)

        tokenizer = AutoTokenizer.from_pretrained(self._text_model_path)
        image_processor = AutoImageProcessor.from_pretrained(self._vision_model_path)
        processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

        inputs = processor(
            text="Some random image", images=random_image, return_tensors="jax"
        )
        return inputs

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
