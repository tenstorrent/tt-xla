# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3 ONNX model loader for embedding tasks.
"""

from ..pytorch.loader import ModelLoader as PyTorchModelLoader, ModelVariant
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....tools.utils import export_torch_model_to_onnx

from typing import Optional


class ModelLoader(PyTorchModelLoader):
    """Qwen 3 ONNX loader for embedding tasks that inherits from the PyTorch loader."""

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model info for the ONNX variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Qwen 3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load Qwen 3 as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        self.torch_loader = PyTorchModelLoader(
            variant=self._variant, num_layers=self.num_layers
        )
        torch_model = self.torch_loader.load_model(**kwargs)
        self.model = getattr(self.torch_loader, "model", torch_model)
        self.tokenizer = self.torch_loader.tokenizer
        inputs = self.torch_loader.load_inputs()
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for Qwen 3 embedding.

        Delegates to the underlying PyTorch loader to ensure tokenizer is initialized
        without re-invoking this ONNX loader's load_model signature.
        """
        return self.torch_loader.load_inputs(**kwargs)
