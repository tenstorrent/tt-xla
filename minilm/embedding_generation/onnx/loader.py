# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniLM ONNX model loader.
"""

# Reuse the PyTorch ModelLoader as the base
from ..pytorch.loader import ModelLoader as PyTorchModelLoader
from ....tools.utils import export_torch_model_to_onnx


class ModelLoader(PyTorchModelLoader):
    """MiniLM ONNX loader that inherits from the PyTorch loader."""

    def load_model(self, onnx_tmp_path, **kwargs):
        """Load MiniLM as a torch model, export to ONNX, then load and return the ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        # default variant ModelVariant.ALL_MINILM_L6_V2 is used if no variant is provided
        self.torch_loader = PyTorchModelLoader(variant=self._variant)
        torch_model = self.torch_loader.load_model()
        self.model = getattr(self.torch_loader, "model", torch_model)
        inputs = self.torch_loader.load_inputs()
        model_name = self.torch_loader._variant_config.pretrained_model_name

        return export_torch_model_to_onnx(
            torch_model,
            onnx_tmp_path,
            inputs,
            model_name,
        )

    def load_inputs(self, **kwargs):
        """Load and return preprocessed inputs for MiniLM embedding generation.

        Delegates to the underlying PyTorch loader to ensure tokenizer is initialized
        without re-invoking this ONNX loader's load_model signature.
        """
        return self.torch_loader.load_inputs(**kwargs)
