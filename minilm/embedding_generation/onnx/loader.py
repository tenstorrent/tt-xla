# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniLM ONNX model loader for embedding generation.

Loads the pre-exported ONNX model from optimum/all-MiniLM-L6-v2.
"""

import onnx
from huggingface_hub import hf_hub_download

from ..pytorch.loader import ModelLoader as PyTorchModelLoader
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)

_ONNX_REPO_ID = "optimum/all-MiniLM-L6-v2"
_ONNX_FILENAME = "model.onnx"


class ModelLoader(PyTorchModelLoader):
    """MiniLM ONNX loader that downloads the pre-exported ONNX model."""

    @classmethod
    def _get_model_info(cls, variant=None):
        return ModelInfo(
            model="MiniLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the pre-exported ONNX model from optimum/all-MiniLM-L6-v2.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        onnx_path = hf_hub_download(repo_id=_ONNX_REPO_ID, filename=_ONNX_FILENAME)
        model = onnx.load(onnx_path)
        return model
