# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT ONNX model loader for sequence classification.

Loads the pre-exported ONNX model from optimum/distilbert-base-uncased-finetuned-sst-2-english.
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

_ONNX_REPO_ID = "optimum/distilbert-base-uncased-finetuned-sst-2-english"
_ONNX_FILENAME = "model.onnx"


class ModelLoader(PyTorchModelLoader):
    """DistilBERT ONNX loader that downloads the pre-exported ONNX model."""

    @classmethod
    def _get_model_info(cls, variant=None):
        return ModelInfo(
            model="DistilBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the pre-exported ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX model.
        """
        onnx_path = hf_hub_download(repo_id=_ONNX_REPO_ID, filename=_ONNX_FILENAME)
        model = onnx.load(onnx_path)
        return model
