# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gecko text embedding model loader implementation.

Gecko is a compact text embedding model from Google, distributed as TFLite
files via litert-community/Gecko-110m-en on HuggingFace. This loader downloads
the TFLite model and SentencePiece tokenizer, wraps them in a PyTorch-compatible
interface for use with the test harness.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from huggingface_hub import hf_hub_download

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
    """Available Gecko model variants for embedding generation."""

    GECKO_256 = "gecko-256"


class GeckoTFLiteWrapper(nn.Module):
    """PyTorch wrapper around a TFLite Gecko model for embedding generation."""

    def __init__(self, tflite_model_path: str):
        super().__init__()
        import ai_edge_litert as tflite

        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_ids_np = input_ids.numpy().astype(np.int32)
        attention_mask_np = attention_mask.numpy().astype(np.int32)

        self.interpreter.resize_tensor_input(
            self.input_details[0]["index"], input_ids_np.shape
        )
        self.interpreter.resize_tensor_input(
            self.input_details[1]["index"], attention_mask_np.shape
        )
        self.interpreter.allocate_tensors()

        self.interpreter.set_tensor(self.input_details[0]["index"], input_ids_np)
        self.interpreter.set_tensor(self.input_details[1]["index"], attention_mask_np)

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return torch.from_numpy(output.copy())


class ModelLoader(ForgeModel):
    """Gecko text embedding model loader implementation."""

    _VARIANTS = {
        ModelVariant.GECKO_256: ModelConfig(
            pretrained_model_name="litert-community/Gecko-110m-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GECKO_256

    TFLITE_FILENAME = "Gecko_256_f32.tflite"

    sample_sentences = ["This is an example sentence for embedding generation"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Gecko",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gecko TFLite model wrapped as a PyTorch module."""
        repo_id = self._variant_config.pretrained_model_name

        tflite_path = hf_hub_download(repo_id=repo_id, filename=self.TFLITE_FILENAME)

        model = GeckoTFLiteWrapper(tflite_path)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Gecko model."""
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download

        repo_id = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            sp_model_path = hf_hub_download(
                repo_id=repo_id, filename="sentencepiece.model"
            )
            self.tokenizer = spm.SentencePieceProcessor(model_file=sp_model_path)

        encoded = self.tokenizer.encode(self.sample_sentences[0], out_type=int)

        max_length = 256
        if len(encoded) > max_length:
            encoded = encoded[:max_length]

        attention_mask = [1] * len(encoded) + [0] * (max_length - len(encoded))
        input_ids = encoded + [0] * (max_length - len(encoded))

        input_ids = torch.tensor([input_ids], dtype=torch.int32)
        attention_mask = torch.tensor([attention_mask], dtype=torch.int32)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
