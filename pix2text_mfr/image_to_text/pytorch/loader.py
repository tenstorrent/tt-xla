# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pix2Text-MFR model loader implementation for mathematical formula recognition.

This model only provides ONNX weights on HuggingFace, so we convert them to
PyTorch by parsing the ONNX graph and mapping initializers to the corresponding
VisionEncoderDecoderModel state dict keys.
"""
import torch
from typing import Optional

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
    """Available Pix2Text-MFR model variants."""

    PIX2TEXT_MFR = "pix2text_mfr"


class ModelLoader(ForgeModel):
    """Pix2Text-MFR model loader for mathematical formula recognition."""

    _VARIANTS = {
        ModelVariant.PIX2TEXT_MFR: ModelConfig(
            pretrained_model_name="breezedeus/pix2text-mfr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PIX2TEXT_MFR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.image_processor = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="pix2text_mfr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoImageProcessor, PreTrainedTokenizerFast

        pretrained = self._variant_config.pretrained_model_name
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained)

    @staticmethod
    def _build_matmul_map(onnx_model):
        """Map onnx::MatMul_xxx initializer names to their graph output paths."""
        mapping = {}
        for node in onnx_model.graph.node:
            if node.op_type == "MatMul":
                for inp in node.input:
                    if inp.startswith("onnx::MatMul"):
                        mapping[inp] = node.output[0]
                        break
        return mapping

    @staticmethod
    def _convert_onnx_to_state_dict(encoder_path, decoder_path, pt_state):
        """Convert ONNX encoder/decoder weights to a PyTorch state dict."""
        import onnx
        from onnx import numpy_helper

        encoder_onnx = onnx.load(encoder_path)
        decoder_onnx = onnx.load(decoder_path)

        encoder_weights = {
            i.name: numpy_helper.to_array(i) for i in encoder_onnx.graph.initializer
        }
        decoder_weights = {
            i.name: numpy_helper.to_array(i) for i in decoder_onnx.graph.initializer
        }

        enc_mm = ModelLoader._build_matmul_map(encoder_onnx)
        dec_mm = ModelLoader._build_matmul_map(decoder_onnx)

        new_state = {}

        # Encoder weights
        for name, arr in encoder_weights.items():
            if name.startswith("onnx::MatMul"):
                output = enc_mm[name]
                clean = output.replace("/MatMul_output_0", "").lstrip("/")
                clean = clean.replace("/", ".")
                pt_key = "encoder." + clean + ".weight"
                if pt_key in pt_state:
                    new_state[pt_key] = torch.from_numpy(arr.T.copy())
            else:
                pt_key = "encoder." + name
                if pt_key in pt_state:
                    new_state[pt_key] = torch.from_numpy(arr.copy())

        # Decoder weights
        for name, arr in decoder_weights.items():
            if name.startswith("onnx::MatMul"):
                output = dec_mm[name]
                if output == "logits":
                    pt_key = "decoder.output_projection.weight"
                else:
                    clean = output.replace("/MatMul_output_0", "").lstrip("/")
                    clean = clean.replace("/", ".")
                    pt_key = (
                        clean.replace("decoder.decoder.", "decoder.model.decoder.", 1)
                        + ".weight"
                    )
                if pt_key in pt_state:
                    new_state[pt_key] = torch.from_numpy(arr.T.copy())
            else:
                if name in pt_state:
                    new_state[name] = torch.from_numpy(arr.copy())

        return new_state

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
        from huggingface_hub import hf_hub_download

        pretrained = self._variant_config.pretrained_model_name

        if self.image_processor is None:
            self._load_processor()

        # Download ONNX files and convert to PyTorch
        encoder_path = hf_hub_download(pretrained, "encoder_model.onnx")
        decoder_path = hf_hub_download(pretrained, "decoder_model.onnx")

        config = VisionEncoderDecoderConfig.from_pretrained(pretrained)
        model = VisionEncoderDecoderModel(config)
        pt_state = model.state_dict()

        new_state = self._convert_onnx_to_state_dict(
            encoder_path, decoder_path, pt_state
        )
        model.load_state_dict(new_state, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        from PIL import Image

        if self.image_processor is None:
            self._load_processor()

        image = Image.new("RGB", (384, 384), color=(255, 255, 255))

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        decoder_input_ids = torch.tensor([[2]]).repeat_interleave(batch_size, dim=0)

        return {"pixel_values": pixel_values, "decoder_input_ids": decoder_input_ids}

    def decode_output(self, outputs):
        if self.image_processor is None:
            self._load_processor()

        if hasattr(outputs, "logits"):
            predicted_ids = outputs.logits.argmax(-1)
        else:
            predicted_ids = outputs[0].argmax(-1)

        generated_text = self.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return generated_text
