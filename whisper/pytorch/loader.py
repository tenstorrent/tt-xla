# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from ...base import ForgeModel
from datasets import load_dataset


class ModelLoader(ForgeModel):

    # Shared configuration parameters
    model_name = "openai/whisper-tiny"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load a Whisper model from Hugging Face."""

        # Initialize processor first with default or overridden dtype
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        cls.processor = WhisperProcessor.from_pretrained(
            cls.model_name, use_cache=False, return_dict=False, **processor_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = WhisperForConditionalGeneration.from_pretrained(
            cls.model_name, use_cache=False, return_dict=False, **model_kwargs
        )
        model.eval()
        return model

    @classmethod
    def load_inputs(cls):
        """Generate sample inputs for Whisper model."""

        # Ensure processor is initialized
        if not hasattr(cls, "processor"):
            cls.load_model()  # This will initialize the processor

        # load dummy dataset and read audio files
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = ds[0]["audio"]
        inputs = cls.processor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
        ).input_features
        return inputs
