# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SeamlessM4T model loader implementation for speech-to-text translation
"""
import torch
import urllib.request
import io
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
    """Available SeamlessM4T model variants."""

    LARGE = "large"


class ModelLoader(ForgeModel):
    """SeamlessM4T model loader implementation for speech-to-text translation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/hf-seamless-m4t-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.config = None
        self.full_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="seamless-m4t",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor_and_config(self):
        """Load processor and config for the current variant.

        Returns:
            tuple: (processor, config) instances
        """
        from transformers import AutoProcessor, SeamlessM4TConfig

        model_name = self._variant_config.pretrained_model_name

        # Load config and processor
        self.config = SeamlessM4TConfig.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

        return self.processor, self.config

    def load_model(self, dtype_override=None):
        """Load and return the SeamlessM4T text decoder submodule for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The SeamlessM4T text decoder submodule.
        """
        from transformers import SeamlessM4TModel

        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Ensure processor and config are loaded
        if self.processor is None or self.config is None:
            self._load_processor_and_config()

        # Load full model
        self.full_model = SeamlessM4TModel.from_pretrained(
            model_name, config=self.config
        )

        if dtype_override is not None:
            self.full_model = self.full_model.to(dtype_override)

        # Return text_decoder submodule only
        return self.full_model.text_decoder

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the SeamlessM4T model with this instance's variant settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input arguments that can be fed to the text decoder.
        """
        import torchaudio

        # Ensure processor and full model are loaded
        if self.processor is None or self.full_model is None:
            raise RuntimeError(
                "Model and processor must be loaded before loading inputs"
            )

        # Load and format audio input
        url = "https://courses.cs.duke.edu/cps001/spring06/class/06_Sound/sounds/preamble.wav"
        with urllib.request.urlopen(url) as response:
            audio_data = response.read()
        audio_buffer = io.BytesIO(audio_data)
        audio, orig_freq = torchaudio.load(audio_buffer)

        # Resample audio
        audio = torchaudio.functional.resample(
            audio, orig_freq=orig_freq, new_freq=16_000
        )

        # Process audio
        audio_inputs = self.processor(audios=audio, return_tensors="pt")

        # Run encoder to get encoder_hidden_states
        encoder_outputs = self.full_model.speech_encoder(
            input_features=audio_inputs.input_features,
            attention_mask=audio_inputs.attention_mask,
        )
        encoder_hidden_states = encoder_outputs[0]

        # Prepare decoder input IDs
        tokenizer = self.processor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        decoder_input_ids = torch.tensor([[bos_token_id]])

        # Add batch dimension if batch_size > 1
        if batch_size > 1:
            decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                batch_size, dim=0
            )

        # Arguments are inputs for the text decoder submodule
        arguments = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }

        return arguments

    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded output information
        """
        if self.processor is None:
            self._load_processor_and_config()

        # Get logits from outputs
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Get predicted token IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode to text
        decoded_text = self.processor.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )

        return f"""
        SeamlessM4T Output:
          - Decoded text: "{decoded_text}"
          - Output shape: {logits.shape}
          - Predicted token IDs: {predicted_ids[0].tolist()}
        """
