# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


class WhisperWrapper(torch.nn.Module):
    def __init__(self, model, variant="default"):
        """
        :param model: Loaded Whisper model (base or conditional gen).
        :param variant: One of ['default', 'large_v3', 'large_v3_turbo']
        """
        super().__init__()
        self.model = model
        self.variant = variant.value.split("/")[-1]

    def forward(self, input_features, decoder_input_ids, attention_mask=None):
        # Inputs arrive as keyword args because load_inputs() returns a dict and
        # TorchModelTester unpacks a Mapping into forward(**inputs).
        if self.variant == "Large_v3_Turbo":
            dec_out = self.model(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            return dec_out.logits

        elif self.variant == "Large_v3":
            output = self.model(
                input_features=input_features, decoder_input_ids=decoder_input_ids
            )
            return output[0]  # raw logits

        else:
            # default wrapper (e.g. WHISPER_TINY, BASE, etc.)
            output = self.model(
                input_features=input_features, decoder_input_ids=decoder_input_ids
            )
            return output.logits
