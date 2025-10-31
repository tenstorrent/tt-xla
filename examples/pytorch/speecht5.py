# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor


EXPORT_PATH = "speecht5"


class SpeechT5RelativePositionalEncodingFixed(nn.Module):
    """
    Implementation of the relative positional encoding that avoids advanced indexing 
    to avoid graph breaks with TT compile.
    """
    def __init__(self, original_module):
        super().__init__()
        self.dim = original_module.dim
        self.max_length = original_module.max_length
        self.pe_k = original_module.pe_k

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        pos_seq = torch.arange(0, seq_len, device=hidden_states.device, dtype=torch.long)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]

        # Replacing advanced indexing with clamp to avoid graph breaks with TT compile.
        # Original code that causes issues:
        #   pos_seq[pos_seq < -self.max_length] = -self.max_length
        #   pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        pos_seq = torch.clamp(pos_seq, -self.max_length, self.max_length - 1)
        pos_seq = pos_seq + self.max_length

        return self.pe_k(pos_seq)


def get_model():
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts"
    )
    model.eval()
    return model


def get_processor():
    processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts"
    )   
    return processor


def get_input():
    processor = get_processor()
    model = get_model()

    # Prepare inputs for the decoder
    # First, get encoder outputs by processing text through the full model encoder
    text = "Hello, my dog is cute."
    inputs = processor(text=text, return_tensors="pt")

    # Create decoder input values (zeros for initial state)
    decoder_input_values = torch.zeros((1, 1, model.config.num_mel_bins))
    
    # Prepare model inputs
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_values": decoder_input_values,
    }

    # # Serialize model_inputs to disk
    # torch.save(model_inputs, "model_inputs.pt")

    # # Load model_inputs from disk
    # model_inputs = torch.load("model_inputs.pt")

    return model_inputs


def dump_tensors():
    xr.set_device_type("TT")

    model = get_model()
    model.compile(backend="tt")

    input = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "dump_inputs": True,
        }
    )
    output = model(input)

    return


def dump_code():
    xr.set_device_type("TT")

    model = get_model()

    model.compile(backend="tt")

    device = xm.xla_device()

    model_inputs = get_input()

    # Move inputs and model to device if needed
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "backend": "codegen_py",
        }
    )

    output = model(**model_inputs)
    print(output)


def run_on_cpu():
    model = get_model()
    processor = get_processor()

    # Process text input using the processor
    text = "Hello, my dog is cute."
    inputs = processor(text=text, return_tensors="pt")
    
    # Create decoder input values (zeros for initial state)
    decoder_input_values = torch.zeros((1, 1, model.config.num_mel_bins))

    # Prepare model inputs
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_values": decoder_input_values,
    }

    output = model(**model_inputs)
    print(output)


def run_on_tt():
    xr.set_device_type("TT")

    model = get_model()
    model.speecht5.encoder.wrapped_encoder.embed_positions = SpeechT5RelativePositionalEncodingFixed(model.speecht5.encoder.wrapped_encoder.embed_positions)
    model.compile(backend="tt")

    device = xm.xla_device()

    model_inputs = get_input()

    # Move inputs and model to device if needed
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    output = model(**model_inputs)
    print(output)

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    run_on_tt()