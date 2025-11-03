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
from transformers import SpeechT5HifiGan

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
        pos_seq = pos_seq.unsqueeze(-1) - pos_seq.unsqueeze(0)

        # Replacing advanced indexing with clamp to avoid graph breaks with TT compile.
        # Original code that causes issues:
        #   pos_seq[pos_seq < -self.max_length] = -self.max_length
        #   pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        pos_seq = torch.where(
            pos_seq < -self.max_length, -self.max_length,
            torch.where(pos_seq >= self.max_length, self.max_length - 1, pos_seq)
        )
        pos_seq = pos_seq + self.max_length

        return self.pe_k(pos_seq)


def get_model():
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts"
    )
    model.eval()
    model.speecht5.encoder.wrapped_encoder.embed_positions = SpeechT5RelativePositionalEncodingFixed(model.speecht5.encoder.wrapped_encoder.embed_positions)
    return model


def get_processor():
    processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts"
    )   
    return processor

def get_vocoder():
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return vocoder


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


def get_speaker_embeddings():
    """
    Load speaker embeddings from CMU Arctic dataset.
    This file looks messy because HF docs are not updated, so we're downloading the file however we can.
    """
    import pandas as pd
    import urllib.request
    import tempfile

    url = "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)

        df = pd.read_parquet(tmp_file.name)

        # Speaker embedding at index 7306 according to example https://huggingface.co/microsoft/speecht5_tts
        xvector = df.iloc[7306]["xvector"]
        speaker_embeddings = torch.tensor(xvector, dtype=torch.float32).unsqueeze(0)

        os.unlink(tmp_file.name)
    return speaker_embeddings

def dump_tensors():
    xr.set_device_type("TT")

    model = get_model()
    model.compile(backend="tt")

    model_inputs = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device if needed
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "dump_inputs": True,
        }
    )
    output = model(**model_inputs)

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
    model.compile(backend="tt")

    device = xm.xla_device()

    model_inputs = get_input()

    # Move inputs and model to device if needed
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    output = model(**model_inputs)
    print(output)

def run_vocoder_tt():
    xr.set_device_type("TT")
    vocoder = get_vocoder()

    # Wrap with fixed version to avoid graph breaks

    # torch inference first
    spectrogram = torch.randn(1, 100, 80)
    '''with torch.no_grad():
        speech = vocoder(spectrogram)'''


    vocoder.compile(backend="tt")
    device = xm.xla_device()
    vocoder = vocoder.to(device)
    spectrogram = spectrogram.to(device)


    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "backend": "codegen_py",
        }
    )

    speech = vocoder(spectrogram)
    print(speech)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    run_on_tt()
