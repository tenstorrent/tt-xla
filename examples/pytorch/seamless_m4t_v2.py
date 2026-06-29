# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SeamlessM4T v2 speech-to-text translation example (text-decoder forward).

SeamlessM4Tv2 is a multi-component speech/text translation model. Its key
device component on the speech-to-text path is the autoregressive text
decoder (speech_encoder -> text_decoder). This example loads the decoder
through the tt-forge-models loader, runs a single decode step on a TT device
from the encoder hidden states the loader produces (the speech encoder runs on
host), and decodes the predicted text token via the loader's lm_head helper.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.seamless_m4t_v2.pytorch import (
    ModelLoader,
    ModelVariant,
)


# --------------------------------
# SeamlessM4T v2 text-decoder forward
# --------------------------------
def seamless_m4t_v2():
    """Run one SeamlessM4Tv2 text-decoder step on a TT device.

    Returns:
        tuple: (loader, decoder output with last_hidden_state moved to host)
    """
    device = torch_xla.device()

    # Match the bringup baseline: bf16 weights/activations, optimization level 2.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    # Build the text decoder and its inputs via the loader's public API.
    # The loader runs the speech encoder on host and returns the resulting
    # encoder_hidden_states alongside the BOS decoder_input_ids.
    loader = ModelLoader(ModelVariant.LARGE)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Move model and inputs to the device.
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compile for the TT backend and run one decode step.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    # Bring the decoder hidden states back to host so the loader can project
    # them through its (host-side) lm_head for decoding.
    output.last_hidden_state = output.last_hidden_state.to("cpu")
    return loader, output


def post_process_output(loader, output):
    """Print the human-readable decoded token for the first decode step."""
    print(loader.decode_output(output))


def test_seamless_m4t_v2():
    """Guard: the decoder forward runs on device and yields a finite token."""
    xr.set_device_type("TT")

    loader, output = seamless_m4t_v2()

    # Expected shape: (batch=1, seq=1, hidden_size).
    hidden_size = loader.config.hidden_size
    last_hidden_state = output.last_hidden_state
    assert last_hidden_state.shape == (
        1,
        1,
        hidden_size,
    ), f"unexpected decoder output shape: {tuple(last_hidden_state.shape)}"
    assert torch.isfinite(last_hidden_state.float()).all(), "non-finite decoder output"

    # The loader's decoder must produce a non-empty human-readable report.
    report = loader.decode_output(output)
    assert isinstance(report, str) and report.strip(), "empty decode_output report"

    print("SeamlessM4Tv2 text-decoder step produced a finite, decodable output.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    loader, output = seamless_m4t_v2()
    post_process_output(loader, output)
