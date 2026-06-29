# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SeamlessM4T v2 speech-to-text translation decoder example.

SeamlessM4T v2 is a multi-component multilingual model (text/speech encoders,
a text decoder, a text-to-unit model and a HiFi-GAN vocoder). This example
drives the headline bringup target -- the ``text_decoder`` submodule -- through
a single transformer-decoder forward pass on a TT device:

- the loader runs the speech encoder on a short audio clip (on host) to produce
  the cross-attention ``encoder_hidden_states``,
- the text decoder is compiled with the "tt" backend and run on device,
- the loader projects the decoder hidden state through the model's ``lm_head``
  and decodes the predicted token into human-readable text.

All weights/inputs come from the tt_forge_models ``ModelLoader`` public API.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.seamless_m4t_v2.pytorch import (
    ModelLoader,
    ModelVariant,
)


# --------------------------------
# Test run
# --------------------------------
def seamless_m4t_v2():
    """Run the SeamlessM4T v2 text decoder forward pass on a TT device."""
    # Match the bringup compiler configuration so the first compile is fast.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    # Load the text-decoder submodule and its inputs via the loader.
    loader = ModelLoader(ModelVariant.LARGE)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    device = torch_xla.device()

    # Move the model and the decoder inputs (input_ids + encoder_hidden_states)
    # to the device.
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compile with the TT backend and run a single decoder forward pass.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    return loader, output


def post_process_output(loader, output):
    """Print the human-readable speech-to-text decode result."""
    # Bring the decoder hidden state back to host so the loader can project it
    # through the (host-side) lm_head and decode the predicted token.
    output.last_hidden_state = output.last_hidden_state.to("cpu")
    print(loader.decode_output(output))


def test_seamless_m4t_v2():
    """Guard the SeamlessM4T v2 decoder example: finite output of expected shape."""
    xr.set_device_type("TT")

    loader, output = seamless_m4t_v2()

    hidden = output.last_hidden_state.to("cpu")
    hidden_size = loader.config.hidden_size

    assert torch.isfinite(hidden).all(), "Decoder hidden state contains non-finite values"
    assert hidden.shape[-1] == hidden_size, (
        f"Unexpected hidden size: got {hidden.shape[-1]}, expected {hidden_size}"
    )

    print(f"SeamlessM4T v2 decoder produced finite output of shape {tuple(hidden.shape)}.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    loader, output = seamless_m4t_v2()
    post_process_output(loader, output)
