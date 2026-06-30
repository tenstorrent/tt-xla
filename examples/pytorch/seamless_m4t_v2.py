# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SeamlessM4T v2 speech-to-text translation example (text-decoder forward pass).

``facebook/seamless-m4t-v2-large`` is a multimodal translation model. The speech
encoder runs on host to turn an audio waveform into ``encoder_hidden_states``;
the text decoder — the part brought up on Tenstorrent hardware — consumes those
hidden states plus a BOS token and predicts the first translated token.

This mirrors the single-forward vision examples (``compiler_options.py`` /
``resnet_dp.py``): build model + inputs from the tt-forge-models loader, compile
the decoder with the ``tt`` backend, run one forward pass on device, and decode
the result into a human-readable first-step prediction.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.seamless_m4t_v2.pytorch import (
    ModelLoader,
    ModelVariant,
)


def run_seamless_m4t_v2():
    """Run the SeamlessM4T v2 text decoder forward pass on a TT device.

    Returns:
        tuple: (loader, decoder output on host) so the caller can decode it.
    """
    # Match the optimization level the bringup used for a fast first compile.
    torch_xla.set_custom_compile_options({"optimization_level": 2})

    # Build the text decoder and its inputs via the loader. ``load_inputs``
    # runs the speech encoder on host and returns ``input_ids`` +
    # ``encoder_hidden_states`` for a single decode step.
    loader = ModelLoader(ModelVariant.LARGE)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    device = torch_xla.device()

    # Move the decoder and its inputs to device.
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compile with the TT backend and run a single forward pass.
    model.compile(backend="tt")
    with torch.no_grad():
        output = model(**inputs)

    last_hidden_state = (
        output.last_hidden_state
        if hasattr(output, "last_hidden_state")
        else output[0]
    )
    return loader, last_hidden_state.cpu()


def post_process_output(loader, last_hidden_state):
    """Print the human-readable first-step translation prediction."""
    summary = loader.decode_output(last_hidden_state)
    print(summary)


def test_seamless_m4t_v2():
    """Guard the example: the decoder produces a finite, well-shaped output."""
    xr.set_device_type("TT")

    loader, last_hidden_state = run_seamless_m4t_v2()

    hidden_size = loader.config.hidden_size
    assert last_hidden_state.shape[-1] == hidden_size, (
        f"expected hidden size {hidden_size}, got {last_hidden_state.shape[-1]}"
    )
    assert torch.isfinite(
        last_hidden_state.float()
    ).all(), "decoder output contains non-finite values"

    print(f"Decoder output shape: {tuple(last_hidden_state.shape)} (finite).")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    loader, last_hidden_state = run_seamless_m4t_v2()
    post_process_output(loader, last_hidden_state)
