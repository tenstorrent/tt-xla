# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import StableDiffusionPipeline

"""from utils import BringupStatus, Category
from infra import RunMode

from third_party.tt_forge_models.stable_diffusion_1_4.pytorch.loader import ModelLoader, ModelVariant

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)"""


# Create a wrapper for the Unet that handles device transfers for inputs to the module
class UnetWrapper(torch.nn.Module):
    def __init__(self, unet, device):
        super().__init__()
        self.unet = unet
        self.device = device

    def __getattr__(self, name):
        # Delegate all attribute access to the wrapped Unet
        if name in ["unet", "device"]:
            return super().__getattr__(name)
        return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # Move all inputs to TT device
        sample = sample.to(self.device)
        timestep = timestep.to(self.device)
        encoder_hidden_states = encoder_hidden_states.to(self.device)

        # Move any other tensor inputs to TT device
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(self.device)

        # Run Unet on TT device
        output = self.unet(sample, timestep, encoder_hidden_states, **kwargs)

        # Move output back to CPU for VAE processing
        if isinstance(output, tuple):
            output = tuple(
                tensor.to("cpu") if isinstance(tensor, torch.Tensor) else tensor
                for tensor in output
            )
        elif hasattr(output, "sample"):
            # Handle Unet2DConditionOutput object
            output.sample = output.sample.to("cpu")
        else:
            # Handle direct tensor output
            output = output.to("cpu")

        return output


@pytest.mark.parametrize("sample_dim", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "num_inf_steps", [1, 2, 5, 20, 50]
)  # default is 50 for the model
def test_sd1_4_pipe(sample_dim, num_inf_steps):
    torch.manual_seed(42)

    print("Starting hybrid pipeline test...")

    # Load full pipeline on CPU
    print("Loading the pipeline on CPU...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("Pipeline loaded successfully")

    # Compile only the Unet for TT device
    print("Compiling Unet for TT device...")
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Move Unet to TT and compile it
    pipe.unet.config.sample_size = (
        sample_dim  # NOTE: Currently sample_size 64 fails w/ OOM errors
    )
    # by default sample_size is 64, which produces a 512x512 image
    # VAE upsamples output from Unet by factor of 8
    pipe.unet.to(device)
    # pipe.unet = UnetWrapper(pipe.unet, device)
    pipe.unet = torch.compile(pipe.unet, backend="tt", options={"auto_to_xla": True})
    print("Unet compiled for TT device, device type")

    # Keep everything else on CPU
    pipe.text_encoder.to("cpu")
    pipe.vae.to("cpu")
    pipe.safety_checker.to("cpu")

    print(
        "Other components kept on CPU, device type: ",
        pipe.text_encoder.device,
        pipe.vae.device,
        pipe.safety_checker.device,
    )

    # Prepare inputs
    prompt = "a photo of a cat holding a sign that says hello world"

    # Run the full pipeline
    print("Running hybrid pipeline...")
    with torch.no_grad():
        # This will run text encoding on CPU, Unet on TT, VAE on CPU
        image = pipe(
            prompt=prompt,
            low_cpu_mem_usage=True,
            num_inference_steps=num_inf_steps,
        ).images[0]
        image.save(f"sd1_4_output_{sample_dim}_{num_inf_steps}.png")

    print("Hybrid pipeline completed successfully!")
    print(f"Generated image size: {image.size}")


# --------------------------------
# main
# --------------------------------

"""
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.CORRECT_RESULT,
)
@pytest.mark.parametrize("sample_dim", [8, 16, 32, 64])
def test_sd1_4_pipe(sample_dim):
    '''try:
        # By default torch_xla uses the CPU device so we have to set it to TT device.
        xr.set_device_type("TT")
    except Exception as e:
        pytest.skip(f"TT device not available: {e}")'''

    sd1_4_pipe(sample_dim)"""
