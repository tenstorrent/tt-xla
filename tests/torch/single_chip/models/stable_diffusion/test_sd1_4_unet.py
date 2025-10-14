# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra.comparators import ComparisonConfig, TorchComparator, PccConfig

import torch
import torch_xla.runtime as xr
import torch_xla

import os

from diffusers import StableDiffusionPipeline, UNet2DConditionModel


@pytest.mark.parametrize("seq_len", [128])
def test_sd1_4_unet(seq_len):
    print("Starting test...")

    print("Loading the pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        # token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        num_inference_steps=1,
    )
    print("Pipeline loaded successfully")

    print("Extracting unet module directly from pipeline...")
    unet_module = pipe.unet
    unet_module.eval()
    print("unet module loaded successfully")

    # Alternative methods to load unet module
    """
    Alts:
        print("Loading unet module from pretrained...")
        unet_module = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            torch_dtype=torch.bfloat16
        )
        print("unet module loaded successfully")

        print("Loading the pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", torch_dtype=torch.bfloat16
        )
        print("Pipeline loaded successfully")

        print("Extracting unet module directly from pipeline...")
        unet_module = pipe.unet
        unet_module.eval()
        print("unet module loaded successfully")
    """

    # Args for forward pass based on UNet2DConditionModel's forward method:
    """
    Args:
        sample (`torch.Tensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.Tensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
    """

    # Prepare inputs for unet module
    print("Preparing inputs for unet module...")
    batch_size = 1
    test_seq_len = seq_len
    in_channels = unet_module.config.in_channels
    sample_size = unet_module.config.sample_size
    feature_dim = unet_module.config.cross_attention_dim

    sample = torch.randn(
        batch_size, in_channels, sample_size, sample_size
    )  # (batch size, channel, height, width)
    encoder_hidden_states = torch.randn(
        batch_size, test_seq_len, feature_dim
    )  # (batch size, sequence_len, feature_dim)
    timestep = torch.tensor([200], dtype=torch.long)  # ()

    inputs = {
        "sample": sample,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
    }
    inputs = {
        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }  # convert to bfloat16

    # Run on CPU for golden
    print("Running on CPU for golden...")
    cpu_output = unet_module(**inputs)

    print("cpu_output received!: ", cpu_output)

    # Compile on TT device
    print("Compiling...")
    compiled_unet = torch.compile(unet_module, backend="tt")

    # Set device type
    print("Setting device type to TT...")
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Move inputs  and modelto TT device
    print("Moving inputs and model to TT device...")
    compiled_unet.to(device)
    inputs_tt = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # Run on TT device
    print("Running on TT device...")
    tt_output = compiled_unet(**inputs_tt)

    # Move outputs to CPU
    print("Moving outputs to CPU...")
    tt_output = {
        k: v.to("cpu") if isinstance(v, torch.Tensor) else v
        for k, v in tt_output.items()
    }

    print("tt_output received!: ", tt_output)

    # compare outputs
    comparator = TorchComparator(
        ComparisonConfig(
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(tt_output, cpu_output)
