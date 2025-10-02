# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from third_party.tt_forge_models.stable_diffusion.pytorch.src.model_utils import stable_diffusion_preprocessing_v35


def test_stable_diffusion():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float32
    )
    transformer_module = pipe.transformer
    transformer_module.eval()

    # Figure out inputs for the transformer module
    prompt = "An astronaut riding a green horse"
    latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds = stable_diffusion_preprocessing_v35(
        pipe, prompt
    )
    
    # Prepare inputs for transformer
    inputs = {
        "hidden_states": latent_model_input,
        "encoder_hidden_states": prompt_embeds, 
        "pooled_projections": pooled_prompt_embeds,
        "timestep": timestep,
    }

    # Run on CPU for golden
    cpu_output = transformer_module(**inputs)
    
    # Run on TT device
    xr.set_device_type("TT")
    device = xm.xla_device()
    compiled_transformer = torch.compile(transformer_module.to("TT"), backend="tt")

    tt_output = compiled_transformer(**inputs)

    # compare outputs
    comparator = TorchComparator(
        ComparisonConfig(
            #atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    
    comparator.compare(tt_output, cpu_output)
    