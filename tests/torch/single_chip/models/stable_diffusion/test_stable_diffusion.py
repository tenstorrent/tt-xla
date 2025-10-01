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
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

def load_pipe(variant):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float32
    )
    modules = pipe.transformer
    
    module.eval()
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad = False

    return pipe


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
def test_stable_diffusion():
    pipe = load_pipe()
    
    # Move the pipeline to CPU
    pipe.to("cpu")