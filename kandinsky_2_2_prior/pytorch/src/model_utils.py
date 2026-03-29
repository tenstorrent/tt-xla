# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for Kandinsky 2.2 Prior model loading and processing.
"""

import torch
from diffusers import KandinskyV22PriorPipeline


def load_pipe(pretrained_model_name):
    """Load Kandinsky 2.2 Prior pipeline.

    Args:
        pretrained_model_name: HuggingFace model identifier

    Returns:
        KandinskyV22PriorPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = KandinskyV22PriorPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )
    pipe.to("cpu")

    modules = [pipe.prior, pipe.text_encoder]
    if pipe.image_encoder is not None:
        modules.append(pipe.image_encoder)

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def kandinsky_prior_preprocessing(pipe, prompt, device="cpu", num_inference_steps=1):
    """Preprocess inputs for the Kandinsky 2.2 Prior transformer model.

    Args:
        pipe: KandinskyV22PriorPipeline instance
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 1)

    Returns:
        tuple: (hidden_states, timestep, proj_embedding, encoder_hidden_states)
    """
    # Tokenize and encode the prompt
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    text_encoder_output = pipe.text_encoder(text_input_ids)
    prompt_embeds = text_encoder_output.text_embeds
    text_encoder_hidden_states = text_encoder_output.last_hidden_state

    # Set up scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Create noised image embeddings (hidden_states for the prior)
    embedding_dim = pipe.prior.config.embedding_dim
    hidden_states = torch.randn(
        (1, embedding_dim), device=device, dtype=prompt_embeds.dtype
    )

    timestep = timesteps[0].unsqueeze(0)

    return hidden_states, timestep, prompt_embeds, text_encoder_hidden_states
