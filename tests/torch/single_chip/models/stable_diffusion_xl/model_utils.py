# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Stable Diffusion Demo Script

import torch


class StableDiffusionXLWrapper(torch.nn.Module):
    def __init__(self, model, added_cond_kwargs, cross_attention_kwargs=None):
        super().__init__()
        self.model = model
        self.cross_attention_kwargs = cross_attention_kwargs
        self.added_cond_kwargs = added_cond_kwargs

    def forward(self, latent_model_input, timestep, prompt_embeds):
        # Ensure all inputs are on the same device as the latents/model
        device = latent_model_input.device
        prompt_embeds = prompt_embeds.to(device)
        timestep_on_device = timestep[0].to(device)
        added_cond_kwargs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self.added_cond_kwargs.items()
        }

        noise_pred = self.model(
            latent_model_input,
            timestep_on_device,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )[0]
        return noise_pred
