# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


class StableDiffusion35Wrapper(torch.nn.Module):
    def __init__(self, model, joint_attention_kwargs=None, return_dict=False):
        super().__init__()
        self.model = model
        self.joint_attention_kwargs = joint_attention_kwargs
        self.return_dict = return_dict

    def forward(
        self, latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds
    ):
        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=self.return_dict,
        )[0]
        return noise_pred
