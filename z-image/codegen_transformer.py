# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone bug repro: codegen for the Z-Image transformer."""

import torch
import torch_xla
from diffusers import ZImagePipeline
from tt_torch import codegen_py

# CONFIG
compile_options = {
    "optimization_level": 1,
    "codegen_try_recover_structure": False,
}
EXPORT_PATH = "z_image_codegen/transformer"
torch_xla.set_custom_compile_options(compile_options)

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16


class TransformerWrapper(torch.nn.Module):
    """Wraps the Z-Image transformer to accept batched tensors instead of List[Tensor]."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, t, cap_feats):
        out_list = self.transformer(
            list(x.unbind(dim=0)),
            t,
            list(cap_feats.unbind(dim=0)),
            return_dict=False,
        )[0]
        return torch.stack(out_list)


def main():
    print("Loading pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    pipe.transformer.eval()

    print("Encoding prompts...")
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt="A photo of a cat sitting on a windowsill",
            negative_prompt="Rain",
            do_classifier_free_guidance=True,
        )

    # Build sample inputs matching a single denoising step with CFG
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.in_channels,
        height=1280,
        width=720,
        dtype=DTYPE,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )

    latents_typed = latents.to(pipe.transformer.dtype)
    latent_input = latents_typed.repeat(2, 1, 1, 1).unsqueeze(2)  # [2, C, 1, H, W]
    cap_feats_list = prompt_embeds + negative_prompt_embeds
    # Pad to same length and stack into a batched tensor
    max_len = max(c.shape[0] for c in cap_feats_list)
    cap_feats_padded = torch.stack(
        [
            torch.nn.functional.pad(c, (0, 0, 0, max_len - c.shape[0]))
            for c in cap_feats_list
        ]
    )  # [2, max_len, hidden_dim]
    timestep = torch.tensor([0.5, 0.5], dtype=torch.float32)

    print("Running transformer codegen...")
    wrapper = TransformerWrapper(pipe.transformer)
    wrapper.eval()

    codegen_py(
        wrapper,
        latent_input,
        timestep,
        cap_feats_padded,
        export_path=EXPORT_PATH,
        compiler_options=compile_options,
    )
    print("Done")


if __name__ == "__main__":
    main()
