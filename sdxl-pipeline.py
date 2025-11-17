import torch
import torch.nn as nn
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding
from transformers import CLIPTextModel

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def load_models():
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID,
        subfolder="unet",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",  # Note: The 'text_encoder' subfolder holds the first text encoder.
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return vae, unet, text_encoder

def load_scheduler():
    scheduler = EulerDiscreteScheduler.from_pretrained(
        MODEL_ID, 
        subfolder="scheduler"
    )
    return scheduler

def load_tokenizer():
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        trust_remote_code=True,
    )

def generate(
    prompt: str,
    uncond_prompt: str, # negative prompt (for example tell the model not to generate a "car")
    do_cfg: bool = True, # whether to use classifier-free guidance
    cfg_scale: float = 7.5, # how much to follow the prompt - higher means more follow the prompt
    num_inference_steps: int = 50, # how many steps to run the model for
    seed = None):

    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    vae, unet, text_encoder = load_models() # to device
    tokenizer = load_tokenizer()
    
    with torch.no_grad():
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77).input_ids # (B, T)
            
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt],
                padding="max_length",
                max_length=77).input_ids # (B, T)
            
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long) # to device
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long) # to device

            cond_emb = text_encoder(cond_tokens).last_hidden_state # (B, T, D)  
            uncond_emb = text_encoder(uncond_tokens).last_hidden_state # (B, T, D)

            input_embeds = torch.cat([uncond_emb, cond_emb], dim=0) # (2B, T, D)
        else:
            raise NotImplementedError("Only CFG is supported for now")
            
        scheduler = load_scheduler()
        scheduler.set_timesteps(num_inference_steps)
        

        latents = torch.randn((batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH), generator=generator) # to device

        for i, timestep in enumerate(scheduler.timesteps):
            timestep_tensor = torch.tensor([timestep] * batch_size, device=latents.device)
            time_emb = get_timestep_embedding(timestep_tensor, embedding_dim=320, flip_sin_to_cos=True, downscale_freq_shift=0) # (B, D) to device
            
            model_input = latents.repeat(2, 1, 1, 1) if do_cfg else latents # (2B, 4, H, W) if do_cfg else (B, 4, H, W)
            


if __name__ == "__main__":

    generate(
        prompt="a photo of a cat",
        uncond_prompt="car",
        do_cfg=True,
        cfg_scale=7.5,
        num_inference_steps=50,
        seed=42)