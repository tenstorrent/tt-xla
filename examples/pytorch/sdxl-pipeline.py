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
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder_2", 
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return vae, unet, text_encoder, text_encoder_2

def load_scheduler():
    scheduler = EulerDiscreteScheduler.from_pretrained(
        MODEL_ID, 
        subfolder="scheduler"
    )
    return scheduler

def load_tokenizers():
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        trust_remote_code=True,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer_2",
        trust_remote_code=True,
    )
    return tokenizer, tokenizer_2
def generate(
    prompt: str,
    uncond_prompt: str,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed = None):
    """
    Generate an image from a prompt using the SDXL model.
    Only supports text2image generation for now.

    Args:
        prompt: The prompt to generate an image from.
        uncond_prompt: Negative prompt (for example tell the model not to generate a "car")
        do_cfg: Whether to use classifier-free guidance.
        cfg_scale: How much to follow the prompt - higher means more follow the prompt.
        num_inference_steps: How many steps to run the model for.
        seed: The seed to use for the random number generator.
    """

    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    vae, unet, text_encoder, text_encoder_2 = load_models() # to device
    tokenizer, tokenizer_2 = load_tokenizers()
    
    with torch.no_grad():
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()

        if do_cfg:
            encoder_hidden_states = []
            pooled_text_embeds = None
            for i, (curr_tokenizer, curr_text_encoder) in enumerate([(tokenizer, text_encoder), (tokenizer_2, text_encoder_2)]):
                cond_tokens = curr_tokenizer.batch_encode_plus(
                    [prompt],
                    padding="max_length",
                    max_length=77).input_ids # (B, T)
                
                uncond_tokens = curr_tokenizer.batch_encode_plus(
                    [uncond_prompt],
                    padding="max_length",
                    max_length=77).input_ids # (B, T)
                
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long) # to device
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long) # to device

                cond_output = curr_text_encoder(cond_tokens)
                cond_hidden_state = cond_output.last_hidden_state # (B, T, D)

                
                uncond_output = curr_text_encoder(uncond_tokens)
                uncond_hidden_state = uncond_output.last_hidden_state # (B, T, D)
                if curr_text_encoder == text_encoder_2:
                    pooled_cond_text_embeds = cond_output.pooler_output # (B, D)
                    pooled_uncond_text_embeds = uncond_output.pooler_output # (B, D)
                    pooled_text_embeds = torch.cat([pooled_uncond_text_embeds, pooled_cond_text_embeds], dim=0) # (2B, D)

                encoder_hidden_states.extend([uncond_hidden_state, cond_hidden_state]) # [(B, T, Di), (B, T, Di)]

            

        else:
            raise NotImplementedError("Only CFG is supported for now")
            
        scheduler = load_scheduler()
        scheduler.set_timesteps(num_inference_steps)
        
        latent_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        # this is for text2image
        latents = torch.randn(latent_shape, generator=generator, dtype=torch.float16) # to device

        target_shape = orig_shape = (HEIGHT, WIDTH)
        crop_top_left = (0, 0)
        time_ids = torch.tensor([*orig_shape, *crop_top_left, *target_shape]) # to device
        # repeat for cond and uncond
        time_ids = time_ids.repeat(2, 1) # (2B, 6)

        for i, timestep in enumerate(scheduler.timesteps):
            #timestep_tensor = torch.tensor([timestep] * batch_size, device=latents.device)
            #time_emb = get_timestep_embedding(timestep_tensor, embedding_dim=320, flip_sin_to_cos=True, downscale_freq_shift=0) # (B, D) to device
            
            model_input = latents.repeat(2, 1, 1, 1) if do_cfg else latents # (2B, 4, H, W) if do_cfg else (B, 4, H, W)
            unet_output = unet(model_input, timestep, encoder_hidden_states, added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": time_ids})
            print(unet_output)
            


if __name__ == "__main__":

    generate(
        prompt="a photo of a cat",
        uncond_prompt="car",
        do_cfg=True,
        cfg_scale=7.5,
        num_inference_steps=50,
        seed=42)

"""
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
        trust_remote_code=True,
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

input_embeds = None
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

latent_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
# this is for text2image
latents = torch.randn(latent_shape, generator=generator, dtype=torch.float16) # to device
model_input = latents.repeat(2, 1, 1, 1)
unet_output = unet(model_input, timestep, encoder_hidden_states=input_embeds)
"""