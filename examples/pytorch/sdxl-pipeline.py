import torch
import torch.nn as nn
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from diffusers.models.embeddings import get_timestep_embedding
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from PIL import Image



class SDXLPipeline:
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    WIDTH = 1024
    HEIGHT = 1024
    LATENTS_WIDTH = WIDTH // 8
    LATENTS_HEIGHT = HEIGHT // 8

    def __init__(self, device='cpu'):
        self.device = device

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.MODEL_ID,
            subfolder="unet",
            variant="fp16",
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.MODEL_ID,
            subfolder="text_encoder",
            variant="fp16",
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.MODEL_ID,
            subfolder="text_encoder_2", 
            variant="fp16",
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

    def load_scheduler(self):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.MODEL_ID, 
            subfolder="scheduler"
        )

    def load_tokenizers(self):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.MODEL_ID,
            subfolder="tokenizer",
            trust_remote_code=True,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.MODEL_ID,
            subfolder="tokenizer_2",
            trust_remote_code=True,
        )

    def generate(
        self,
        prompt: str,
        uncond_prompt: str,
        do_cfg: bool = True,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int = None):
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
        
        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            if do_cfg:
                encoder_hidden_states = []
                pooled_text_embeds = None
                for i, (curr_tokenizer, curr_text_encoder) in enumerate([(self.tokenizer, self.text_encoder), (self.tokenizer_2, self.text_encoder_2)]):
                    cond_tokens = curr_tokenizer.batch_encode_plus(
                        [prompt],
                        padding="max_length",
                        max_length=77).input_ids # (B, T)
                    
                    uncond_tokens = curr_tokenizer.batch_encode_plus(
                        [uncond_prompt],
                        padding="max_length",
                        max_length=77).input_ids # (B, T)
                    
                    cond_tokens = torch.tensor(cond_tokens, dtype=torch.long).to(device=self.device)
                    uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long).to(device=self.device)

                    cond_output = curr_text_encoder(cond_tokens, output_hidden_states=True)
                     # "2" because SDXL always indexes from the penultimate layer.
                     # for reference, check https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py:412-414
                    cond_hidden_state = cond_output.hidden_states[-2] # (B, T, D)

                    uncond_output = curr_text_encoder(uncond_tokens, output_hidden_states=True)
                     # "2" because SDXL always indexes from the penultimate layer.
                     # for reference, check https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py:412-414
                    uncond_hidden_state = uncond_output.hidden_states[-2] # (B, T, D)

                    if curr_text_encoder == self.text_encoder_2:
                        pooled_cond_text_embeds = cond_output.text_embeds # (B, D)
                        pooled_uncond_text_embeds = uncond_output.text_embeds # (B, D)
                        pooled_text_embeds = torch.cat([pooled_uncond_text_embeds, pooled_cond_text_embeds], dim=0) # (2B, D)

                    curr_hidden_state = torch.cat([uncond_hidden_state, cond_hidden_state], dim=0) # (2B, T, Di)
                    encoder_hidden_states.append(curr_hidden_state)

                encoder_hidden_states = torch.cat(encoder_hidden_states, dim=-1) # (2B, T, D1 + D2)

            else:
                raise NotImplementedError("Only CFG is supported for now")
                
            self.scheduler.set_timesteps(num_inference_steps)
            
            latent_shape = (batch_size, 4, self.LATENTS_HEIGHT, self.LATENTS_WIDTH)
            # this is for text2image
            latents = torch.randn(latent_shape, generator=generator, dtype=torch.float16).to(device=self.device)
            latents = latents * self.scheduler.init_noise_sigma # important to preset the stddev

            target_shape = orig_shape = (self.HEIGHT, self.WIDTH)
            crop_top_left = (0, 0)
            time_ids = torch.tensor([*orig_shape, *crop_top_left, *target_shape]).to(device=self.device)
            # repeat for cond and uncond
            time_ids = time_ids.repeat(2, 1) # (2B, 6)

            for i, timestep in enumerate(self.scheduler.timesteps):
                print(f"Step {i} of {num_inference_steps}")
                model_input = latents.repeat(2, 1, 1, 1) if do_cfg else latents # (2B, 4, H, W) if do_cfg else (B, 4, H, W)
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                unet_output = self.unet(model_input, timestep, encoder_hidden_states, added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": time_ids}).sample
                #unet_output = torch.randn(2, 4, 64, 64, dtype=torch.float16) # working with hardcoded value to save time
                
                if do_cfg:
                    uncond_output, cond_output = unet_output.chunk(2)
                    model_output = uncond_output + (cond_output - uncond_output) * cfg_scale
                else:
                    raise NotImplementedError("Only CFG is supported for now")
                
                latents = self.scheduler.step(model_output, timestep, latents).prev_sample

            # decode from latent space        
            print(f"Decoding from latent space")    
            latents = latents / self.vae.config.scaling_factor # i assume this is needed?
            latents = latents.to(dtype=torch.float32)
            images = self.vae.decode(latents).sample # (B, 4, Latent_Height, Latent_Width) -> (B, 3, Image_Height, Image_Width)
            standardize = lambda x : (torch.clamp(x/2 + 0.5, 0., 1.)*255.).to(dtype=torch.uint8)
            images = standardize(images)
            images_np = images.cpu().squeeze().numpy()
            images_np = images_np.transpose(1, 2, 0)
            images_pil = Image.fromarray(images_np)
            images_pil.save("output.png")
                
                
            return images
                
if __name__ == "__main__":
    pipeline = SDXLPipeline(device='cpu')
    pipeline.setup()
    
    pipeline.generate(
        prompt="a photo of a cat",
        uncond_prompt="a photo of a dog",
        do_cfg=True,
        cfg_scale=7.5,
        num_inference_steps=50,
        seed=42
    )