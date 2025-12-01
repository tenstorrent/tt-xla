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

from enum import Enum

class SDXLVariants(Enum):
    SDXL_IMG_1024 = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_IMG_512 = "hotshotco/SDXL-512"

class SDXLConfig:
    available_devices = ['cpu', 'cuda']
    available_dims = [512, 1024]

    def __init__(self, width=1024, height=1024, device='cpu'):
        assert width == height, "Currently we only support square images"
        assert width in SDXLConfig.available_dims, f"Invalid width: {width}. Available dimensions: {SDXLConfig.available_dims}"
        self.width = width
        self.height = height
        self.latents_width = width // 8
        self.latents_height = height // 8
        self.model_id = SDXLVariants.SDXL_IMG_1024.value if width == 1024 else SDXLVariants.SDXL_IMG_512.value
        assert device in SDXLConfig.available_devices, f"Invalid device: {device}. Available devices: {SDXLConfig.available_devices}"
        self.device = device

class SDXLPipeline:
    def __init__(self, config: SDXLConfig):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.width = config.width
        self.height = config.height
        self.latents_width = config.latents_width
        self.latents_height = config.latents_height

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizers()

    def load_models(self):
        # Hotshotco doesn't have native fp16 weights, so we just download bigger model and cast ourselves
        variant = "fp16" if self.model_id == SDXLVariants.SDXL_IMG_1024.value else None

        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            variant=variant,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        device = xm.xla_device()
        self.unet = self.unet.to(device)
        self.unet.compile(backend="tt")

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            variant=variant,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.model_id,
            subfolder="text_encoder_2", 
            variant=variant,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

    def load_scheduler(self):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, 
            subfolder="scheduler"
        )

    def load_tokenizers(self):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
            trust_remote_code=True,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.model_id,
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
            
            latent_shape = (batch_size, 4, self.latents_height, self.latents_width)
            # this is for text2image
            latents = torch.randn(latent_shape, generator=generator, dtype=torch.float16).to(device=self.device)
            latents = latents * self.scheduler.init_noise_sigma # important to preset the stddev

            target_shape = orig_shape = (self.height, self.width)
            crop_top_left = (0, 0)
            time_ids = torch.tensor([*orig_shape, *crop_top_left, *target_shape]).to(device=self.device)
            # repeat for cond and uncond
            time_ids = time_ids.repeat(2, 1) # (2B, 6)
            tt_device = xm.xla_device()
            tt_cast = lambda x : x.to(tt_device, dtype=torch.bfloat16)
            cpu_cast = lambda x : x.to('cpu').to(dtype=torch.float16)

            for i, timestep in enumerate(self.scheduler.timesteps):
                print(f"Step {i} of {num_inference_steps}")
                model_input = latents.repeat(2, 1, 1, 1) if do_cfg else latents # (2B, 4, H, W) if do_cfg else (B, 4, H, W)
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                model_input = tt_cast(model_input)
                timestep = tt_cast(timestep)
                encoder_hidden_states = tt_cast(encoder_hidden_states)
                pooled_text_embeds = tt_cast(pooled_text_embeds)
                time_ids = tt_cast(time_ids)
                unet_output = self.unet(model_input, timestep, encoder_hidden_states, added_cond_kwargs={"text_embeds": pooled_text_embeds, "time_ids": time_ids})
                unet_output = cpu_cast(unet_output.sample)

                xr.clear_computation_cache() # turns out that consteval cache is accumulating in DRAM which causes DRAM OOM. This is a temp workaround.

                if do_cfg:
                    uncond_output, cond_output = unet_output.chunk(2)
                    model_output = uncond_output + (cond_output - uncond_output) * cfg_scale
                else:
                    raise NotImplementedError("Only CFG is supported for now")

                timestep = cpu_cast(timestep)
                latents = cpu_cast(latents)
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
    xr.set_device_type("TT")
    config = SDXLConfig(width=512, height=512, device='cpu')
    pipeline = SDXLPipeline(config=config)
    pipeline.setup()
    
    pipeline.generate(
        prompt="a photo of a cat",
        uncond_prompt="a photo of a dog",
        do_cfg=True,
        cfg_scale=7.5,
        num_inference_steps=50,
        seed=42
    )