import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
        x_flat, y_flat = x.flatten(), y.flatten()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()

        return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

def run_sdxl_unet_inference():
    xr.set_device_type("TT")

    # Load SDXL pipeline and extract the UNet
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    model = pipeline.unet.to(torch.bfloat16)
    model = model.eval()


     # Generate inputs for SDXL UNet
    # sample: (batch_size, in_channels=4, height, width)
    sample = torch.rand((2, 4, 64, 64), dtype=torch.bfloat16)

    # timestep: scalar or tensor representing the denoising step
    timestep = torch.randint(0, 1000, (1,))

    # encoder_hidden_states: (batch_size, sequence_length, cross_attention_dim=2048)
    encoder_hidden_states = torch.rand((2, 77, 2048), dtype=torch.bfloat16)

    # SDXL requires additional conditioning arguments
    added_cond_kwargs = {
        "text_embeds": torch.rand((2, 1280), dtype=torch.bfloat16),  # Pooled text embeddings
        "time_ids": torch.rand((2, 6), dtype=torch.bfloat16)  # Time conditioning
    }

    with torch.no_grad():
        output = model(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

    model.compile(backend="tt")

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    sample = sample.to(device)
    timestep = timestep.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    added_cond_kwargs["text_embeds"] = added_cond_kwargs["text_embeds"].to(device)
    added_cond_kwargs["time_ids"] = added_cond_kwargs["time_ids"].to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

    print(output)


def run_sdxl_unet_downblock_inference():
    xr.set_device_type("TT")

    # Load SDXL pipeline and extract the UNet
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    unet = pipeline.unet.to(torch.bfloat16)
    unet = unet.eval()

    model = unet.down_blocks[0]

     # Generate inputs for SDXL UNet
    # sample: (batch_size, in_channels=4, height, width)
    sample = torch.zeros(2, 320, 64, 64, dtype=torch.bfloat16)
    emb = torch.zeros(2, 1280, dtype=torch.bfloat16)

    with torch.no_grad():
        output = model(sample, emb)

    model.compile(backend="tt")

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    sample = sample.to(device)
    emb = emb.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(sample, emb)

    print("DONE")


def run_sdxl_unet_resnet():
    xr.set_device_type("TT")

    # Load SDXL pipeline and extract the UNet
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.bfloat16,
        variant="fp16"
    )
    unet = pipeline.unet.to(torch.bfloat16)
    unet = unet.eval()

    model = unet.down_blocks[0].resnets[0]

     # Generate inputs for SDXL UNet
    # sample: (batch_size, in_channels, height, width)
    sample = torch.zeros(2, 320, 64, 64, dtype=torch.bfloat16)
    emb = torch.zeros(2, 1280, dtype=torch.bfloat16)

    model.eval()

    with torch.no_grad():
        torch_output = model(sample, emb)

    model.compile(backend="tt")
    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    sample = sample.to(device)
    emb = emb.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(sample, emb)

    print(torch.allclose(torch_output.cpu(), output.cpu(), atol=1e-2, rtol=1e-2))
    print(f"compute_pcc(torch_output, output): {compute_pcc(torch_output, output)}")


class ResNetGroupNorm(nn.Module):
    def __init__(self, norm1):
        super().__init__()
        self.norm1 = norm1

    def forward(self, sample):
        return self.norm1(sample)


def run_sdxl_unet_resnet_groupnorm():
    xr.set_device_type("TT")

    # Load SDXL pipeline and extract the UNet
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    unet = pipeline.unet.to(torch.bfloat16)
    unet = unet.eval()

    # Extract GroupNorm from first ResNet block
    resnet_block = unet.down_blocks[0].resnets[0]
    model = ResNetGroupNorm(resnet_block.norm1)

    # Generate inputs for GroupNorm
    # sample: (batch_size, in_channels=320, height, width)
    sample = torch.zeros(2, 320, 64, 64, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_output = model(sample)

    model.compile(backend="tt")

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    sample = sample.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(sample)

    print("DONE")

def run_sdxl_unet_text_encoder():
    xr.set_device_type("TT")

    # Load SDXL pipeline and extract the UNet
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    model = pipeline.text_encoder
    model = model.to(torch.bfloat16)
    model = model.eval()

    model.compile(backend="tt")

    B, T, D = 1, 77, 768

    prompt = "a photo of a cat"
    input_ids = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=T,
        truncation=True,
        return_tensors="pt",
    )['input_ids']

    with torch.no_grad():
        output = model(input_ids)

    print(output)

    print("DONE")

def run_sdxl_unet_vae():
    xr.set_device_type("TT")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    sample_img = torch.randn(1, 3, 512, 512, dtype=torch.bfloat16)
    model = pipeline.vae
    model = model.to(torch.bfloat16)
    model = model.eval()
    model.compile(backend="tt")

    with torch.no_grad():
        output = model(sample_img)

    print(output)

def run_sdxl_vae_encoder():
    xr.set_device_type("TT")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    sample_img = torch.randn(1, 3, 512, 512, dtype=torch.bfloat16)
    model = pipeline.vae.encoder
    model = model.to(torch.bfloat16)
    model = model.eval()
    model.compile(backend="tt")

    device = xm.xla_device()

    sample_img = sample_img.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(sample_img)

    print(output)

def temp():
    xr.set_device_type("TT")

    torch.random.manual_seed(0)
    
    # Load only the UNet component instead of the entire pipeline
    model = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet",
        torch_dtype=torch.bfloat16,
        variant="fp16"
    )
    
    # Generate inputs for SDXL UNet
    # sample: (batch_size, in_channels=4, height, width)
    sample = torch.rand((1, 4, 64, 64), dtype=torch.bfloat16)
    
    # timestep: scalar or tensor representing the denoising step
    timestep = torch.randint(0, 1000, (1,))
    
    # encoder_hidden_states: (batch_size, sequence_length, cross_attention_dim=2048)
    encoder_hidden_states = torch.rand((1, 77, 2048), dtype=torch.bfloat16)
    
    # SDXL requires additional conditioning arguments
    added_cond_kwargs = {
        "text_embeds": torch.rand((1, 1280), dtype=torch.bfloat16),  # Pooled text embeddings
        "time_ids": torch.rand((1, 6), dtype=torch.bfloat16)  # Time conditioning
    }

    model = model.eval()

    with torch.no_grad():
        torch_output = model(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    
    torch_output = torch_output[0].cpu()


    model.compile(backend="tt")

    device = xm.xla_device()

    # Move inputs and model to device
    sample = sample.to(device)
    timestep = timestep.to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    added_cond_kwargs["text_embeds"] = added_cond_kwargs["text_embeds"].to(device)
    added_cond_kwargs["time_ids"] = added_cond_kwargs["time_ids"].to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

    output = output[0].cpu()

    print(f"torch.allclose(torch_output, output, atol=1e-4): {torch.allclose(torch_output, output, atol=1e-4)}")
    print(f"compute_pcc(torch_output, output): {compute_pcc(torch_output, output)}")



if __name__ == "__main__":
    temp()