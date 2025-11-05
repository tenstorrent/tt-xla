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


def run_vae_decoder():
    xr.set_device_type("TT")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16
    )
    # VAE decoder takes latent representation (batch_size, latent_channels=4, latent_height=64, latent_width=64)
    # SDXL VAE has a latent dimension of 4 channels and spatial compression factor of 8 (512/8 = 64)
    latent = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    model = vae.decoder
    model = model.to(torch.bfloat16)
    model = model.eval()

    model.compile(backend="tt")

    device = xm.xla_device()

    latent = latent.to(device)
    model = model.to(device)
    #model.eval()

    with torch.no_grad():
        output = model(latent)

    print(output)


def run_vae_encoder():
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

def run_vae_decoder_direct():
    xr.set_device_type("TT")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16
    ) 
    # VAE decoder takes latent representation (batch_size, latent_channels=4, latent_height=64, latent_width=64)
    # SDXL VAE has a latent dimension of 4 channels and spatial compression factor of 8 (512/8 = 64)
    latent = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    model = vae.decoder
    model = model.to(torch.bfloat16)
    model = model.eval()

    with torch.no_grad():
            torch_output = model(latent)

    model.compile(backend='tt')

    device = xm.xla_device()

    latent = latent.to(device)
    model = model.to(device)
    #model.eval()

    with torch.no_grad():
        output = model(latent)

    print(f"--------------------------------")
    print(f"OUTPUT:")
    print(output)
    print(f"--------------------------------")
    print(f"torch.allclose(torch_output, output, atol=1e-2): {torch.allclose(torch_output.cpu(), output.cpu(), atol=1e-4)}")
    print(f"PCC between torch_output and output: {compute_pcc(torch_output.cpu(), output.cpu())}")
    print(f"--------------------------------")

class UpBlocksWrapper(nn.Module):
    """Wrapper module to test only the upsample blocks."""
    def __init__(self, up_blocks):
        super().__init__()
        self.up_blocks = up_blocks
    
    def forward(self, x):
        for block in self.up_blocks:
            x = block(x)
        return x


def run_vae_decoder_test_upsamples():
    """
    Test all upsample blocks together as a wrapper module to identify where precision breaks.
    This function compiles only the up_blocks together and tests them.
    """
    xr.set_device_type("TT")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16
    )
    
    latent = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    model = vae.decoder
    model = model.to(torch.bfloat16)
    model = model.eval()
    
    # Get input to up_blocks by running through conv_in (on CPU)
    print(f"Getting input to up_blocks via conv_in...")
    with torch.no_grad():
        up_blocks_input = model.conv_in(latent)
    
    print(f"========================================")
    print(f"Testing up_blocks (all blocks together)")
    print(f"========================================\n")
    
    # Get torch reference output for up_blocks
    print(f"Computing torch reference output...")
    with torch.no_grad():
        torch_output = up_blocks_input.clone()
        for block in model.up_blocks:
            torch_output = block(torch_output)
    
    # Create wrapper module for up_blocks
    up_blocks_wrapper = UpBlocksWrapper(model.up_blocks).to(torch.bfloat16).eval()
    
    # Compile once
    print(f"Compiling up_blocks_wrapper with backend='tt'...")
    up_blocks_wrapper = torch.compile(up_blocks_wrapper, backend='tt')
    
    # Move to device and run
    device = xm.xla_device()
    up_blocks_input_device = up_blocks_input.to(device)
    up_blocks_wrapper = up_blocks_wrapper.to(device)
    
    print(f"Running on TT device...")
    with torch.no_grad():
        tt_output = up_blocks_wrapper(up_blocks_input_device)
    
    tt_output_cpu = tt_output.cpu()
    
    print(f"========================================")
    print(f"Results:")
    print(f"========================================")
    pcc = compute_pcc(torch_output, tt_output_cpu)
    allclose_result = torch.allclose(torch_output, tt_output_cpu, atol=1e-2)
    print(f"PCC: {pcc:.6f}")
    print(f"allclose (atol=1e-2): {allclose_result}")
    print(f"Max diff: {torch.max(torch.abs(torch_output - tt_output_cpu))}")
    print(f"========================================")


def run_vae_decoder_test_midblock():
    """
    Test the mid_block separately to identify where precision breaks.
    This function compiles only the mid_block and tests it.
    """
    xr.set_device_type("TT")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16
    )
    
    latent = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    model = vae.decoder
    model = model.to(torch.bfloat16)
    model = model.eval()
    
    # Get input to mid_block by running through conv_in only (on CPU)
    # mid_block expects 512 channels output from conv_in
    print(f"Getting input to mid_block via conv_in...")
    with torch.no_grad():
        mid_blocks_input = model.conv_in(latent)
    
    print(f"========================================")
    print(f"Testing mid_block")
    print(f"========================================\n")
    
    # Get torch reference output for mid_block
    print(f"Computing torch reference output...")
    with torch.no_grad():
        torch_output = model.mid_block(mid_blocks_input)
    
    # Create wrapper module for mid_block
    class MidBlockWrapper(nn.Module):
        """Wrapper module to test only the mid block."""
        def __init__(self, mid_block):
            super().__init__()
            self.mid_block = mid_block
        
        def forward(self, x):
            return self.mid_block(x)
    
    mid_block_wrapper = MidBlockWrapper(model.mid_block).to(torch.bfloat16).eval()
    
    # Compile once
    print(f"Compiling mid_block_wrapper with backend='tt'...")
    mid_block_wrapper = torch.compile(mid_block_wrapper, backend='tt')
    
    # Move to device and run
    device = xm.xla_device()
    mid_blocks_input_device = mid_blocks_input.to(device)
    mid_block_wrapper = mid_block_wrapper.to(device)
    
    print(f"Running on TT device...")
    with torch.no_grad():
        tt_output = mid_block_wrapper(mid_blocks_input_device)
    
    tt_output_cpu = tt_output.cpu()
    
    print(f"========================================")
    print(f"Results:")
    print(f"========================================")
    pcc = compute_pcc(torch_output, tt_output_cpu)
    allclose_result = torch.allclose(torch_output, tt_output_cpu, atol=1e-2)
    print(f"PCC: {pcc:.6f}")
    print(f"allclose (atol=1e-2): {allclose_result}")
    print(f"Max diff: {torch.max(torch.abs(torch_output - tt_output_cpu))}")
    print(f"========================================")


def run_groupnorm_test():
    xr.set_device_type("TT")
    model = nn.GroupNorm(32, 512)
    model = model.to(torch.bfloat16)
    model = model.eval()
    
    input = torch.randn(1, 512, 16, 16, dtype=torch.bfloat16) + 10.0
    
    # Get torch reference output (on CPU)
    print(f"Computing torch reference output...")
    with torch.no_grad():
        torch_output = model(input)
    
    # Compile with TT backend
    print(f"Compiling model with backend='tt'...")
    model = torch.compile(model, backend='tt')
    
    # Move to device and run
    device = xm.xla_device()
    input_device = input.to(device)
    model = model.to(device)
    
    print(f"Running on TT device...")
    with torch.no_grad():
        tt_output = model(input_device)
    
    tt_output_cpu = tt_output.cpu()
    
    print(f"========================================")
    print(f"GN Test Results:")
    print(f"========================================")
    pcc = compute_pcc(torch_output, tt_output_cpu)
    allclose_result = torch.allclose(torch_output, tt_output_cpu, atol=1e-2)
    print(f"PCC: {pcc:.6f}")
    print(f"allclose (atol=1e-2): {allclose_result}")
    print(f"Max diff: {torch.max(torch.abs(torch_output - tt_output_cpu))}")
    print(f"========================================")


if __name__ == "__main__":
    run_groupnorm_test()