import importlib

import torch
from tqdm import tqdm
from diffusers.utils import export_to_video
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState
from loguru import logger


def _sinusoidal_embedding_1d_cpu(dim, position):
    """CPU-safe replacement for krea's sinusoidal_embedding_1d.

    The original (remote model.py) hardcodes
    ``device=torch.cuda.current_device()`` inside ``torch.arange(...)``, which
    raises "Torch not compiled with CUDA enabled" on a CPU-only build. This copy
    is identical except the arange is built on ``position.device``.
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, device=position.device, dtype=torch.float64).div(half),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def test_krea_realtime_cpu():

    repo_id = "krea/krea-realtime-video"
    pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
    pipe.load_components(
        trust_remote_code=True,
        torch_dtype={"default": torch.bfloat16, "vae": torch.bfloat16},
    )

    # krea's remote code hardcodes CUDA in sinusoidal_embedding_1d. The transformer
    # (CausalWanModel) calls it as a module global, so patch it in the transformer's
    # own module. Must run after load_components() has imported the remote code.
    importlib.import_module(
        type(pipe.transformer).__module__
    ).sinusoidal_embedding_1d = _sinusoidal_embedding_1d_cpu

    # krea's KV-cache context re-encoding (WanRTRecomputeKVCache, active from the
    # 3rd block onward) hardcodes .half() on the frames it feeds to vae.encode,
    # which clashes with the bf16 VAE. Wrap prepare_latents to re-cast the frames
    # to the VAE dtype so the conv input and weights match.
    _before_denoise = importlib.import_module(
        type(pipe.transformer).__module__.rsplit(".", 1)[0] + ".before_denoise"
    )
    _orig_prepare_latents = _before_denoise.WanRTRecomputeKVCache.prepare_latents

    def _prepare_latents_vae_dtype(self, components, frames):
        return _orig_prepare_latents(self, components, frames.to(components.vae.dtype))

    _before_denoise.WanRTRecomputeKVCache.prepare_latents = _prepare_latents_vae_dtype

    # for block in pipe.transformer.blocks:
    #     block.self_attn.fuse_projections()

    num_blocks = 9

    frames = []
    state = PipelineState()
    prompt = ["a cat sitting on a boat"]

    generator = torch.Generator(device=pipe.device).manual_seed(42)
    for block_idx in tqdm(range(num_blocks)):
        logger.info(f"Generating block {block_idx + 1}/{num_blocks} (block_idx={block_idx})")
        state = pipe(
            state,
            prompt=prompt,
            num_inference_steps=6,
            num_blocks=num_blocks,
            block_idx=block_idx,
            generator=generator,
        )
        frames.extend(state.values["videos"][0])

    export_to_video(frames, "krea_realtime_cpu_output.mp4", fps=24)
