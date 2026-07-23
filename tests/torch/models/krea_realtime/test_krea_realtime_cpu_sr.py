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


def test_kera():
    repo_id = "krea/krea-realtime-video"
    pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True)
    pipe.load_components(
        trust_remote_code=True,
        # device_map="cuda",
        # torch_dtype={"default": torch.bfloat16, "vae": torch.float16},
        device_map="cpu",
        torch_dtype={"default": torch.bfloat16, "vae": torch.bfloat16},
    )

    # krea's remote code hardcodes CUDA in sinusoidal_embedding_1d. The transformer
    # (CausalWanModel) calls it as a module global, so patch it in the transformer's
    # own module. Must run after load_components() has imported the remote code.
    importlib.import_module(
        type(pipe.transformer).__module__
    ).sinusoidal_embedding_1d = _sinusoidal_embedding_1d_cpu

    # for block in pipe.transformer.blocks:
    #     block.self_attn.fuse_projections()
    
    logger.info("\n=== PIPE COMPONENTS ===")
    logger.info(f"pipe.vae:\n{pipe.vae}\n")
    logger.info(f"pipe.transformer:\n{pipe.transformer}\n")
    logger.info(f"pipe.text_encoder:\n{pipe.text_encoder}\n")
    logger.info(f"pipe.tokenizer:\n{pipe.tokenizer}\n")

    logger.info("\n=== UNIQUE PARAMETER DTYPES ===")
    for name in ["vae", "transformer", "text_encoder"]:
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "parameters"):
            continue
        dtypes = set()
        for param in component.parameters():
            dtypes.add(param.dtype)
        logger.info(f"pipe.{name} unique dtypes: {dtypes}")

    logger.info("\n=== UNIQUE BUFFER DTYPES ===")
    for name in ["vae", "transformer", "text_encoder"]:
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "buffers"):
            continue
        dtypes = set()
        for buf in component.buffers():
            dtypes.add(buf.dtype)
        logger.info(f"pipe.{name} unique buffer dtypes: {dtypes}")

    num_blocks = 1

    frames = []
    state = PipelineState()
    prompt = ["a cat sitting on a boat"]

    generator = torch.Generator(device=pipe.device).manual_seed(42)
    for block_idx in tqdm(range(num_blocks)):
        state = pipe(
            state,
            prompt=prompt,
            num_inference_steps=4,
            num_blocks=num_blocks,
            block_idx=block_idx,
            generator=generator,
        )
        frames.extend(state.values["videos"][0])

    export_to_video(frames, "output_single_forward_pass.mp4", fps=24)
