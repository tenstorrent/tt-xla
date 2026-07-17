from diffusers import HunyuanImagePipeline
import torch
from loguru import logger


def test_HunyuanImage_2_1():

    repo = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"

    pipe = HunyuanImagePipeline.from_pretrained(repo, torch_dtype=torch.float32)

    logger.info("pipe={}", pipe)

    module_components = ["vae", "transformer", "text_encoder", "text_encoder_2"]
    all_components = module_components + [
        "tokenizer",
        "tokenizer_2",
        "scheduler",
        "guider",
        "ocr_guider",
    ]

    logger.info("\n=== PIPE COMPONENTS ===")
    for name in all_components:
        component = getattr(pipe, name, None)
        logger.info(f"pipe.{name}:\n{component}\n")

    logger.info("\n=== UNIQUE PARAMETER DTYPES ===")
    for name in module_components:
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "parameters"):
            continue
        dtypes = set()
        for param in component.parameters():
            dtypes.add(param.dtype)
        logger.info(f"pipe.{name} unique dtypes: {dtypes}")

    logger.info("\n=== UNIQUE BUFFER DTYPES ===")
    for name in module_components:
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "buffers"):
            continue
        dtypes = set()
        for buf in component.buffers():
            dtypes.add(buf.dtype)
        logger.info(f"pipe.{name} unique buffer dtypes: {dtypes}")

    logger.info("\n=== COMPONENT SIZES ===")
    for name in module_components:
        component = getattr(pipe, name, None)
        if component is None or not hasattr(component, "parameters"):
            continue
        num_params = 0
        for param in component.parameters():
            num_params += param.numel()
        num_buffers = 0
        if hasattr(component, "buffers"):
            for buf in component.buffers():
                num_buffers += buf.numel()
        logger.info(
            f"pipe.{name}: params={num_params / 1e9:.3f}B, "
            f"buffers={num_buffers / 1e9:.3f}B, "
            f"total={(num_params + num_buffers) / 1e9:.3f}B"
        )

    prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word 'Tencent' on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
    generator = torch.Generator().manual_seed(649151)
    out = pipe(
        prompt,
        num_inference_steps=8,
        distilled_guidance_scale=3.5,
        height=2048,
        width=2048,
        generator=generator,
    ).images[0]
    out.save("hyimage-distilled_output.png")
