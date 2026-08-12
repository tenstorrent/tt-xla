


import pytest
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import HiDreamImagePipeline
from loguru import logger


# Per model card: guidance_scale / num_inference_steps differ per variant.
#   Fast -> 0.0 / 16, (Dev -> 0.0 / 28), Full -> 5.0 / 50
@pytest.mark.parametrize(
    "model_name, guidance_scale, num_inference_steps",
    [
        # ("HiDream-ai/HiDream-I1-Fast", 0.0, 16),
        ("HiDream-ai/HiDream-I1-Full", 5.0, 50),
    ],
)
def test_hidream_l1(model_name, guidance_scale, num_inference_steps):

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )

    pipe = HiDreamImagePipeline.from_pretrained(
        model_name,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    )

    logger.info("\n=== MODEL: {} ===", model_name)

    logger.info("pipe={}", pipe)

    module_components = ["vae", "transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"]
    all_components = module_components + ["tokenizer", "tokenizer_2", "tokenizer_3", "tokenizer_4", "scheduler"]

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

    logger.info(
        "\n=== MODEL: {} | guidance_scale={} | num_inference_steps={} ===",
        model_name, guidance_scale, num_inference_steps,
    )

    # image = pipe(
    #     'A cat holding a sign that says "HiDream.ai".',
    #     height=1024,
    #     width=1024,
    #     guidance_scale=guidance_scale,
    #     num_inference_steps=num_inference_steps,
    #     generator=torch.Generator().manual_seed(0),
    # ).images[0]

    # variant = model_name.split("-")[-1].lower()  # "fast" / "full"
    # image.save(f"HiDream-I1_{variant}_result.png")