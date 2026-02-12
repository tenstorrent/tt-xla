import gc

import torch
from diffusers import AutoencoderKL, ZImageTransformer2DModel
from transformers import AutoModel, AutoTokenizer


def load_text_encoder():
    tokenizer = AutoTokenizer.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="tokenizer"
    )
    text_encoder = AutoModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    )
    text_encoder.eval()
    return text_encoder, tokenizer


def load_text_encoder_inputs(tokenizer):
    prompt = "A cat sitting on a mat"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }


def load_transformer():
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.eval()
    return transformer


def load_transformer_inputs():
    latent = torch.randn(16, 1, 128, 128, dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)
    cap_feat = torch.randn(10, 2560, dtype=torch.bfloat16)
    return {
        "x": [latent],
        "t": timestep,
        "cap_feats": [cap_feat],
    }


def load_vae_decoder():
    vae = AutoencoderKL.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae.eval()
    return vae.decoder


def load_vae_decoder_inputs():
    latents = torch.randn(1, 16, 128, 128, dtype=torch.bfloat16)
    return {"latents": latents}

if __name__ == "__main__":
    import torch_xla
    import torch_xla.runtime as xr
    xr.set_device_type("TT")

    run_decoder = False
    run_transformer = True


    if run_decoder:
        decoder = load_vae_decoder()
        decoder.compile(backend="tt")
        decoder = decoder.to(torch_xla.device())
        inputs = load_vae_decoder_inputs()
        inputs["latents"] = inputs["latents"].to(torch_xla.device())
        with torch.no_grad():
            output = decoder(inputs["latents"])
        torch_xla.sync()

    if run_transformer:
        transformer = load_transformer()
        inputs = load_transformer_inputs()

        # eager
        with torch.no_grad():
            output = transformer(inputs["x"], inputs["t"], inputs["cap_feats"])

        transformer.compile(backend="tt")
        transformer = transformer.to(torch_xla.device())
        inputs["x"][0] = inputs["x"][0].to(torch_xla.device())
        inputs["t"] = inputs["t"].to(torch_xla.device())
        inputs["cap_feats"][0] = inputs["cap_feats"][0].to(torch_xla.device())
        with torch.no_grad():
            output = transformer(inputs["x"], inputs["t"], inputs["cap_feats"])
        torch_xla.sync()

    print(output)
