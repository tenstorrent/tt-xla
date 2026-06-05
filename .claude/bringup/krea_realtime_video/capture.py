# SPDX-License-Identifier: Apache-2.0
"""Offline I/O capture for krea/krea-realtime-video components.

Loads each component individually on CPU, monkey-patches its forward to log the
real input/output tensor shapes+dtypes (NO installed-source mutation), runs one
forward with synthetic inputs, and writes a JSON I/O spec to io_spec.json.
"""
import gc
import json
import logging
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "/proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla/.claude/bringup/krea_realtime_video/capture.log"
        ),
    ],
)
LOG = logging.getLogger("capture")

KREA = "krea/krea-realtime-video"
WAN = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DT = torch.bfloat16

# shape constants
MAX_SEQ_LEN = 512
TEXT_EMBED_DIM = 4096
NUM_CHANNELS_LATENTS = 16
NUM_LATENT_FRAMES = 3
NUM_FRAMES_PER_BLOCK = 3
LATENT_H = 60
LATENT_W = 104
UMT5_VOCAB = 256384
SEQ_LENGTH = 32760
FRAME_SEQ_LENGTH = 1560
KV_CACHE_NUM_FRAMES = 3
LOCAL_ATTN_SIZE = KV_CACHE_NUM_FRAMES + NUM_FRAMES_PER_BLOCK  # 6
KV_CACHE_SIZE = LOCAL_ATTN_SIZE * FRAME_SEQ_LENGTH  # 9360

spec = {}


def _shapes(obj):
    if torch.is_tensor(obj):
        return {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        return [_shapes(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _shapes(v) for k, v in obj.items()}
    return repr(type(obj).__name__)


def fixed_sinusoidal_embedding_1d(dim, position):
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


# ---------------------------------------------------------------- text_encoder
def cap_text_encoder():
    from transformers import UMT5EncoderModel

    LOG.info("loading text_encoder (UMT5EncoderModel) ...")
    m = UMT5EncoderModel.from_pretrained(
        WAN, subfolder="text_encoder", torch_dtype=DT, device_map="cpu"
    ).eval()
    input_ids = torch.randint(0, UMT5_VOCAB, (1, MAX_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.long)
    LOG.info("text_encoder FWD_IN %s", _shapes([input_ids, attention_mask]))
    with torch.no_grad():
        out = m(input_ids=input_ids, attention_mask=attention_mask)
    lhs = out.last_hidden_state
    LOG.info("text_encoder FWD_OUT last_hidden_state %s", _shapes(lhs))
    spec["text_encoder"] = {
        "class": "UMT5EncoderModel",
        "inputs": [
            {"name": "input_ids", "shape": list((1, MAX_SEQ_LEN)), "dtype": "torch.int64"},
            {"name": "attention_mask", "shape": list((1, MAX_SEQ_LEN)), "dtype": "torch.int64"},
        ],
        "outputs": [_shapes(lhs)],
        "called_per_step": False,
    }
    del m, out, lhs
    gc.collect()


# ---------------------------------------------------------------- transformer
def cap_transformer():
    from diffusers import AutoModel

    LOG.info("loading transformer (CausalWanModel, trust_remote_code) ...")
    t = AutoModel.from_pretrained(
        KREA, subfolder="transformer", torch_dtype=DT, device_map="cpu",
        trust_remote_code=True,
    ).eval()
    # patch CUDA-hardcoded sinusoidal embedding (tt-xla#4464)
    t.forward.__globals__["sinusoidal_embedding_1d"] = fixed_sinusoidal_embedding_1d
    for blk in t.blocks:
        blk.self_attn.local_attn_size = -1
        blk.self_attn.num_frame_per_block = NUM_FRAMES_PER_BLOCK
    num_blocks = len(t.blocks)
    num_heads = t.config.num_heads
    head_dim = t.config.dim // num_heads
    LOG.info("transformer blocks=%d heads=%d head_dim=%d", num_blocks, num_heads, head_dim)

    x = torch.randn(1, NUM_CHANNELS_LATENTS, NUM_LATENT_FRAMES, LATENT_H, LATENT_W, dtype=DT)
    tt = torch.full((1, NUM_FRAMES_PER_BLOCK), 1000.0, dtype=torch.float32)
    context = torch.randn(1, MAX_SEQ_LEN, TEXT_EMBED_DIM, dtype=DT)

    kv_shape = [1, KV_CACHE_SIZE, num_heads, head_dim]
    ca_shape = [1, MAX_SEQ_LEN, num_heads, head_dim]
    kv_cache = [
        {"k": torch.zeros(kv_shape, dtype=DT).contiguous(),
         "v": torch.zeros(kv_shape, dtype=DT).contiguous(),
         "global_end_index": 0, "local_end_index": 0}
        for _ in range(num_blocks)
    ]
    crossattn_cache = [
        {"k": torch.zeros(ca_shape, dtype=DT).contiguous(),
         "v": torch.zeros(ca_shape, dtype=DT).contiguous(),
         "is_init": False}
        for _ in range(num_blocks)
    ]
    LOG.info("transformer FWD_IN %s", _shapes([x, tt, context]))
    with torch.no_grad():
        out = t(x=x, t=tt, context=context, kv_cache=kv_cache, seq_len=SEQ_LENGTH,
                crossattn_cache=crossattn_cache, current_start=0, cache_start=None)
    LOG.info("transformer FWD_OUT %s", _shapes(out))
    spec["transformer"] = {
        "class": "CausalWanModel",
        "wrapper": "CausalWanWrapper",
        "inputs": [
            {"name": "x", "shape": list((1, NUM_CHANNELS_LATENTS, NUM_LATENT_FRAMES, LATENT_H, LATENT_W)), "dtype": "torch.bfloat16"},
            {"name": "t", "shape": list((1, NUM_FRAMES_PER_BLOCK)), "dtype": "torch.float32"},
            {"name": "context", "shape": list((1, MAX_SEQ_LEN, TEXT_EMBED_DIM)), "dtype": "torch.bfloat16"},
        ],
        "outputs": [_shapes(out)],
        "called_per_step": True,
        "num_blocks": num_blocks, "num_heads": num_heads, "head_dim": head_dim,
    }
    del t, out, kv_cache, crossattn_cache
    gc.collect()


# ---------------------------------------------------------------- vae decoder
def cap_vae():
    from diffusers import AutoencoderKLWan

    LOG.info("loading vae (AutoencoderKLWan) ...")
    v = AutoencoderKLWan.from_pretrained(
        WAN, subfolder="vae", torch_dtype=DT, device_map="cpu"
    ).eval()
    z = torch.randn(1, NUM_CHANNELS_LATENTS, NUM_LATENT_FRAMES, LATENT_H, LATENT_W, dtype=DT)
    LOG.info("vae FWD_IN z %s", _shapes(z))
    with torch.no_grad():
        out = v.decode(z, return_dict=False)[0]
    LOG.info("vae FWD_OUT %s", _shapes(out))
    spec["vae"] = {
        "class": "AutoencoderKLWan",
        "wrapper": "VAEDecoderWrapper",
        "inputs": [
            {"name": "z", "shape": list((1, NUM_CHANNELS_LATENTS, NUM_LATENT_FRAMES, LATENT_H, LATENT_W)), "dtype": "torch.bfloat16"},
        ],
        "outputs": [_shapes(out)],
        "called_per_step": False,
    }
    del v, out
    gc.collect()


if __name__ == "__main__":
    torch.manual_seed(0)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "text_encoder"):
        cap_text_encoder()
    if which in ("all", "vae"):
        cap_vae()
    if which in ("all", "transformer"):
        cap_transformer()
    out_path = "/proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla/.claude/bringup/krea_realtime_video/io_spec.json"
    # merge with any existing partial spec
    try:
        existing = json.load(open(out_path))
    except Exception:
        existing = {}
    existing.update(spec)
    json.dump(existing, open(out_path, "w"), indent=2)
    LOG.info("WROTE %s components=%s", out_path, list(existing.keys()))
