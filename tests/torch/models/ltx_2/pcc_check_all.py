# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC validation for every LTX-2 submodule.

For each component:
  1. Create model + inputs on CPU
  2. Run forward on CPU -> reference output
  3. Move model + inputs to TT device
  4. Compile + run on TT -> device output
  5. Compute PCC between CPU and TT outputs

Usage:
  cd /root/tt-xla && source venv/activate
  cd tests/torch/models/ltx_2
  python pcc_check_all.py
"""

import sys, os, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch_xla
import torch_xla.runtime as xr


def compute_pcc(cpu_tensor, tt_tensor, name=""):
    """Compute Pearson Correlation Coefficient between two tensors."""
    a = cpu_tensor.detach().float().flatten()
    b = tt_tensor.detach().cpu().float().flatten()
    if a.numel() == 0:
        return 1.0
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    max_diff = (a - b).abs().max().item()
    print(f"  {name}: PCC={pcc:.6f}, max_diff={max_diff:.4f}")
    return pcc


# ── helpers ──────────────────────────────────────────────────────────
def setup_tt():
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    return torch_xla.device()


def run_on_tt(model, inputs, device, compile_kw=None):
    """Move model to device, compile, run, return output on CPU."""
    model_tt = model.to(device)
    kw = compile_kw or {}
    compiled = torch.compile(model_tt, backend="tt", **kw)
    with torch.no_grad():
        out = compiled(*inputs) if isinstance(inputs, (list, tuple)) else compiled(**inputs)
    torch_xla.sync(wait=True)
    return out


# ── 1. Vocoder ───────────────────────────────────────────────────────
def check_vocoder(device):
    print("\n=== Vocoder ===")
    from diffusers.pipelines.ltx2 import LTX2Vocoder
    model = LTX2Vocoder.from_pretrained(
        "Lightricks/LTX-2", subfolder="vocoder", torch_dtype=torch.bfloat16,
    ).eval()
    x = torch.randn(1, 2, 100, 64, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = model(x)

    out = run_on_tt(model, [x.to(device)], device)
    return compute_pcc(ref, out, "vocoder")


# ── 2. Audio VAE Encoder ────────────────────────────────────────────
def check_audio_vae_encoder(device):
    print("\n=== Audio VAE Encoder ===")
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio
    vae = AutoencoderKLLTX2Audio.from_pretrained(
        "Lightricks/LTX-2", subfolder="audio_vae", torch_dtype=torch.bfloat16,
    )
    encoder = vae.encoder.eval()
    del vae
    x = torch.randn(1, 2, 100, 64, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = encoder(x)

    out = run_on_tt(encoder, [x.to(device)], device)
    return compute_pcc(ref, out, "audio_vae_enc")


# ── 3. Audio VAE Decoder ────────────────────────────────────────────
def check_audio_vae_decoder(device):
    print("\n=== Audio VAE Decoder ===")
    from diffusers.models.autoencoders import AutoencoderKLLTX2Audio
    vae = AutoencoderKLLTX2Audio.from_pretrained(
        "Lightricks/LTX-2", subfolder="audio_vae", torch_dtype=torch.bfloat16,
    )
    decoder = vae.decoder.eval()
    del vae
    x = torch.randn(1, 8, 25, 16, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = decoder(x)

    out = run_on_tt(decoder, [x.to(device)], device)
    return compute_pcc(ref, out, "audio_vae_dec")


# ── 4. Video VAE Encoder ────────────────────────────────────────────
def check_video_vae_encoder(device):
    print("\n=== Video VAE Encoder ===")
    from conv3d_decompose import patch_conv3d_to_conv2d
    patch_conv3d_to_conv2d()

    from diffusers import AutoencoderKLLTX2Video
    vae = AutoencoderKLLTX2Video.from_pretrained(
        "Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.bfloat16,
    )
    encoder = vae.encoder.eval()
    del vae
    x = torch.randn(1, 3, 9, 128, 128, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = encoder(x)

    out = run_on_tt(encoder, [x.to(device)], device)
    return compute_pcc(ref, out, "video_vae_enc")


# ── 5. Video VAE Decoder ────────────────────────────────────────────
def check_video_vae_decoder(device):
    print("\n=== Video VAE Decoder ===")
    # Conv3d decomposition already applied above
    from diffusers import AutoencoderKLLTX2Video
    vae = AutoencoderKLLTX2Video.from_pretrained(
        "Lightricks/LTX-2", subfolder="vae", torch_dtype=torch.bfloat16,
    )
    decoder = vae.decoder.eval()
    del vae
    x = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = decoder(x)

    out = run_on_tt(decoder, [x.to(device)], device)
    return compute_pcc(ref, out, "video_vae_dec")


# ── 6. Latent Upsampler ─────────────────────────────────────────────
def check_latent_upsampler(device):
    print("\n=== Latent Upsampler ===")
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    model = LTX2LatentUpsamplerModel.from_pretrained(
        "Lightricks/LTX-2", subfolder="latent_upsampler", torch_dtype=torch.bfloat16,
    ).eval()
    x = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = model(x)

    out = run_on_tt(model, [x.to(device)], device)
    return compute_pcc(ref, out, "latent_upsampler")


# ── 7. Text Connectors ──────────────────────────────────────────────
def check_text_connectors(device):
    print("\n=== Text Connectors ===")
    from ltx2_patches import patch_attention_processor
    patch_attention_processor()

    from diffusers.pipelines.ltx2 import LTX2TextConnectors
    model = LTX2TextConnectors(
        caption_channels=3840, text_proj_in_factor=3,
        video_connector_num_attention_heads=30, video_connector_attention_head_dim=128,
        video_connector_num_layers=2, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=30, audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2, audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=4096, rope_theta=10000.0, rope_double_precision=True,
        causal_temporal_positioning=False, rope_type="interleaved",
    ).to(torch.bfloat16).eval()

    text_h = torch.randn(1, 64, 3840 * 3, dtype=torch.bfloat16)
    mask = torch.ones(1, 64, dtype=torch.long)

    with torch.no_grad():
        ref_v, ref_a, ref_m = model(text_h, mask)

    model_tt = model.to(device)
    compiled = torch.compile(model_tt, backend="tt", fullgraph=True)
    with torch.no_grad():
        tt_v, tt_a, tt_m = compiled(text_h.to(device), mask.to(device))
    torch_xla.sync(wait=True)

    pcc_v = compute_pcc(ref_v, tt_v, "connectors_video")
    pcc_a = compute_pcc(ref_a, tt_a, "connectors_audio")
    return min(pcc_v, pcc_a)


# ── 8. Text Encoder (2-layer minimal) ───────────────────────────────
def check_text_encoder(device):
    print("\n=== Text Encoder (Gemma3 2-layer) ===")
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

    config = Gemma3TextConfig(
        hidden_size=3840, intermediate_size=15360, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=8, head_dim=256,
        vocab_size=262208, sliding_window=None, use_cache=False,
    )
    model = Gemma3TextModel(config).to(torch.bfloat16).eval()

    torch.manual_seed(42)
    ids = torch.randint(0, 262208, (1, 64), dtype=torch.long)
    causal = torch.triu(
        torch.full((64, 64), float("-inf"), dtype=torch.bfloat16), diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=causal, output_hidden_states=True)
    ref_last = ref.hidden_states[-1]

    model_tt = model.to(device)
    compiled = torch.compile(model_tt, backend="tt")
    with torch.no_grad():
        tt_out = compiled(input_ids=ids.to(device), attention_mask=causal.to(device),
                          output_hidden_states=True)
    torch_xla.sync(wait=True)
    tt_last = tt_out.hidden_states[-1]

    return compute_pcc(ref_last, tt_last, "text_encoder_last_hidden")


# ── 9. Transformer (4-layer minimal) ────────────────────────────────
def check_transformer(device):
    print("\n=== Transformer (4-layer DiT) ===")
    from ltx2_patches import patch_attention_processor, patch_view_of
    patch_attention_processor()
    patch_view_of()

    from diffusers import LTX2VideoTransformer3DModel

    model = LTX2VideoTransformer3DModel(
        num_layers=4, rope_type="split",
    ).to(torch.bfloat16).eval()

    torch.manual_seed(42)
    B, nv, na, tl = 1, 32, 8, 16
    kwargs = dict(
        hidden_states=torch.randn(B, nv, 128, dtype=torch.bfloat16),
        audio_hidden_states=torch.randn(B, na, 128, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(B, tl, 3840, dtype=torch.bfloat16),
        audio_encoder_hidden_states=torch.randn(B, tl, 3840, dtype=torch.bfloat16),
        timestep=torch.tensor([500], dtype=torch.long),
        encoder_attention_mask=torch.ones(B, tl, dtype=torch.long),
        audio_encoder_attention_mask=torch.ones(B, tl, dtype=torch.long),
        num_frames=2, height=4, width=4, audio_num_frames=na,
    )

    with torch.no_grad():
        ref = model(**kwargs)
    ref_v, ref_a = ref.sample, ref.audio_sample

    # Wrapper to clone outputs (avoids view_of aliasing)
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, **kw):
            o = self.m(**kw)
            return o.sample.clone(), o.audio_sample.clone()

    model_tt = model.to(device)
    wrapper = Wrapper(model_tt)
    compiled = torch.compile(wrapper, backend="tt", fullgraph=True)

    kw_tt = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    with torch.no_grad():
        tt_v, tt_a = compiled(**kw_tt)
    torch_xla.sync(wait=True)

    pcc_v = compute_pcc(ref_v, tt_v, "transformer_video")
    pcc_a = compute_pcc(ref_a, tt_a, "transformer_audio")
    return min(pcc_v, pcc_a)


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = setup_tt()
    results = {}

    tests = [
        ("Vocoder", check_vocoder),
        ("Audio VAE Encoder", check_audio_vae_encoder),
        ("Audio VAE Decoder", check_audio_vae_decoder),
        ("Video VAE Encoder", check_video_vae_encoder),
        ("Video VAE Decoder", check_video_vae_decoder),
        ("Latent Upsampler", check_latent_upsampler),
        ("Text Connectors", check_text_connectors),
        ("Text Encoder", check_text_encoder),
        ("Transformer", check_transformer),
    ]

    for name, fn in tests:
        try:
            t0 = time.time()
            pcc = fn(device)
            elapsed = time.time() - t0
            status = "PASS" if pcc > 0.95 else "FAIL"
            results[name] = (pcc, status, elapsed)
            print(f"  [{status}] {name}: PCC={pcc:.6f} ({elapsed:.0f}s)")
        except Exception as e:
            results[name] = (0.0, "ERROR", 0)
            print(f"  [ERROR] {name}: {type(e).__name__}: {str(e)[:200]}")

        # Force cleanup between tests
        gc.collect()
        torch_xla.sync(wait=True)

    print("\n" + "=" * 60)
    print("PCC SUMMARY")
    print("=" * 60)
    for name, (pcc, status, elapsed) in results.items():
        print(f"  [{status:5s}] {name:25s} PCC={pcc:.6f}  ({elapsed:.0f}s)")
    print("=" * 60)

    all_pass = all(s == "PASS" for _, (_, s, _) in results.items())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
