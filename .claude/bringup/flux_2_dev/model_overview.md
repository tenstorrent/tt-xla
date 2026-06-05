# flux_2_dev / Dev-Vae (AutoencoderKLFlux2 decoder)

| Field             | Value |
|-------------------|-------|
| HF model ID       | black-forest-labs/FLUX.2-dev (subfolder `vae`) |
| Model class       | VAEDecoderWrapper wrapping diffusers.AutoencoderKLFlux2 |
| Task              | mm_image_ttt |
| Modality          | vision (image decode) |
| Parameters        | 84,046,115 (≈ 0.084 B) |
| Forward signature | `(z)` |
| Parallelism       | single_device (fits n150/p150) |

## Inputs
- `z` (latent): shape=(1, 32, 8, 8), dtype=torch.float32  — captured at 64x64

## Expected output (CPU forward, real weights, fp32)
- shape=(1, 3, 64, 64), dtype=torch.float32, all finite (CPU_SANITY_OK)

## References
- HF page : https://huggingface.co/black-forest-labs/FLUX.2-dev
- Paper   : none

## Bringup notes
- Source skill : model-bringup-overview
- Generated    : 2026-06-05 09:37
- Component test: tests/torch/models/flux_2_dev/test_vae_decoder.py::test_vae_decoder (PCC 0.99, fp32, opt_level=1)
- Runnable arch : p150 (Blackhole host; n150/Wormhole skipped by host probe)
