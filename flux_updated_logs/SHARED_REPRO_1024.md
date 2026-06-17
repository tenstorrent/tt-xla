# FLUX.2 shared-commit 1024 reproduction — resume on a fresh latest-`main` build

## Why we're here
The shared tt-forge-models branch `claude/black-forest-labs-flux-2-dev-27690487044`
(commit `9177e4ea14`) reports the denoiser running on 4x Blackhole `(1,4)` at
1024x1024, PCC 0.9884, prompt-faithful image.

On THIS workspace it OOMs at 1024 (`bank_manager.cpp:462`, ~250 MB/bank short of
32 GB) — and it OOMs **identically when running the shared branch's verbatim code**
(its own `test_multichip.py`, its own loaders/shard spec). The shard spec was
diffed byte-for-byte vs the shared branch: identical. Same 4 chips, same mesh,
same resolution. So the blocker is the **tt-mlir / tt-metal toolchain**, not the
spec/chips/resolution. Our pin here: tt-mlir 2026-06-09, tt-metal 2026-05-29.

## On the new latest-`main` build (newer toolchain) — repro steps
IMPORTANT: do NOT just `git checkout akannan/flux2_shared_repro` — that branch
carries the OLD tt-mlir pin. Instead keep latest-`main`'s toolchain and only
point tt_forge_models at the shared commit:

```bash
# in the fresh latest-main tt-xla (newer tt-mlir/tt-metal already built)
git -C third_party/tt_forge_models fetch origin \
    claude/black-forest-labs-flux-2-dev-27690487044
git -C third_party/tt_forge_models checkout 9177e4ea14

source venv/activate
export HF_TOKEN=...                       # FLUX.2 is gated
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export FLUX2_HEIGHT=1024 FLUX2_WIDTH=1024 FLUX2_STEPS=4
export FLUX2_OUT_DIR=$(pwd)/flux_updated_logs/shared_repro
mkdir -p "$FLUX2_OUT_DIR"
tt-smi -r
timeout 3600 python third_party/tt_forge_models/flux2/pytorch/test_multichip.py \
    2>&1 | tee flux_updated_logs/shared_repro_test_multichip_1024.log
# expect: "COMPOSITE SUCCESS -> .../generated.png" and a prompt-faithful hermit-crab image
```

Composite placement in that script: text encoder (24B) -> CPU, denoiser (32B) ->
TT (TP, mesh `(1,N)`), VAE -> CPU.

## If 1024 fits on the new toolchain → next steps (the original plan)
1. Confirm the generated image is prompt-faithful and grab the PCC from
   `probe_denoiser.py` at 1024.
2. Move the **text encoder** onto TT (24B; should fit sharded on 32 GB/chip
   Blackhole — it OOM'd only on the 12.85 GB/chip WH box). Run it as its own
   compiled graph, freed before the denoiser loop.
3. Move the **VAE** onto TT (replicate across the mesh — the shared loader already
   dropped the `(1,1)` special case so it runs on the full mesh).
4. Sequence components so they're not all resident at once (denoiser alone ~16 GB/chip).

## If 1024 still OOMs on the new toolchain
Fall back to `probe_denoiser.py` to find the max-fitting resolution
(`FLUX2_HEIGHT=896/768/...`) and report PCC + image there.
