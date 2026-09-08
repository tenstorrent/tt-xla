# DiffusionGemma encoder PCC probes

Tooling behind the root cause in
[tt-xla#6054](https://github.com/tenstorrent/tt-xla/issues/6054).
Each probe runs the encoder on CPU and TT with **identical inputs** and reports
PCC at every stage, so a cliff (op fault) is distinguishable from a gradual
slide (accumulation).

## Finding

| input | tokens | seed PCC | final PCC |
|---|---|---|---|
| text | 15 | 1.000000 | 0.985059 |
| text | **277** | 1.000000 | **0.908314** |
| image | 277 | 1.000000 *(CPU embeds injected)* | 0.937430 |
| image | 277 | 0.994693 *(real)* | 0.873028 |

Sequence length is the primary driver (**−0.077**, text only); the vision
front-end seed error is secondary (**−0.064**). At matched length and seed the
image path is *better* than text, so there is no image-content penalty.

## Scripts

| script | purpose |
|---|---|
| `probe_layers.py` | per-stage PCC on the image path (pooler + 31 hidden states) |
| `probe_layers2.py` | same, parameterised by `PROBE_VARIANT` (e.g. `ENCODER`) |
| `probe_len.py` | adds `PROBE_TARGET_TOKENS` — grows a text prompt to a token count, for length-controlled runs |
| `probe_inject.py` | injects the CPU golden's merged embeddings so the image path starts at PCC 1.0, isolating the seed error |

```bash
source venv/activate
python tools/diffusiongemma_pcc_probes/probe_layers.py                                   # image, 277 tok
PROBE_VARIANT=ENCODER python tools/diffusiongemma_pcc_probes/probe_layers2.py            # text, 15 tok
PROBE_VARIANT=ENCODER PROBE_TARGET_TOKENS=277 python tools/diffusiongemma_pcc_probes/probe_len.py
python tools/diffusiongemma_pcc_probes/probe_inject.py                                   # seed-error isolation
```

Requires an 8-device `n300-llmbox`. Each run loads the encoder twice (CPU golden
+ TT sharded), ~47 GiB each, so expect it to be slow.

## Caveats

- `probe_inject.py` prints **both** outcome legends unconditionally at the end.
  Neither is a computed verdict — read the numbers.
- Labels are offset: printed `layer 00` is the post-merge embedding, `layer 01`
  onward are text layers 0…29.
- The 277-token text prompt is a repeated sentence, so the +0.029
  image-vs-text margin is directional; the length effect itself is not in doubt.
- At 277 tokens the curve is **not monotonic** (−0.066 at layer 26, +0.090 at
  layer 29). Unexplained, and deliberately out of scope for #6054.
