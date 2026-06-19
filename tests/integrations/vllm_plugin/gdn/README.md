# GDN op tests — two-machine FLA golden workflow

Validates the pure-PyTorch Gated Delta Net ops in
`integrations/vllm_plugin/vllm_tt/layers/gdn/` against the standalone
`flash-linear-attention` (FLA) kernels.

Machines:
- **G** — GPU + `flash-linear-attention` installed, no TT card.
- **T** — TT card, no GPU. Slow link to G → goldens are transferred once, offline.

## 1. Generate goldens on G

```bash
pip install flash-linear-attention            # the canonical reference
python tests/integrations/vllm_plugin/gdn/gen_goldens.py --cuda \
    --out tests/integrations/vllm_plugin/gdn/golden/gdn_golden.pt
```

`gen_goldens.py` builds a parametrized grid, runs the FLA kernels, and saves
**inputs + golden outputs/states** (inputs are saved, not just seeded, so the
consumer never depends on cross-machine RNG). For the delta-rule kernels it
auto-detects FLA's recurrent-state layout by cross-checking against the
framework-agnostic reference in `_reference.py`, and stamps the bundle with the
`torch`/`fla` versions. Each case records its `source` (`fla` vs `ref`).

## 2. Transfer the bundle to T

The committed small bundle lives at `golden/gdn_golden.pt`. Large full-layer
goldens are git-ignored — `rsync` them once:

```bash
rsync gdn_golden_full.pt T:/.../gdn/golden/
```

## 3. Consume on T (or any CPU box)

```bash
# CPU math check (no hardware):
pytest -v tests/integrations/vllm_plugin/gdn/test_gdn_ops.py
# On the TT device:
GDN_TEST_DEVICE=tt pytest -v tests/integrations/vllm_plugin/gdn/test_gdn_ops.py
```

Thresholds: fp32 PCC ≥ 0.999, bf16 PCC ≥ 0.99 (chunk-sequential accumulation
order differs from FLA). If `golden/gdn_golden.pt` is absent the test regenerates
**reference** goldens in-process on CPU (no FLA required) so the math is still
exercised; the `source` tag in each test id shows `fla` vs `ref`.

## End-to-end coherence

After the per-op tests pass on T, validate the wired layer:

```bash
pytest -svv tests/integrations/vllm_plugin/generative/test_mrope.py
```

with `max_num_seqs=1` and a short `max_model_len` first, asserting coherence
(`assert_output_coherent`), then scale up.
