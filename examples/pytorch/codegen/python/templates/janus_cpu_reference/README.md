# CPU reference (matches tt-xla codegen)

CPU golden is **`Layer0LnAttnNoDep`** from tt-xla — the **same module** passed to `codegen_py()` in
`examples/pytorch/codegen/python/janus_layer0_ln_attn_no_dep.py`.

## Requirements

- **transformers 5.5.1** in tt-metal `python_env` (same as tt-xla)
- **tt-xla** on disk — set `JANUS_TTXLA_ROOT` if not at `../../../31_may_yyz/tt-xla`
- **Fixtures** — `JANUS_LAYER0_FIXTURE_DIR` → `janus_logs/layer0_tensors/Pro_1B`

## Commands (tt-metal root)

```bash
source python_env/bin/activate
export JANUS_TTXLA_ROOT=/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
export JANUS_LAYER0_FIXTURE_DIR=/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla/janus_logs/layer0_tensors/Pro_1B

# optional: precompute golden
python janus_layer0_ln_attn_no_dep_codegen/cpu_reference/generate_golden.py

# TTNN + PCC table (calls compare at end of graph_0/main.py)
python janus_layer0_ln_attn_no_dep_codegen/graph_0/main.py 2>&1 | tee cpu_vs_tt.log
```

## Interpretation

| Comparison | Typical `self_attn` PCC |
|------------|-------------------------|
| **This CPU** vs **TTNN export** (`main.py`) | ~0.99 |
| **This CPU** vs **Forge device** (tt-xla pytest) | ~0.77 |

See `31_may_yyz/tt-xla/janus_layer0_forge_vs_ttnn_compare/EXPERIMENTS.md`.
