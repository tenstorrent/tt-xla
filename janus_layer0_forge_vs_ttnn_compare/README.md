# Forge vs TTNN layer-0 LN+attn comparison

Same **saved decode fixtures** and stacked stages `[3, 2, 1, 2048]` as
`test_layer0_ln_attn_no_dep_pro_1b`, but compares **Forge/XLA device output** to
**exported TTNN `graph_0/main.py` output** (not only vs CPU).

## Artifacts

Written under `artifacts/` (override with `JANUS_COMPARE_ARTIFACTS_DIR`):

| File | Script | Environment |
|------|--------|-------------|
| `forge_stacked_pro1b.pt` | `capture_forge.py` | tt-xla venv + TT device |
| `ttnn_stacked_pro1b.pt` | `capture_ttnn.py` | tt-metal `python_env` + TT device |
| `cpu_stacked_pro1b.pt` | `capture_cpu.py` | optional (CPU only) |

## Steps

```bash
# 1) Forge (tt-xla)
cd /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
python janus_layer0_forge_vs_ttnn_compare/capture_forge.py

# 2) TTNN (tt-metal)
cd /proj_sw/user_dev/ctr-akannan/31_may_tt_metal/tt-metal
python /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla/janus_layer0_forge_vs_ttnn_compare/capture_ttnn.py

# Optional CPU column
python janus_layer0_forge_vs_ttnn_compare/capture_cpu.py

# 3) Offline report (no device)
cd /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
python janus_layer0_forge_vs_ttnn_compare/compare.py 2>&1 | tee forge_vs_ttnn.log
```

Set `JANUS_TTNN_GRAPH_DIR` if codegen is not under
`31_may_tt_metal/tt-metal/janus_layer0_ln_attn_no_dep_codegen/graph_0`.

## How to read results

- **Forge vs TTNN `self_attn` PCC ~0.99+** — export matches what Forge ran; the ~0.34 TTNN vs CPU gap is mostly CPU-reference / numerics vs both TT paths.
- **Forge vs TTNN `self_attn` PCC ~0.34** — TTNN export diverges from Forge (export bug or different op lowering).
- **Forge vs CPU ~0.77, TTNN vs CPU ~0.34** (known) — Forge is the “canonical” bad attn repro; TTNN export is worse.
