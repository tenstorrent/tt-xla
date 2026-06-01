# Layer-0 PCC experiments (what each number means)

## Three different things people call "the same sanity"

| # | What runs | Inputs | Compare | Typical `self_attn` PCC |
|---|-----------|--------|---------|-------------------------|
| **A** | tt-xla `test_layer0_ln_attn_no_dep_pro_1b` | saved fixtures | **Forge** vs **CPU**, **separate bundles** | **~0.99** |
| **B** | `capture_forge.py` / `capture_ttnn.py` | same graph | offline artifact PCC | Forge ≈ TTNN ≈ CPU |
| **C** | `graph_0/main.py` on tt-metal | codegen **tensorbins** | **TTNN** vs **CPU golden** | **~0.99** |
| **D** | `compare.py` offline | saved `.pt` files | Forge artifact vs TTNN artifact | **~1.0** |

### Invalid compare (do not use)

| What | Problem | Fake `self_attn` PCC |
|------|---------|---------------------|
| One module, `run_op_test` CPU then Forge | CPU forward mutates shared ``past_key_values`` | **~0.77** |

Use ``run_decoder_stacked_stage_forge_vs_cpu_isolated`` / ``op_test.run_layer0_ln_attn_forge_vs_cpu_isolated``.

## How to read results

- **A ~0.99**: Forge on device matches eager CPU (fair).
- **D ~1.0**: Export matches live Forge capture.
- **C ~0.99**: TTNN replay matches CPU golden on tt-metal.

A real bug is **low PCC with isolated compare**, not the old shared-module ~0.77.

## Scripts

- Sanity: ``pytest … test_layer0_ln_attn_no_dep_pro_1b``
- CLI: ``python janus_layer0_forge_vs_ttnn_compare/run_cpu_vs_forge_sanity.py``
- KV repro (educational): ``python janus_layer0_forge_vs_ttnn_compare/repro_kv_order.py``
