# Layer-0 PCC experiments (what each number means)

## Three different things people call "the same sanity"

| # | What runs | Inputs | Compare | Typical `self_attn` PCC |
|---|-----------|--------|---------|-------------------------|
| **A** | tt-xla `test_layer0_ln_attn_no_dep_pro_1b` | `inputs_embeds` + KV on module only | **Forge (device)** vs **CPU eager** (`Layer0LnAttnNoDep.forward`) | **~0.77** (the PCC-drop repro) |
| **B** | `capture_forge.py` | same as A | saves `forge_stacked_*.pt` | (artifact) |
| **C** | `graph_0/main.py` on tt-metal | **`tensorbin` files** from codegen (`arg3`, `arg4`, `arg7`, …) | **TTNN export** vs **CPU golden** (`cpu_reference/forward.py`) | **~0.99** after transformers 5.5.1 align |
| **D** | `compare.py` offline | saved tensors | **Forge artifact** vs **TTNN artifact** | **~1.0** |

**A is not the same experiment as C.**

- **A** = live compiled PyTorch module on TT, one forward from decode embeds + KV (no frozen trace buffers).
- **C** = replay the **exported TTNN program** with **weights + activations dumped at codegen time** (`export_tensors: True`).

So **0.77 vs 0.99 does not mean the export is "wrong"** — you are measuring different pipelines.

## How to read results (logic)

If **D** says Forge ≈ TTNN (~1.0) but **A** says Forge ≠ CPU (~0.77):

- Export faithfully captures what Forge ran when codegen ran.
- The **~0.77 gap is Forge compile vs CPU eager** on the **Python module**, not "export vs CPU" on tt-metal.

If **C** says TTNN ≈ CPU (~0.99):

- CPU golden (fixtures + eager forward) matches the **exported trace replay** on tt-metal.
- That can still be true while **A** is ~0.77 if the **live Forge compile** diverges from both trace replay and CPU (unusual but possible — **re-run the matrix** below).

If **A** and **C** should match but don't, re-run everything in one session (script below).

## Canonical repro for the bug

Use **experiment A** (tt-xla pytest). That is the ~0.77 signature.

tt-metal `main.py` is for **export bring-up** (TTNN vs CPU golden), not for re-proving the Forge PCC drop.
