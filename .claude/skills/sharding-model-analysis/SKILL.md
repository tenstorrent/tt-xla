---
name: sharding-model-analysis
description: Analyze a PyTorch model, then design and implement an optimal multi-device sharding strategy for Tenstorrent hardware that minimizes collective-communication (CCL) ops. A strategy is both which parallelism to use (tensor / sequence / data) and how to shard under it - e.g. Megatron column→row is one of several tensor-parallel schemes. Use whenever the user asks to shard a model, distribute it across devices, write Shardy annotations, reduce CCLs, or analyze a model for tensor/sequence parallelism. Trigger even when the user only says "sharding strategy" or names a model with a device count, without asking for a full plan.
---

# Sharding Model Analysis

Shard a model across Tenstorrent devices to make **inference as fast as possible**. Speed comes from splitting compute across devices so they run in parallel - but every split that crosses a tensor boundary forces a collective-communication (CCL) op, and each CCL is a cross-device sync that stalls every device until it finishes. So the goal is to **parallelize as much compute as possible while paying the fewest CCL ops for it** (and the fewest bytes per CCL). Memory balance is **not** an objective: shard for speed first, and shard to save DRAM only when the speed-optimal layout can't fit on the hardware.

## Core rules

- **Optimize for speed, not memory.** The objective is fast inference - parallelize compute across devices and pay the minimum CCL cost for that parallelism. How evenly DRAM is balanced is irrelevant *unless* the model doesn't otherwise fit.
- **Minimize CCLs or their size.** Among the layouts that parallelize the compute, pick the one with the fewest CCLs/ fewest bytes moved, pay attention that sometimes several small CCLs can beat one large one.
- **Only shard for memory as a loud fallback.** If the speed-optimal sharding can't fit in DRAM, *then* switch to a memory-saving layout (e.g. activation/sequence-parallel) and flag the speed trade-off explicitly. Never silently pick a high-CCL or memory-first strategy.
- **One mesh shape per pipeline.** When a pipeline chains several models, they all share one mesh shape. Pick the shape that's optimal for the slowest (longest-running) model and make the rest adopt it; if some model can't run under that shape, fall back to a shape every model supports.
- **Cite every non-trivial decision** - Megatron paper section, an internal PR/discussion, or prior work - so the user can verify it.
- **Never write Shardy syntax from memory.** Read [references/shardy_sharding.md](references/shardy_sharding.md) first.
- **Run `source venv/activate`** before any repo command or test.

## Reference map - read on demand

| Need | Read |
|---|---|
| Enumerate a model's modules + param shapes | [references/run_analysis.md](references/run_analysis.md) |
| Standard mesh shapes per device count | [references/mesh_shapes.md](references/mesh_shapes.md) |
| Megatron / ZeRO background | [references/general_sharding.md](references/general_sharding.md) |
| Which collective each sharding pattern emits (CCL accounting) | [references/ccl_cheatsheet.md](references/ccl_cheatsheet.md) |
| Shardy annotation syntax | [references/shardy_sharding.md](references/shardy_sharding.md) |
| Video-gen VAE sharding (what the compiler supports today) | [references/video_vae.md](references/video_vae.md) |
| Video-gen DiT sharding (what the compiler supports today) | [references/video_dit.md](references/video_dit.md) |
| Known unsupported patterns / open compiler issues | [references/compiler_support.md](references/compiler_support.md) |
| Report skeleton to fill in | [assets/sharding_analysis_template.md](assets/sharding_analysis_template.md) |
| Live device count | run `scripts/num_devices.py` |

## Output

Maintain a single evolving report, **`sharding_analysis.md`**, in the user's working directory, started from [assets/sharding_analysis_template.md](assets/sharding_analysis_template.md). It grows phase by phase and ends as the clean final deliverable. It must contain:

- Model architecture (components and submodules)
- Recommended strategy, plus the rejected alternatives and why
- Per-component CCL accounting (count and approximate transfer size) and the model total
- Bottlenecks and any compiler limitations hit

Second deliverable: the user's model script with the sharding implemented.

## Workflow

Track these phases with `TodoWrite` and keep the user updated. Finish each phase before starting the next; pause for the user where noted or if you are unsure on how to continue.

**1 - Identify the model.** Get the user's load script and read it to pin down the exact model and variant. If you can't find it, ask. Do not proceed while the model is uncertain.

**2 - Map the architecture.** Enumerate every submodule and parameter shape using the recipes in [references/run_analysis.md](references/run_analysis.md) (`print`, `named_modules`, `named_parameters`). Research unfamiliar blocks in source or online. Record the architecture in `sharding_analysis.md`.

**3 - Fix the mesh shapes.** Run `scripts/num_devices.py` for the device count and map it to [references/mesh_shapes.md](references/mesh_shapes.md). Confirm with the user. A sharded dim must divide cleanly by its mesh-axis size, so keep every target mesh in mind - aim for one strategy that holds across all of them.

**4 - Design per-component sharding.** For each component, pick the layout that minimizes CCLs, then minimizes CCL bytes. Default to Megatron-style column→row pairs ([references/general_sharding.md](references/general_sharding.md)): one all-reduce per pair, intermediate never gathered. Use [references/ccl_cheatsheet.md](references/ccl_cheatsheet.md) to count the collective each choice emits, and check [references/compiler_support.md](references/compiler_support.md) before committing - don't design around a pattern that can't lower yet. For video models, start from [references/video_vae.md](references/video_vae.md) and [references/video_dit.md](references/video_dit.md) - they record current compiler support. After per-component choices, re-evaluate the whole model: components must compose without forcing extra reshards.

**5 - Write the plan and get sign-off.** Fill in `sharding_analysis.md`: the recommendation, per-component reasoning with rejected alternatives, per-component and total CCL count/size, and bottlenecks, all cited. **Then ask the user to approve before implementing.**

**6 - Implement with Shardy.** Read [references/shardy_sharding.md](references/shardy_sharding.md), then implement exactly the plan from `sharding_analysis.md`. Verify the code. Ask the user to add IR-export options and run the test:
```
export_path="path/to/model",
export_model_name="model_name"
```
Have them tell you when the run finishes so you can inspect the logs.

**7 - Verify against the IR.** Inspect both **TTIR** and **TTNN**; confirm the emitted sharding matches the plan. On a mismatch, isolate the offending module into a minimal test and repeat phases 6–7 on it to find the root cause; record findings in `sharding_analysis.md` and tell the user what's needed to enable it. If the model *runs* slowly (long compile is fine), a CCL op may be hanging - ask the user to enable debug logging:
```
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
export TT_RUNTIME_MEMORY_LOG_LEVEL=operation
```
and build tt-xla in debug mode:
```
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build
```
A stuck CCL means the op hangs: ask a user whether to open an issue for it and choose an alternative strategy (also note it inside the `sharding_analysis.md`).

**8 - Finalize the report.** Once everything works, clean up `sharding_analysis.md` into the final deliverable: architecture, per-module sharding, CCL count/size analysis, and the trade-offs that justify the chosen strategy.
