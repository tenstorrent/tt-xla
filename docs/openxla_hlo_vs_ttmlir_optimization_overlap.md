# OpenXLA HLO vs. tt-mlir (TTIR/TTNN) Optimization Pass Overlap

**Author:** Analysis prepared for the tt-xla compiler team
**Date:** 2026-06-25
**Question:** Do we run any OpenXLA HLO optimization passes today? And how much of tt-mlir's
TTIR/TTNN optimization work overlaps with OpenXLA's HLO-dialect passes — enough to delete tt-mlir
passes and instead bake those optimizations into the StableHLO graph via Torch-XLA?

---

## TL;DR

1. **Confirmed: tt-xla runs zero OpenXLA HLO-dialect optimization passes.** The PJRT plugin receives a
   VHLO/StableHLO MLIR module and hands it straight to tt-mlir (`StableHLO → TTIR → TTNN`). It never
   builds an `xla::HloModule`, never calls `RunHloPasses` / `HloPassPipeline`, and never lowers to the
   classic XLA HLO (or MHLO) dialect. Torch-XLA is explicitly configured to *emit* StableHLO
   (`XLA_STABLEHLO_COMPILE=1`) instead of going through an XLA backend compiler, so the HLO optimization
   pipeline that normally runs inside a backend's `RunHloPasses` is bypassed entirely. The understanding
   in the task is correct.

2. **The overlap with OpenXLA HLO passes is real but narrow and shallow.** It is concentrated in (a)
   generic infrastructure passes (canonicalize / CSE / DCE / inliner) that tt-mlir already gets for free
   from upstream MLIR, and (b) algebraic / tensor-manipulation (TM) simplification, which already has a
   *StableHLO-dialect* equivalent wired into tt-mlir's own StableHLO pipeline. The high-value, expensive
   tt-mlir passes — the TTNN Optimizer (memory layout, L1 sharding, op-config search, spill management),
   layout assignment, tilize/untilize, hardware workarounds, fused-kernel matching (conv+bias, SDPA,
   RoPE), deallocation — have **no OpenXLA HLO analogue** because OpenXLA's comparable passes are
   CPU/GPU-codegen specific.

3. **Recommendation: do not pursue "delete tt-mlir passes, offload to OpenXLA HLO."** The ROI is low and
   there are hard architectural blockers (the JAX path, tt-mlir's role as a standalone multi-frontend
   compiler, and the HLO→StableHLO round-trip). If pre-simplification of the incoming graph is desired,
   the right lever is the **StableHLO-dialect simplification passes already present in tt-mlir's StableHLO
   pipeline** (`stablehlo-aggressive-simplification`, `stablehlo-aggressive-folder`, canonicalizer), not
   OpenXLA HLO. Details and reasoning below.

---

## Part 1 — Confirmation: we do not run OpenXLA HLO optimization passes

### The compile call chain (tt-xla → tt-mlir)

The PJRT plugin's compile entry hard-requires an **MLIR** program format and rejects anything else
(an HLO proto is not accepted):

```
onClientCompile()                       pjrt_implementation/src/api/client_instance.cc:831
  └─ rejects program_format != "mlir"   client_instance.cc:847-855
  └─ ClientInstance::compileMlirProgram  client_instance.cc:478
       └─ ModuleBuilder::buildModule     pjrt_implementation/src/api/module_builder/module_builder.cc:297
            1. createVHLOModule              :322   parse bytes as MLIR (VHLO)
            2. convertFromVHLOToSHLO          :330   StableHLO deserialize pipeline (upstream StableHLO)
            3. runFrontendSHLOPipeline        :336   tt-xla arg-attribute annotation
            4. runCompilerStableHLOPipeline   :374   tt-mlir createStableHLOPipeline
            5. convertFromSHLOToTTIR          :407   tt-mlir createStableHLOToTTIRPipeline  ← StableHLO→TTIR
            6. convertFromTTIRToTTNN          :427   tt-mlir createTTIRToTTNNCommonPipeline
            7. flatbuffer / codegen emission
```

The input arrives as **VHLO (versioned StableHLO)**, is deserialized to StableHLO, and the first
*structural* transformation is tt-mlir's `createConvertStableHLOToTTIRPass`
(`TTIRPipelines.cpp:83`). There is no HLO-dialect stage anywhere in this path.

### Evidence that no XLA HLO optimization runs

- A whole-repo grep (excluding `third_party/`, `build/`, `venv/`) for `RunHloPasses`, `xla::HloModule`,
  `HloPassPipeline`, `HloModuleProto`, `stablehlo-legalize-to-hlo`, `ConvertHloToStableHlo`,
  `GpuCompiler`/`CpuCompiler` returns **zero matches**. The XLA backend-compiler `RunHloPasses` stage —
  where AlgebraicSimplifier, ConstantFolding, fusion, layout assignment, etc. normally run — is never
  invoked.
- **Torch-XLA path:** `python_package/torch_plugin_tt/__init__.py:38` sets
  `os.environ["XLA_STABLEHLO_COMPILE"] = "1"` with the comment that it is "required for the TT PJRT plugin
  to work correctly … to enable stablehlo compilation." This makes Torch-XLA emit a StableHLO MLIR program
  to PJRT `Compile` (format `"mlir"`) rather than an `HloModuleProto` for backend optimization. The "tt"
  dynamo backend (`python_package/tt_torch/backend/backend.py:346`) does only FX-graph-level work
  (decompositions, composite handling, `torch.export`) before Torch-XLA traces to StableHLO; it never
  calls an XLA backend compiler.
- **JAX path:** `python_package/jax_plugin_tt/__init__.py:23` registers the plugin via
  `xb.register_plugin("tt", ...)`. JAX's `jit` lowers to StableHLO and hands the module to PJRT. The
  monkeypatch only adds MLIR-level StableHLO composite lowerings (e.g. `mark_weight`, `gelu`).
- The only `mhlo` references in the plugin are **named attribute strings** (`mhlo.num_partitions`,
  `mhlo.spmd_output_sharding`) used to read sharding/device metadata, plus a "clean for XLA ingestion"
  pass that runs on a **discarded clone** of the module purely to extract Shardy/GSPMD output-sharding
  strings (`module_builder.cc:384`). Neither lowers to the HLO dialect or runs an HLO pass.

### Verified against the custom Torch-XLA fork (`/localdev/hshah/pytorch-xla`, `tenstorrent/pytorch-xla`)

The fork was inspected directly to rule out fork-specific HLO optimization. Findings:

- `RunHloPasses`, `HloPassPipeline`, `HloPassFix`, and `AlgebraicSimplifier` have **zero** matches
  anywhere in the fork's C++ — the XLA backend HLO-optimization pipeline is never assembled or run.
- The compile path used by tt-xla is `PjRtComputationClient::Compile`
  (`torch_xla/csrc/runtime/pjrt_computation_client.cpp:758`): when `XLA_STABLEHLO_COMPILE` is set it
  calls `ConvertHloToStableHlo(instance.computation.mutable_proto(), ...)` and then
  `client_->CompileAndLoad(mlir_module, ...)` (`:763,770`) — i.e. it converts the **raw, unoptimized
  HloModuleProto produced by the XLA builder** straight to StableHLO and forwards it to the TT PJRT
  plugin. The backend `Compile` that would normally run `RunHloPasses` belongs to the TT plugin (tt-mlir),
  not XLA.
- The entire HLO→StableHLO conversion (`mhloToStablehloHelper`,
  `torch_xla/csrc/runtime/stablehlo_helper.cpp:70-100`) runs exactly these passes, **none of which is an
  HLO optimization pass**:
  1. `CreatePrepareXlaMlirDebuginfoPass` (`:73`) — debug-info plumbing
  2. `mhlo::createLegalizeDotToDotGeneralPass` (`:80`) — legalization workaround (`dot`→`dot_general`)
  3. `mhlo::createExpandHloTuplesPass` (`:83`) — flatten tuple outputs (structural legalization)
  4. `createCanonicalizerPass` (`:85`) — generic MLIR canonicalize (cleanup after tuple flatten)
  5. `mhlo::createHloLegalizeToStablehloPass` (`:86`) — MHLO→StableHLO dialect **translation**
  6. `CreateBuildStableHLOCompositePass` (`:88`) — group composite ops (e.g. gelu) — torch-xla custom
  7. `CreateRemoveXlaMarkTensorOpsPass` (`:90`) — remove `xla.mark_tensor` boundary ops — torch-xla custom
  8. `createCanonicalizerPass` (`:91`) + `createCSEPass` (`:92`) — generic MLIR canonicalize + CSE
- The only "optimization-like" steps are the **generic MLIR `canonicalize` (×2) and `CSE` (×1)** — dialect
  cleanup, not the XLA HLO optimization pipeline (no constant folding, DCE, fusion, reshape/transpose
  movement, layout, etc.). This means the StableHLO that reaches tt-mlir has had a light MLIR
  canonicalize+CSE applied, and nothing more.
- The fork's `openxla_patches/` contains only three diffs (`count_down.diff` = CPU conv,
  `gpu_nvml.diff`, `gpu_race_condition.diff`) — none touch any pass pipeline, and none run for the TT
  plugin.
- The Python layer (`torch_xla/`) contains no HLO-pass invocation; the dynamo bridge routes through the
  same `PjRtComputationClient::Compile` path above.

Net: the custom fork **adds no HLO optimization passes** beyond upstream behavior. It performs HLO→MHLO→
StableHLO **translation** plus a generic MLIR canonicalize/CSE cleanup, then hands the StableHLO to the TT
plugin. This strengthens (does not change) the Part 1 verdict.

### Important nuance — optimizations that *do* run before TTIR (but are not OpenXLA HLO)

tt-mlir runs its own **StableHLO-dialect** pipeline (`createStableHLOPipeline`,
`third_party/tt-mlir/.../StableHLO/Pipelines/StableHLOPipelines.cpp:17`) before converting to TTIR:
- `createInlinerPass` (`:20`), `createCanonicalizerPass` (`:139`)
- optional `StablehloAggressiveSimplificationPass` (`:31`, **off by default here**)
- `StableHLOFusingPass` (`:35`)
- Shardy sharding propagation / reshard / collective passes (`createConvertXlaSdyToSdyPass`, etc.)

These are **upstream-StableHLO + Shardy + tt-mlir** passes, not OpenXLA HLO-dialect backend passes. This
matters for Part 2: it means tt-mlir already has a hook for StableHLO-native simplification, which is the
cleaner alternative to OpenXLA HLO (discussed below).

---

## Part 2 — Pass-by-pass overlap analysis

### Framing facts that shape the whole comparison

1. **At the default optimization level (`optimization_level = 0`), the heavy tt-mlir Optimizer does not
   run at all.** `TTNNPipelines.h:50-58, 542-552`: opt-level 0 disables the entire layout/sharding/
   op-config search subsystem; layouts come from the simple `TTNNLayout` pass plus workarounds. The
   "interesting" hardware optimization (ShardSolver, L1 chains, beam-search layout propagation, op-config
   search, spill management) only engages at opt-level ≥ 1–2. So the set of passes one might "remove and
   offload" is dominated by the lightweight *simplification* passes, not the expensive solver.

2. **tt-mlir "fusion" ≠ OpenXLA "fusion."** OpenXLA's `InstructionFusion` / `MultiOutputFusion` /
   `FusionMerger` are loop/codegen fusion for CPU/GPU (merge compute loops, then emit). tt-mlir's fusion
   (`TTIRFusing`, `TTNNFusing`) matches op *patterns* into specific TTNN **fused library kernels**
   (conv+bias, matmul+activation, softmax, RMSNorm/LayerNorm, SDPA, RoPE, split-QKV). These are not the
   same kind of transformation and do not substitute for each other.

3. **OpenXLA has no single "target-independent optimization pipeline" you can call.** The portable
   "simplification" set is assembled *inside each backend compiler* (`cpu_compiler.cc:478-551`,
   `gpu_compiler.cc:920-959`) as an `HloPassFix<HloPassPipeline>`. To reuse it you would hand-assemble an
   `HloPassPipeline` from the portable subset and run it on an `HloModule`. There is no off-the-shelf
   "run the generic HLO opts" function.

### Category-by-category mapping

Legend for **Overlap**: ●●● strong / ●●○ partial-conceptual / ●○○ weak / ○○○ none.
**Offload?** = could this tt-mlir work plausibly be replaced by running an OpenXLA HLO pass in Torch-XLA?

#### A. Generic infrastructure — canonicalization / CSE / DCE / inlining

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| `createCanonicalizerPass` (run 5+×, `TTNNPipelines.cpp:47…471`) | `AlgebraicSimplifier` + `TupleSimplifier` + folds | ●●● | No benefit |
| `createCSEPass` (run 3×) | `HloCSE` (XLA's GVN equivalent, `hlo_cse.h:36`) | ●●● | No benefit |
| `createInlinerPass` | `CallInliner` (`call_inliner.h:41`), `FlattenCallGraph` | ●●● | No benefit |
| `createRemoveDeadValuesPass` (opt-in, off) | `HloDCE` (`hlo_dce.h:40`), `HloModuleDCE` | ●●● | No benefit |

These are *upstream MLIR* passes that tt-mlir gets for free. They are cheap, run in-tree, and operate on
TTIR/TTNN. OpenXLA's versions operate on HLO. There is full conceptual overlap but **nothing to gain** by
offloading — MLIR already provides them, and they would still be needed on TTIR/TTNN after the StableHLO→
TTIR conversion creates new redundancy.

#### B. Algebraic / TM simplification (the real overlap candidate)

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| `TTIREraseInverseOps` + `TTIRExplicateTMs` (commute permute/reshape/broadcast through elementwise so inverse TM pairs cancel; `Passes.td:165,188`) | `ReshapeMover` (`reshape_mover.h:49`), `TransposeFolding` (`transpose_folding.h:31`), broadcast/reshape/transpose rewrites inside `AlgebraicSimplifier` | ●●○ | Partially, with caveats |
| `TTIRImplicitBroadcastFold` (`Passes.td:10`) | `AlgebraicSimplifier` broadcast simplification, `BroadcastCanonicalizer` | ●●○ | Partially |
| `TTIRFoldFullToScalar` (`Passes.td:474`), `TTIRFoldConstantReshapeBroadcast` (`Passes.td:347`), `TTIRMoveReshapeToConstant` (`Passes.td:318`) | `AlgebraicSimplifier` + `HloConstantFolding` | ●●○ | Partially |
| `TTIRFusing` *activation idioms* (`max(x,0)→relu`, silu/mish/gelu/hardsigmoid, `sum/n→mean`; `TTIRFusing.cpp:409…`) | `AlgebraicSimplifier` (transcendental/select/compare folds) — but XLA does **not** synthesize named activation ops | ●○○ | No (these target TTNN ops) |

This is where OpenXLA's `AlgebraicSimplifier` (the single densest HLO pass — arithmetic identities,
reassociation, broadcast/reshape/transpose/slice/pad canonicalization, convert-chain elimination,
select/compare folding; `algebraic_simplifier.h:448`) genuinely overlaps with tt-mlir's TM-simplification.

**Caveat that limits the overlap:** tt-mlir's TM passes are tuned to *what TTNN can represent and what is
cheap on the Tenstorrent grid*. `TTIREraseInverseOps` cancels data-movement TMs because reshapes/permutes
are expensive on-device; it is not a generic algebraic simplifier. OpenXLA's `ReshapeMover`/
`AlgebraicSimplifier` optimize for HLO/CPU/GPU invariants and may produce StableHLO that is *neutral or
even worse* for the StableHLO→TTIR→TTNN path. So even where the categories match, the *target* differs.

#### C. Constant folding / constant evaluation

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| MLIR canonicalizer `fold` + `TTIRFoldConstantReshapeBroadcast` | `HloConstantFolding` (`hlo_constant_folding.h:32`), `HloConstantSplitter` | ●●○ | Partially |
| `ConstEvalHoistTransform` (hoist const-only subgraphs into cached `load_cached` fns; `TTNNPipelines.cpp:325,425,441`) | *No equivalent* — XLA folds at compile time; it does not hoist to runtime-cached functions | ○○○ | No |
| `CPUHoistConstEvalTransform`, `TTNNConstEvalInputsToSystemMemory` | none | ○○○ | No |

Compile-time constant *folding* overlaps. But tt-mlir's flagship `ConstEvalHoistTransform` is a
*different concept*: it does not fold constants away at compile time, it outlines constant/parameter-only
subgraphs into cached functions that execute once on-device. OpenXLA has nothing analogous, so this
cannot be offloaded.

#### D. Fusion (no usable overlap)

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| `TTIRFusing` / `TTNNFusing` (conv+bias, matmul+activation, softmax, RMS/LayerNorm, SDPA, split-QKV, RoPE, TopK; `TTIRFusing.cpp`, `TTNNFusing.cpp:300`) | `InstructionFusion` (`instruction_fusion.h:62`), `MultiOutputFusion`, `FusionMerger` | ○○○ | No |

Fundamentally different transformations (kernel-pattern matching vs. loop fusion). Offloading is not
possible, and OpenXLA simplification run *before* tt-mlir could even **break the op shapes the tt-mlir
fusers pattern-match on** (e.g. an HLO-level rewrite of a softmax sub-expression would defeat
`TTNNFusing`'s SDPA matcher). This is a risk, not an opportunity.

#### E. Gather / scatter / dot / convolution normalizers

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| TTIR decompositions: `TTIRFlattenSlidingWindow` (conv→1×1×N), `TTIRDecomposeComplexReshape/Permute`, `TTIRReductionForceKeepDim` (`Passes.td:130,236,409,427`) | `DotDecomposer` (`dot_decomposer.h:30`), `GatherSimplifier`/`GatherExpander`, `ScatterSimplifier`, `ConvolutionGroupConverter`, `BatchedGatherScatterNormalizer` | ●○○ | No |

Both compilers normalize/decompose dot/conv/gather/scatter, but tt-mlir's decompositions target *TTNN op
constraints* (e.g. force `keep_dim`, flatten sliding windows into TTNN-friendly forms). XLA's normalizers
target HLO canonical forms. The intent overlaps; the output forms do not transfer.

#### F. Control flow

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| `createInlinerPass`, `TTIRConsolidateStaticCacheUpdates` (`Passes.td:499`) | `WhileLoopSimplifier`, `WhileLoopConstantSinking`, `WhileLoopInvariantCodeMotion`, `ConditionalSimplifier`, `CallInliner` | ●○○ | Marginal |

StableHLO from JAX/Torch-XLA is largely already inlined/flattened by the frontend, so most XLA
while/conditional machinery has little to act on by the time it reaches tt-mlir. Low value.

#### G. Type / precision / dtype

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| `ElementTypeNormalization` (i64/f64→supported, i1→bf16; `Passes.td:104`), `PredicateTypeAlignment`, `TTIRQuantDataTypeConversion`, `TTNNWeightDtypeConversion` (bfp_bf8/bf4), `TTIRQuantDequantConversion` | `FloatNormalization` (`float_normalization.h:35`), `BFloat16ConversionFolding`, `BFloat16Propagation`, `HloElementTypeConverter`, `OperandUpcaster` | ●●○ (concept) / ○○○ (target) | No |

The *concept* (insert converts for unsupported types, propagate low precision) overlaps. But tt-mlir's
type passes encode Tenstorrent-specific facts: the supported type set, tile constraints, block-float
formats (`bfp_bf8`/`bfp_bf4`), and weight-only quantization. OpenXLA's `FloatNormalization` is driven by a
backend `FloatSupport` descriptor that does not know about TTNN. Not offloadable.

#### H. Sharding / collectives / multi-device

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| Shardy passes in `createStableHLOPipeline`; `TTNNConfigureCCLOps`, `TTNNAllocateDistributedOp*`, all-reduce→reduce-scatter+all-gather (in `TTNNWorkarounds`) | `ShardingPropagation`, SPMD partitioner, `AllReduce*`/`AllGather*`/`ReduceScatter*` family, `CollectivePermute*` | ●●○ (at StableHLO/Shardy level) | Already shared via Shardy |

Note tt-mlir already consumes **Shardy** (the same sharding framework OpenXLA uses) at the StableHLO
level. So the sharding-propagation overlap is already realized through a shared upstream component, not
through OpenXLA's HLO collective passes. The TTNN-level CCL configuration is hardware-specific.

#### I. Layout / memory / the TTNN Optimizer — **no analogue**

| tt-mlir | OpenXLA HLO | Overlap | Offload? |
|---|---|---|---|
| `TTNNLayout`, `TTNNDecomposeLayouts` (tilize/untilize), `TTNNRowMajorLayoutPropagation` | `LayoutAssignment` (`layout_assignment.h:259`), `LayoutNormalization` | ●○○ | No |
| **TTNN Optimizer**: `GreedyMemoryLayoutPropagation` (beam search over DRAM-interleaved vs L1-sharded; `OptimizerPasses/GreedyMemoryLayoutPropagation.cpp:45`), `ShardSolver` (`Analysis/ShardSolver.cpp:47`), L1 chains, `GreedyL1SpillManagement` (Belady-style eviction), op-config search (`MatmulProgramConfig`, `Conv2dConfigSearchSpace`), `OperationValidationAndFallback` | none — XLA layout assignment is CPU/GPU specific; `HloMemoryScheduler`/`HloRematerialization` are different problems | ○○○ | No |
| `TTNNMemoryManagement`, `TTNNDeallocate`, `TTNNSetComputeKernelConfig`, `TTNNTraceHoistTransform` | none target-independent | ○○○ | No |

This bucket is the **majority of tt-mlir's optimization value and code**, and it has essentially no
OpenXLA HLO counterpart. OpenXLA's layout assignment, memory scheduling, and rematerialization are
solving CPU/GPU-codegen problems with no relationship to Tenstorrent grid/L1/sharding. None of it can be
offloaded.

### Overlap summary

| Category | Overlap with OpenXLA HLO | Can it be offloaded to OpenXLA HLO? |
|---|---|---|
| Canonicalize / CSE / DCE / inline | ●●● | No (already native MLIR, cheap, still needed on TTIR/TTNN) |
| Algebraic / TM simplification | ●●○ | Partially — but better served by StableHLO-native passes (see below) |
| Constant folding | ●●○ | Folding only; const-eval **hoisting** has no analogue |
| Const-eval hoisting / CPU offload | ○○○ | No |
| Fusion (kernel-pattern) | ○○○ | No — and OpenXLA simplification can *break* tt-mlir fusers |
| Dot/conv/gather/scatter normalization | ●○○ | No (different target forms) |
| Control flow | ●○○ | Marginal (already inlined by frontend) |
| Type / precision / dtype | ●●○ concept | No (TTNN-specific) |
| Sharding / collectives | ●●○ | Already shared via Shardy upstream |
| Layout / memory / TTNN Optimizer | ○○○ | No — the bulk of tt-mlir, no analogue |

---

## Part 3 — Feasibility assessment

### What could *conceptually* be offloaded

Only the **target-independent simplification core**: `AlgebraicSimplifier`, `HloConstantFolding`,
`HloDCE`, `HloCSE`, `TupleSimplifier`, `ReshapeMover`, `TransposeFolding`, `ZeroSizedHloElimination`,
`SortSimplifier`. These would partially cover tt-mlir's canonicalizer + CSE/DCE + `TTIREraseInverseOps` +
`TTIRImplicitBroadcastFold` + `TTIRFoldFullToScalar` + the algebraic idioms in `TTIRFusing`.

### Why the ROI is low and the blockers are hard

1. **You cannot actually delete the tt-mlir passes.** tt-mlir is a standalone, multi-frontend compiler
   (tt-xla JAX path, tt-torch, tt-forge, and arbitrary StableHLO input). Its canonicalization/
   simplification passes guarantee correctness and TTNN-compatibility for *any* incoming StableHLO. If
   you inject HLO optimizations only in the Torch-XLA path:
   - The **JAX path does not benefit** (JAX lowers to StableHLO through its own path, not the Torch-XLA
     HLO hook). tt-mlir must keep all the passes for JAX anyway.
   - tt-mlir's correctness would become **coupled to the producer**, which is exactly the kind of
     fragility a compiler boundary exists to prevent.
   So at best the passes become *conditionally skippable*, not removable — which adds complexity rather
   than removing it.

2. **The valuable tt-mlir work has no analogue.** The TTNN Optimizer, layout/sharding/op-config search,
   tilize/untilize, workarounds, deallocation, and fused-kernel matching — the bulk of the compiler — are
   Tenstorrent-specific and stay regardless. So even in the best case, offloading trims only the cheap
   simplification layer.

3. **Mechanical cost in Torch-XLA + a lossy round-trip.** OpenXLA exposes no generic
   `RunOptimizationPasses`; you would hand-assemble an `HloPassPipeline` from the portable subset
   (`cpu_compiler.cc:478-551` is the template), run it on the `HloModule`, then convert HLO→MHLO→
   StableHLO. That conversion does not always round-trip cleanly (layout-sensitive forms, ops introduced
   by simplification, optimization barriers), and Torch-XLA upstream has no hook to run extra HLO passes
   before StableHLO emission — this would be a fork/patch to maintain against a fast-moving upstream.

4. **Risk of regressions.** Running OpenXLA simplification *before* tt-mlir can normalize away the exact
   op shapes that `TTIRFusing`/`TTNNFusing` pattern-match on (softmax → SDPA, the RoPE idioms, conv+bias),
   silently disabling Tenstorrent fused kernels. The simplification that helps a CPU/GPU loop nest is not
   guaranteed to help — and can hurt — a fused-kernel hardware backend.

### The better alternative if pre-simplification is wanted

If the goal is "hand tt-mlir a cleaner, smaller StableHLO graph," the correct lever is the
**StableHLO-dialect simplification passes that are already wired into tt-mlir's StableHLO pipeline** but
currently disabled by default:

- `StablehloAggressiveSimplificationPass` (`StableHLOPipelines.cpp:31`, off by default) — the
  StableHLO-native equivalent of `AlgebraicSimplifier`.
- `stablehlo-aggressive-folder` — the StableHLO-native equivalent of `HloConstantFolding`.
- The canonicalizer that already runs at `StableHLOPipelines.cpp:139`.

Advantages over the OpenXLA-HLO route:
- **Frontend-agnostic:** benefits the JAX path and any StableHLO producer, not just Torch-XLA.
- **No HLO round-trip:** operates directly on the StableHLO dialect already in hand — no
  HLO→MHLO→StableHLO conversion, no Torch-XLA fork.
- **Upstream-maintained:** these passes live in the StableHLO project and track the dialect.
- **In-tree and gated:** enabling them is a pipeline-option change, fully under tt-mlir's control, and
  can be ordered *before* the fuser passes so it does not defeat kernel matching.

This captures essentially the same target-independent simplification value the OpenXLA HLO passes would,
without the dependency, the round-trip, or the frontend coupling.

### Bottom line

- The overlap exists but is dominated by **generic infrastructure** (already free from MLIR) and
  **algebraic/TM simplification** (better served by StableHLO-native passes already in tt-mlir).
- The expensive, differentiating tt-mlir passes have **no OpenXLA HLO equivalent** and stay no matter
  what.
- Offloading to OpenXLA HLO passes via Torch-XLA would **not let you delete tt-mlir passes** (JAX +
  standalone tt-mlir still need them), would add a fragile, frontend-specific HLO round-trip, and risks
  disabling Tenstorrent fused kernels.
- **Recommended direction:** if graph pre-simplification is desired, turn on the StableHLO-dialect
  simplification/folding passes already present in tt-mlir's StableHLO pipeline (ordered before fusion),
  and treat tt-mlir's TTIR/TTNN passes as the source of truth rather than something to delete.

---

## Appendix — primary sources

**tt-xla / compile flow**
- `pjrt_implementation/src/api/client_instance.cc:478,831,847-855`
- `pjrt_implementation/src/api/module_builder/module_builder.cc:297,322,330,336,374,407,427`
- `python_package/torch_plugin_tt/__init__.py:38`; `python_package/tt_torch/backend/backend.py:346`;
  `python_package/jax_plugin_tt/__init__.py:23`

**tt-mlir passes** (`third_party/tt-mlir/src/tt-mlir`)
- Pass defs: `include/ttmlir/Dialect/TTIR/Transforms/Passes.td`,
  `include/ttmlir/Dialect/TTNN/Transforms/Passes.td`
- Pipeline + opt-level: `lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`,
  `include/ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h:50-58,542-552`
- StableHLO pipeline: `lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp:17,31,35,139`
- Optimizer/analysis: `lib/Dialect/TTNN/Analysis/{MemoryLayoutAnalysis,ShardSolver,L1ChainConfig}.cpp`,
  `lib/Dialect/TTNN/Transforms/OptimizerPasses/*`

**OpenXLA HLO passes** (`/localdev/hshah/xla/xla`)
- Simplification pipeline templates: `service/cpu/cpu_compiler.cc:478-551`,
  `service/gpu/gpu_compiler.cc:920-959`
- Core: `hlo/transforms/simplifiers/{algebraic_simplifier,hlo_constant_folding,hlo_dce,reshape_mover,
  tuple_simplifier,zero_sized_hlo_elimination}.h`, `service/{hlo_cse,transpose_folding}.h`
