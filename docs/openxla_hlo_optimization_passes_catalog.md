# OpenXLA HLO Optimization Pass Catalog

**Source:** OpenXLA repo cloned at `/localdev/hshah/xla` (commit `16d188d8a3`). Source lives under `/localdev/hshah/xla/xla/...`; paths below are cited relative to the repo as `xla/...`.

**Companion doc:** [`openxla_hlo_vs_ttmlir_optimization_overlap.md`](./openxla_hlo_vs_ttmlir_optimization_overlap.md) — analysis of how these passes overlap with tt-mlir's TTIR/TTNN passes.

## Scope & methodology

This catalogs **every HLO-level pass** OpenXLA provides in its two pass homes:

- `xla/hlo/transforms/` — the modern, target-independent home (subdirs `simplifiers/`, `expanders/`, `collectives/`).
- `xla/service/` — the legacy home; many target-independent passes still live here, plus `service/spmd/` (SPMD/sharding) and `service/debug/`.

**Deliberately excluded:** backend-specific passes under `xla/service/cpu/`, `xla/service/gpu/`, `xla/service/llvm_ir/`, and codegen/runtime backends. Those are not portable HLO optimizations (they assume a specific backend's cost model, layout rules, or codegen) and are not relevant to a StableHLO-producing frontend. A handful of passes below are infrastructure/verification/scheduling rather than optimizations — they are included for completeness and clearly flagged.

**How to read the entries.** Each entry gives the pass class name, its header/impl path, a 1–2 sentence description grounded in the header doc comment, and a minimal before→after example in HLO text. Examples marked **(conceptual)** illustrate a transform that is structural or too large to show faithfully as short text. Examples are illustrative, not verbatim test fixtures.

**Count:** ~170 pass classes across the sections below.

**Note on the "common simplification pipeline."** The passes that backends iterate to a fixed point as the portable simplification set (see the companion doc) are: `AlgebraicSimplifier`, `HloConstantFolding`, `HloDCE`, `HloCSE`, `TupleSimplifier`, `SortSimplifier`, `ReshapeMover`, `ZeroSizedHloElimination`, `WhileLoopSimplifier`, `WhileLoopConstantSinking`, `ConditionalSimplifier`, `GatherSimplifier`/`GatherExpander`, `TransposeFolding`. These are the strongest overlap candidates with another compiler's generic optimizer.

---

## Table of contents

1. [`xla/hlo/transforms/simplifiers/` — Simplification passes](#1-xlahlotransformssimplifiers--simplification-passes)
2. [`xla/hlo/transforms/expanders/` — Op expanders / decompositions](#2-xlahlotransformsexpanders--op-expanders--decompositions)
3. [`xla/hlo/transforms/collectives/` — Collective-communication optimizations](#3-xlahlotransformscollectives--collective-communication-optimizations)
4. [`xla/hlo/transforms/` (top level) — Misc HLO transforms](#4-xlahlotransforms-top-level--misc-hlo-transforms)
5. [`xla/service/` — Optimization / transformation passes](#5-xlaservice--optimization--transformation-passes)
6. [`xla/service/` — Infrastructure / scheduling / verification](#6-xlaservice--infrastructure--scheduling--verification)
7. [`xla/service/spmd/` and `xla/service/debug/` — SPMD & debug](#7-xlaservicespmd-and-xlaservicedebug--spmd--debug)

---

## 1. `xla/hlo/transforms/simplifiers/` — Simplification passes

This is the core target-independent optimization set.

### `AlgebraicSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/algebraic_simplifier.h` (impl `algebraic_simplifier.cc`)
- **Description:** A large DFS rewrite visitor that applies a broad set of algebraic identities and structural simplifications to HLO, running to a fixed point (up to `kAlgSimpRerunLimit` iterations); it canonicalizes and removes redundant arithmetic, broadcast/reshape/transpose, convert, select/compare, and dot operations.
- **Example:**
  ```
  // arithmetic identities
  add(x, 0) -> x ;  mul(x, 1) -> x ;  sub(x, 0) -> x ;  div(x, C) -> mul(x, 1/C)
  // broadcast / reshape / transpose simplification
  reshape(reshape(x)) -> reshape(x) ;  transpose(transpose(x)) -> transpose(x)
  reshape(broadcast(x)) -> broadcast(x) ;  transpose(x) -> bitcast/reshape when layout permits
  // dot strength reduction
  dot([1xK],[KxN]) -> reshape(multiply+reduce)        // vector*matrix lowered to reduce
  // convert chains
  convert(x : f32->f32) -> x                          // no-op convert removed
  // select / compare folds
  select(true, a, b) -> a ;  select(p, x, x) -> x
  // power simplifications
  pow(x, 1) -> x ;  pow(x, 0) -> 1 ;  pow(x, 0.5) -> sqrt(x) ;  pow(pow(a,X),Y) -> pow(a, X*Y)
  ```

### `AllGatherPadDsSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/all_gather_pad_ds_simplifier.h` (impl `all_gather_pad_ds_simplifier.cc`)
- **Description:** Optimizes a `DynamicSlice(Pad(AllGather))` pattern (a sharded all-gather result that is zero-padded then sliced per partition) by directly permuting the needed data from other partitions instead of materializing the large padded tensor, rewriting it into `CollectivePermute` + `Concatenate` (+ a `select` for the padded region).
- **Example:**
  ```
  // before
  ag = f64[1,8,40] all-gather(param)
  pad = f64[1,96,40] pad(ag, constant(0)), padding=0_0x0_88x0_0
  ds  = f64[1,24,40] dynamic-slice(pad, 0, ds_indices, 0)
  // after  (conceptual)
  cp1..cp3 = f64[1,2,40] collective-permute(param), source_target_pairs={...}
  concat   = f64[1,24,40] concatenate(param, cp1, cp2, cp3, broadcast(0))
  select   = select(compare(...), concat, broadcast(0))
  ```

### `AllGatherDynamicSlicePermutedOffsetSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/all_gather_permuted_ds_simplifier.h` (impl `all_gather_permuted_ds_simplifier.cc`)
- **Description:** Simplifies an `all-gather` followed by a permuted per-partition `dynamic-slice` (offsets taken from a permuted index list keyed on partition-id) into a single `collective-permute`.
- **Example:**
  ```
  // before
  ag = f32[256,8,128] all-gather(p), dimensions={0}
  offset = dynamic-slice({224,192,...,32,0}, partition-id())
  ds = f32[32,8,128] dynamic-slice(ag, offset, ...)
  // after
  cp = f32[32,8,128] collective-permute(p),
       source_target_pairs={{0,7},{1,6},{2,5},{3,4},{4,3},{5,2},{6,1},{7,0}}
  ```

### `AllReduceFolder`
- **Path:** `xla/hlo/transforms/simplifiers/all_reduce_folder.h` (impl `all_reduce_folder.cc`)
- **Description:** Folds an all-reduce that feeds directly into another all-reduce into a single all-reduce by expanding (combining) the replica groups of the two ops.
- **Example:**
  ```
  // before
  ar0 = all-reduce(x)   replica_groups={{0,1},{2,3},{4,5},{6,7}}
  ar1 = all-reduce(ar0) replica_groups={{0,2},{1,3},{4,6},{5,7}}
  // after
  ar1 = all-reduce(x)   replica_groups={{0,1,2,3},{4,5,6,7}}
  ```

### `ArCrsCombiner`
- **Path:** `xla/hlo/transforms/simplifiers/ar_crs_combiner.h` (impl `ar_crs_combiner.cc`)
- **Description:** In spatially-partitioned (MPMD/SPMD) modules, combines a cross-module all-reduce (CMAR) followed by simple linear ops and then a cross-replica all-reduce (CRAR/CRS) into a single all-core all-reduce, eliminating the early CMAR (dividing the other addend operand by the partition count when it feeds an add/sub).
- **Example:**
  ```
  // before
  cmar = all-reduce(x) channel_id=1   ; partial spatial reduction
  s    = add(cmar, z)
  crar = all-reduce(s)                ; cross-replica sum
  // after  (conceptual)
  s        = add(x, div(z, num_partitions))
  all-core = all-reduce(s)            ; single full all-reduce
  ```

### `BatchDotSimplification`
- **Path:** `xla/hlo/transforms/simplifiers/batch_dot_simplification.h` (impl `batch_dot_simplification.cc`)
- **Description:** Simplifies batch dot operations (reaching fixed point in one pass) by eliding degenerate (size-1) batch dimensions from the dot; kept separate from AlgebraicSimplifier so it runs before DotDecomposer.
- **Example:**
  ```
  // before
  dot = f32[1,5,7] dot(f32[1,5,11] a, f32[1,11,7] b), batch_dims={0}, contract={2,1}
  // after
  d = f32[5,7] dot(reshape(a)->[5,11], reshape(b)->[11,7])
  dot = f32[1,5,7] reshape(d)
  ```

### `BFloat16ConversionFolding`
- **Path:** `xla/hlo/transforms/simplifiers/bfloat16_conversion_folding.h` (impl `bfloat16_conversion_folding.cc`)
- **Description:** Folds F32 ↔ BF16 convert ops into the operands or users of instructions when the backend supports operating directly in BF16, removing redundant conversions; intended for the end of the pipeline since it can introduce mixed precision.
- **Example:**
  ```
  // before  (backend supports BF16 add)
  a = f32 convert(bf16 x) ; b = f32 convert(bf16 y) ; s = f32 add(a, b) ; out = bf16 convert(s)
  // after
  out = bf16 add(x, y)
  ```

### `BroadcastCanonicalizer`
- **Path:** `xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h` (impl `broadcast_canonicalizer.cc`)
- **Description:** Canonicalizes broadcast ops so that their `dimensions` attribute is sorted in increasing order (inserting a transpose to absorb the reordering).
- **Example:**
  ```
  // before
  bcast = f32[2,3,4] broadcast(f32[3,2] x), dimensions={1,0}
  // after
  bcast = f32[2,3,4] broadcast(f32[2,3] transpose(x)), dimensions={0,1}
  ```

### `CallParameterCleanup`
- **Path:** `xla/hlo/transforms/simplifiers/call_parameter_cleanup.h` (impl `call_parameter_cleanup.cc`)
- **Description:** Removes dead (unused) parameters of called computations, and rewrites pass-through parameters so callers' users reference the call operand directly (then removes the now-dead parameter).
- **Example:**
  ```
  // before
  callee(p0, p1) { ROOT r = add(p0, p0) }   // p1 unused
  c = call(a, b), to_apply=callee
  // after
  callee(p0) { ROOT r = add(p0, p0) }
  c = call(a), to_apply=callee
  ```

### `MoveParametersAndConstantsToFront`
- **Path:** `xla/hlo/transforms/simplifiers/computation_canonicalizers.h` (impl `computation_canonicalizers.cc`)
- **Description:** Reorders a computation so all `kParameter` and `kConstant` instructions appear at the front, simplifying later transforms (e.g. command-buffer computation construction).
- **Example:**
  ```
  // before
  a = add(p0, p0) ; c = constant(1) ; b = add(a, c)
  // after  (conceptual)
  p0 = parameter(0) ; c = constant(1) ; a = add(p0, p0) ; b = add(a, c)
  ```

### `MoveGTEsRightAfterTupleDefinition`
- **Path:** `xla/hlo/transforms/simplifiers/computation_canonicalizers.h` (impl `computation_canonicalizers.cc`)
- **Description:** Moves `GetTupleElement` instructions to immediately after the instruction that produces their tuple (e.g. run before command-buffer scheduling).
- **Example:**
  ```
  // before
  t = tuple(x, y) ; ... ; gte = get-tuple-element(t), index=0 ; use(gte)
  // after  (conceptual)
  t = tuple(x, y) ; gte = get-tuple-element(t), index=0 ; ... ; use(gte)
  ```

### `ConditionalCanonicalizer`
- **Path:** `xla/hlo/transforms/simplifiers/conditional_canonicalizer.h` (impl `conditional_canonicalizer.cc`)
- **Description:** Canonicalizes conditional (if/case) instructions so that all of them have tuple inputs/outputs, wrapping any non-tuple input/output into a single-element tuple.
- **Example:**
  ```
  // before
  c = f32[] conditional(pred, op_true, op_false), branches=...
  // after  (conceptual)
  c = (f32[]) conditional(...)              // output wrapped in tuple
  r = f32[] get-tuple-element(c), index=0
  ```

### `ConstantDeferring`
- **Path:** `xla/hlo/transforms/simplifiers/constant_deferring.h` (impl `constant_deferring.cc`)
- **Description:** Reorders a computation's instruction schedule to defer constant instructions to as close to their users as possible (reducing the live range of constants).
- **Example:**
  ```
  // before (schedule)
  c = constant(...) ; a = ... ; b = ... ; u = add(b, c)
  // after (schedule)  (conceptual)
  a = ... ; b = ... ; c = constant(...) ; u = add(b, c)
  ```

### `ConvOperandSwapper`
- **Path:** `xla/hlo/transforms/simplifiers/conv_operand_swapper.h` (impl `conv_operand_swapper.cc`)
- **Description:** Swaps a convolution's operands (input ↔ kernel, with appropriate dimension/reverse adjustments) when doing so yields a more efficient/lowerable operation.
- **Example:**
  ```
  // before
  conv = convolution(input, kernel), dim_numbers=...
  // after  (conceptual)
  conv = convolution(kernel, input), dim_numbers=...(swapped+reversed)
  ```

### `ConvertMover`
- **Path:** `xla/hlo/transforms/simplifiers/convert_mover.h` (impl `convert_mover.cc`)
- **Description:** Moves narrowing converts up and widening converts down the graph (when numerically neutral) so more work runs in lower precision and convert/reshape pairs become commutable, exposing further simplification.
- **Example:**
  ```
  // before
  big   = f32 convert(bf16 x) ; r = f32 reshape(big) ; small = bf16 convert(r)
  // after  (conceptual)
  r2    = bf16 reshape(x)        // convert(convert(reshape)) collapses
  ```

### `ConvertOperandFolding`
- **Path:** `xla/hlo/transforms/simplifiers/convert_operand_folder.h` (impl `convert_operand_folder.cc`)
- **Description:** Folds `convert` operands that widen the type into instructions that already support wider-than-shape-inference result accumulation, letting the op consume the narrower operands directly.
- **Example:**
  ```
  // before
  out = s32 dot/add(s32 convert(s8 a), s32 convert(s8 b))
  // after
  out = s32 dot/add(s8 a, s8 b)
  ```

### `ConvolutionGroupConverter`
- **Path:** `xla/hlo/transforms/simplifiers/convolution_group_converter.h` (impl `convolution_group_converter.cc`)
- **Description:** Rewrites grouped convolutions (`feature_group_count > 1`, or batch groups depending on config) into equivalent convolutions with `feature_group_count = 1`, typically via filter expansion, for backends that cannot lower grouped convs directly.
- **Example:**
  ```
  // before
  conv = convolution(input, filter), feature_group_count=4
  // after  (conceptual)
  conv = convolution(input, expanded_filter), feature_group_count=1
  ```

### `DeadDynamicUpdateSliceElimination`
- **Path:** `xla/hlo/transforms/simplifiers/dead_dynamic_update_slice_elimination.h` (impl `dead_dynamic_update_slice_elimination.cc`)
- **Description:** Removes a `dynamic-update-slice` whose written region is never read: when all its users are constant-indexed slices and none read the updated region, it replaces the DUS with its input operand (applied root-to-top so cascading DUSs are removed too).
- **Example:**
  ```
  // before
  dus = dynamic-update-slice(operand, update, c10, c0)   // writes [10:..]
  s   = slice(dus), slice={[0:5]}                          // never reads [10:..]
  // after
  s   = slice(operand), slice={[0:5]}
  ```

### `DotDimensionMerger`
- **Path:** `xla/hlo/transforms/simplifiers/dot_dimension_merger.h` (impl `dot_dimension_merger.cc`)
- **Description:** Merges consecutive batch dimensions of a `dot` into a single batch dimension by inserting reshapes around it.
- **Example:**
  ```
  // before
  dot = f32[2,3,M,N] dot(a, b), batch_dims={0,1}, ...
  // after
  d   = f32[6,M,N] dot(reshape(a), reshape(b)), batch_dims={0}
  dot = f32[2,3,M,N] reshape(d)
  ```

### `DotMerger`
- **Path:** `xla/hlo/transforms/simplifiers/dot_merger.h` (impl `dot_merger.cc`)
- **Description:** Merges two independent dots that share an operand into one dot over a concatenation of the other operands, followed by slices; gated by a `max_size_to_merge` threshold so only small dots are merged.
- **Example:**
  ```
  // before
  x = dot(a, b) ; y = dot(a, c)
  // after
  z = dot(a, concat(b, c)) ; x = slice(z) ; y = slice(z)
  ```

### `DynamicDimensionSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h` (impl `dynamic_dimension_simplifier.cc`)
- **Description:** Simplifies operations that compute dynamic dimension sizes (e.g. merging concats/reshapes on size scalars) so later passes can analyze dynamic shapes more easily.
- **Example:**
  ```
  // before
  c1 = s32[1] concat(a) ; c2 = s32[2] concat(c1, b)
  // after  (conceptual)
  c2 = s32[2] concat(a, b)
  ```

### `FusionConstantSinking`
- **Path:** `xla/hlo/transforms/simplifiers/fusion_constant_sinking.h` (impl `fusion_constant_sinking.cc`)
- **Description:** Sinks scalar/constant operands of a fusion into the fusion computation (removing the corresponding fusion parameter), so the constant lives inside the fused region.
- **Example:**
  ```
  // before
  c = f32[] constant(2)
  f = fusion(x, c), calls=fused { p0, p1 -> mul(p0, p1) }
  // after
  f = fusion(x), calls=fused { p0 -> mul(p0, constant(2)) }
  ```

### `GatherSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/gather_simplifier.h` (impl `gather_simplifier.cc`)
- **Description:** Rewrites a general `gather` into transposes/reshapes plus a simpler canonical gather whose `start_indices` is 2-D, `index_vector_dim=1`, `collapsed_slice_dims=[]`, and `offset_dims=[1,2,...]`.
- **Example:**
  ```
  // before
  g = gather(operand, start_indices), offset_dims={...}, collapsed_slice_dims={...}, index_vector_dim=...
  // after  (conceptual)
  g' = gather(operand, reshape(transpose(start_indices)->2D)),
       index_vector_dim=1, collapsed_slice_dims={}, offset_dims={1,2,...}
  // then reshape/transpose back to original shape
  ```

### `GemvRewriter`
- **Path:** `xla/hlo/transforms/simplifiers/gemv_rewriter.h` (impl `gemv_rewriter.cc`)
- **Description:** Rewrites matrix-vector / vector-matrix multiplies into matrix-matrix dots by adding a trivial size-1 dimension to the vector operand (and removing it from the result).
- **Example:**
  ```
  // before
  d = f32[m] dot(f32[m,n] A, f32[n] x)
  // after
  d2 = f32[m,1] dot(A, reshape(x)->[n,1]) ; d = f32[m] reshape(d2)
  ```

### `HloComputationDeduplicator`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_computation_deduplicator.h` (impl `hlo_computation_deduplicator.cc`)
- **Description:** Deduplicates identical computations within a module, keeping the first one in postorder and redirecting callers to it (or, with `mark_fusion_duplications`, only marking duplicate fusions for analysis tooling).
- **Example:**
  ```
  // before
  comp_a { ... }   comp_b { ... }     // identical bodies
  c1 = call(...), to_apply=comp_a ; c2 = call(...), to_apply=comp_b
  // after
  c1 = call(...), to_apply=comp_a ; c2 = call(...), to_apply=comp_a   // comp_b removed
  ```

### `HloConstantFolding`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_constant_folding.h` (impl `hlo_constant_folding.cc`)
- **Description:** Performs constant folding to avoid unnecessary computation on constants; a `kDefault` level folds only when it always helps runtime, while `kAggressive` folds everything possible (including unbounded while-loop iterations).
- **Example:**
  ```
  // before
  c = f32[2] constant({1,2}) ; ROOT add = f32[2] add(c, c)
  // after
  ROOT c = f32[2] constant({2,4})
  ```

### `HloConstantSplitter`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_constant_splitter.h` (impl `hlo_constant_splitter.cc`)
- **Description:** Splits constant instructions so each has a single user, typically before domain placement or sharding propagation to prevent a shared constant from short-circuiting domains. May produce dead instructions (run HloDCE after).
- **Example:**
  ```
  // before  (conceptual)
  c = constant(...)        // used by op1 and op2
  // after
  c1 = constant(...) -> op1 ;  c2 = constant(...) -> op2
  ```

### `HloDCE`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_dce.h` (impl `hlo_dce.cc`)
- **Description:** Removes dead instructions from each computation and dead computations from the module; an instruction is dead if unreachable from the root, a computation dead if non-entry and unreachable from the entry computation.
- **Example:**
  ```
  // before
  dead = f32[2] add(p0, p0)   // unused
  ROOT r = f32[2] multiply(p0, p1)
  // after
  ROOT r = f32[2] multiply(p0, p1)
  ```

### `HloElementTypeConverter`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_element_type_converter.h` (impl `hlo_element_type_converter.cc`)
- **Description:** Eliminates a given element type as op input/output by inserting Convert ops, letting a backend support a type while only implementing Convert for it.
- **Example:**
  ```
  // before (eliminate F16 -> F32)
  ROOT add = f16[2] add(f16[2] x, f16[2] y)
  // after
  add = f32[2] add(convert(x), convert(y)) ; ROOT c = f16[2] convert(add)
  ```

### `HloMemoryScheduler`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h` (impl `hlo_memory_scheduler.cc`)
- **Description:** Schedules the HLO instructions in a module to minimize peak memory use (via a chosen `ModuleSchedulerAlgorithm`, defaulting to `DefaultMemoryScheduler`), setting the module's `HloSchedule`. (The header also declares `HloTrivialScheduler` and `HloDescheduler`.)
- **Example:**
  ```
  // before (conceptual): unscheduled module
  // after: HloModule::set_schedule installs a sequential order chosen to minimize peak live memory
  ```

### `HloRematerialization`
- **Path:** `xla/hlo/transforms/simplifiers/hlo_rematerialization.h` (impl `hlo_rematerialization.cc`)
- **Description:** Rematerializes (recomputes / compresses / host-offloads) instructions to reduce peak memory use, given a memory limit; requires a scheduled module and should be run very late since CSE would undo it.
- **Example:**
  ```
  // before (conceptual): big tensor `t` kept live across many ops -> high peak memory
  // after: `t` recomputed just before its later use, shortening its live range below the limit
  ```

### `HostMemoryTransferAsyncifier`
- **Path:** `xla/hlo/transforms/simplifiers/host_memory_transfer_asyncifier.h` (impl `host_memory_transfer_asyncifier.cc`)
- **Description:** Finds copies between host and device memory (e.g. device-to-host DynamicUpdateSlice, host-to-device DynamicSlice) and converts them into asynchronous ops.
- **Example:**
  ```
  // before
  ROOT dus = f32[..]{S(host)} dynamic-update-slice(dst, update, idx)
  // after
  start = (..) dynamic-update-slice-start(dst, update, idx)
  ROOT done = f32[..]{S(host)} dynamic-update-slice-done(start)
  ```

### `InstructionHoister`
- **Path:** `xla/hlo/transforms/simplifiers/instruction_hoister.h` (impl `instruction_hoister.cc`)
- **Description:** Hoists parameters, constants, bitcasts and GetTupleElement operations (configurable) to increase opportunities for prefetching.
- **Example:**
  ```
  // before (conceptual): parameters/constants defined mid-computation
  // after: those instructions moved to the top, exposing more prefetch opportunities
  ```

### `OptimizeInputOutputBufferAlias`
- **Path:** `xla/hlo/transforms/simplifiers/optimize_input_output_buffer_alias.h` (impl `optimize_input_output_buffer_alias.cc`)
- **Description:** Finds input/output buffers of matching shape that can be aliased and writes the alias config into the HloModule (each input may alias at most one output).
- **Example:**
  ```
  // before (conceptual): Params P1(f32[3]),P2(s32[3]),P4(f32[16,12]); Outputs O1(s32[3]),O2(f32[3]),O3(f32[16,12])
  // after: alias config records (O1,P2), (O2,P1), (O3,P4)
  ```

### `RecognizeReduceWindow`
- **Path:** `xla/hlo/transforms/simplifiers/recognize_reduce_window.h` (impl `recognize_reduce_window.cc`)
- **Description:** Recognizes overlapping slices and pads being combined via elementwise operations and transforms them into reduce-window operations.
- **Example:**
  ```
  // before
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT add = f32[8] add(slice_1, slice_2)
  // after
  ROOT rw = f32[8] reduce-window(slice(x), zero), window={size=2 rhs_dilate=2}, to_apply=add
  ```

### `ReduceWindowResizer`
- **Path:** `xla/hlo/transforms/simplifiers/reduce_window_resizer.h` (impl `reduce_window_resizer.cc`)
- **Description:** Rewrites 1D reduce-windows by reshaping them to 2D reduce-windows, which helps tiling and avoids overly large reduction complexities.
- **Example:**
  ```
  // before
  ROOT rw = f32[10] reduce-window(f32[10] x, init), window={size=5 pad=2_2}, ...
  // after  (conceptual): reshape to f32[10,1], 2D reduce-window, reshape back to f32[10]
  ```

### `ReduceWindowRewriter`
- **Path:** `xla/hlo/transforms/simplifiers/reduce_window_rewriter.h` (impl `reduce_window_rewriter.cc`)
- **Description:** Rewrites quadratically-written ReduceWindow ops for performance: reshapes all R1 reduce-windows to R2 (and back), and rewrites CumSum/CumProd-style reduce-windows into tree reductions; must be run to a fixed point.
- **Example:**
  ```
  // before
  ROOT rw = f32[10]{0} reduce-window(f32[10] input, constant), window={size=5 pad=2_2}, to_apply=add
  // after
  rw = f32[10,1]{0,1} reduce-window(reshape(input), constant), window={size=5x1 pad=2_2x0_0}, to_apply=add
  ROOT reshape = f32[10]{0} reshape(rw)
  ```

### `AssociativeScanRewriter`
- **Path:** `xla/hlo/transforms/simplifiers/reduce_window_rewriter.h` (impl `reduce_window_rewriter.cc`)
- **Description:** Decomposes associative scan instructions into a single reduce-window op, or (when `0 < base_length < scan_length`) into a reduce-window tree.
- **Example:**
  ```
  // before (conceptual): associative scan over a length-N axis
  // after: single reduce-window (or a tree of reduce-windows if base_length < N)
  ```

### `ReshapeMover`
- **Path:** `xla/hlo/transforms/simplifiers/reshape_mover.h` (impl `reshape_mover.cc`)
- **Description:** Sinks reshape/transpose ("rearrange") ops down through elementwise ops, i.e. `op(rearrange(x), rearrange(y)) => rearrange(op(x, y))`, also handling trivially-rearrangeable operands like scalar broadcasts. Run to a fixed point with algsimp.
- **Example:**
  ```
  // before
  ROOT add = f32[8,7] add(reshape(x), reshape(y))
  // after
  add = f32[...] add(x, y) ; ROOT r = f32[8,7] reshape(add)
  ```

### `ResultCaster`
- **Path:** `xla/hlo/transforms/simplifiers/result_caster.h` (impl `result_caster.cc`)
- **Description:** Inserts a Convert on an instruction's result to its preferred accumulation element type when the backend doesn't support direct accumulation in that type; paired with OperandUpcaster.
- **Example:**
  ```
  // before
  ROOT dot = bf16[2,2] dot(bf16[2,2] a, bf16[2,2] b), ...
  // after
  dot = f32[2,2] dot(a, b), ... ; ROOT c = bf16[2,2] convert(dot)
  ```

### `RootInstructionSinker`
- **Path:** `xla/hlo/transforms/simplifiers/root_instruction_sinker.h` (impl `root_instruction_sinker.cc`)
- **Description:** In a scheduled module, sinks the ROOT of non-fusion computations to the bottom by creating a new ROOT: a tuple of GTEs for tuple roots, or a bitcast for non-tuple roots.
- **Example:**
  ```
  // before (scheduled): ROOT old_root = f32[2] add(...)  ... more instrs follow
  // after
  ... (other instrs) ... ; ROOT new_root = f32[2] bitcast(old_root)
  ```

### `SimplifyFPConversions`
- **Path:** `xla/hlo/transforms/simplifiers/simplify_fp_conversions.h` (impl `simplify_fp_conversions.cc`)
- **Description:** Simplifies chains of floating-point conversion (convert) ops.
- **Example:**
  ```
  // before
  a = bf16[2] convert(f32[2] x) ; ROOT b = f16[2] convert(a)
  // after
  ROOT b = f16[2] convert(x)
  ```

### `SliceHoister`
- **Path:** `xla/hlo/transforms/simplifiers/slice_hoister.h` (impl `slice_hoister.cc`)
- **Description:** Hoists slice operations through (up above) add operations.
- **Example:**
  ```
  // before
  add_op = f32[8,9] add(p0, p1) ; ROOT slice_op = f32[2,9] slice(add_op), slice={[0:2],[0:9]}
  // after
  ROOT add = f32[2,9] add(slice(p0), slice(p1))
  ```

### `SliceSinker`
- **Path:** `xla/hlo/transforms/simplifiers/slice_sinker.h` (impl `slice_sinker.cc`)
- **Description:** Sinks slice operations used by a group of elementwise operations and merges that group of elementwise operations.
- **Example:**
  ```
  // before
  add0 = f32[2,9] add(slice(p0,[0:2]), slice(p1,[0:2]))
  add1 = f32[6,9] add(slice(p0,[2:8]), slice(p1,[2:8]))
  // after
  add = f32[8,9] add(p0, p1)
  add0 = slice(add,[0:2]) ; add1 = slice(add,[2:8])
  ```

### `SortSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/sort_simplifier.h` (impl `sort_simplifier.cc`)
- **Description:** Removes unused operands from sort, where an unused operand is one at an index whose output is not used.
- **Example:**
  ```
  // before
  sort = (f32[64,87], s32[64,87], u32[64,87]) sort(keys, v0, v1), dimensions={1}  // index 1 unused
  // after
  sort = (f32[64,87], u32[64,87]) sort(keys, v1), dimensions={1}
  ```

### `SubByteCollectiveNormalization`
- **Path:** `xla/hlo/transforms/simplifiers/sub_byte_collective_normalization.h` (impl `sub_byte_collective_normalization.cc`)
- **Description:** Replaces pure data-movement collectives on sub-byte data types with equivalent collectives operating on whole bytes.
- **Example:**
  ```
  // before (conceptual)
  ROOT ag = s4[N] all-gather(s4[M] x), ...
  // after: bitcast to packed bytes, all-gather on bytes, bitcast back to s4
  ```

### `SubByteNormalization`
- **Path:** `xla/hlo/transforms/simplifiers/sub_byte_normalization.h` (impl `sub_byte_normalization.cc`)
- **Description:** Modifies the sub-byte `element_size_in_bits` layout annotation: `REMOVE_ELEMENT_SIZE` strips it (platforms without packed types); `SET_ELEMENT_SIZE` sets it to bitwidth(type) for <8-bit types and 0 otherwise.
- **Example:**
  ```
  // before (SET_ELEMENT_SIZE)
  x = s4[8]{0} parameter(0)
  // after
  x = s4[8]{0:E(4)} parameter(0)   // element_size_in_bits=4
  ```

### `TreeReductionRewriter`
- **Path:** `xla/hlo/transforms/simplifiers/tree_reduction_rewriter.h` (impl `tree_reduction_rewriter.cc`)
- **Description:** Increases reduction precision by first applying a reduce-window (kSame padding) then reducing the smaller result, effecting a pairwise-summation variant with smaller error bound.
- **Example:**
  ```
  // before
  ROOT r = f32[] reduce(f32[1024] x, init), dimensions={0}, to_apply=add
  // after
  rw = f32[32] reduce-window(x, init), window={size=32 stride=32 pad=same}, to_apply=add
  ROOT r = f32[] reduce(rw, init), dimensions={0}, to_apply=add
  ```

### `TupleSimplifier`
- **Path:** `xla/hlo/transforms/simplifiers/tuple_simplifier.h` (impl `tuple_simplifier.cc`)
- **Description:** Simplifies patterns of Tuple and GetTupleElement instructions, e.g. collapsing a Tuple built from order-preserving GTEs of a tuple-shaped op back to that op.
- **Example:**
  ```
  // before
  t = (f32[2], f32[2]) some-op(...) ; ROOT nt = tuple(gte(t,0), gte(t,1))
  // after
  ROOT t = (f32[2], f32[2]) some-op(...)
  ```

### `FlattenCallGraph`
- **Path:** `xla/hlo/transforms/simplifiers/flatten_call_graph.h` (impl `flatten_call_graph.cc`)
- **Description:** Flattens the call graph into a tree by duplicating computations called from multiple sites, so each (sequential) call site has a unique computation; simplifies buffer assignment and points-to analysis.
- **Example:**
  ```
  // before (conceptual): comp F called by both A and B
  // after: F cloned into F.1 (called by A) and F.2 (called by B)
  ```

### `UnflattenCallGraph`
- **Path:** `xla/hlo/transforms/simplifiers/unflatten_call_graph.h` (impl `unflatten_call_graph.cc`)
- **Description:** Inverse of flattening: finds identical called computations (content-based canonical hashing) and replaces them with calls to a single computation; only `kCall`-called computations are unflattened.
- **Example:**
  ```
  // before (conceptual): identical comps F.1 (called by A) and F.2 (called by B)
  // after: both A and B call a single merged F
  ```

### `ZeroSizedHloElimination`
- **Path:** `xla/hlo/transforms/simplifiers/zero_sized_hlo_elimination.h` (impl `zero_sized_hlo_elimination.cc`)
- **Description:** Replaces zero-sized HLOs with a zero-sized constant literal.
- **Example:**
  ```
  // before
  ROOT s = f32[0] slice(f32[10] x), slice={[0:0]}
  // after
  ROOT c = f32[0] constant({})
  ```

### `FloatNormalization`
- **Path:** `xla/hlo/transforms/simplifiers/float_normalization.h` (impl `float_normalization.cc`)
- **Description:** Adds type conversions (e.g. F32 ↔ BF16) for HLO instructions that don't support low-precision input/output or mixed precision, according to a backend-specific `FloatSupport` instance.
- **Example:**
  ```
  // before (backend lacks bf16 support for `add`)
  ROOT add = bf16[2] add(bf16[2] x, bf16[2] y)
  // after
  add = f32[2] add(convert(x), convert(y)) ; ROOT c = bf16[2] convert(add)
  ```

### `BFloat16MixedPrecisionRemoval`
- **Path:** `xla/hlo/transforms/simplifiers/float_normalization.h` (impl `float_normalization.cc`)
- **Description:** Unconditionally removes mixed F32/BF16 uses (excluding convert) by inserting F32 ↔ BF16 conversions, regardless of backend support, to make the module valid for passes that don't handle mixed precision (used by the Despecializer).
- **Example:**
  ```
  // before (mixed precision)
  ROOT add = f32[2] add(bf16[2] x, f32[2] y)
  // after
  ROOT add = f32[2] add(f32[2] convert(x), f32[2] y)
  ```

---

## 2. `xla/hlo/transforms/expanders/` — Op expanders / decompositions

These rewrite a single high-level HLO op into smaller primitives. `OpExpanderPass` (`xla/hlo/transforms/expanders/op_expander_pass.h`) is the shared base class, not a standalone optimization.

### `BitcastDtypesExpander`
- **Path:** `xla/hlo/transforms/expanders/bitcast_dtypes_expander.h` (impl `bitcast_dtypes_expander.cc`)
- **Description:** Expands a `bitcast-convert` between dtypes of different bit widths into a sequence of reshape/broadcast/bitcast/reduce ops that reinterpret the bytes.
- **Example:**
  ```
  // before
  ROOT out = s8[10,4] bitcast-convert(s32[10] p)
  // after (conceptual): reshape/broadcast s32[10]->s32[10,4], shift+mask bytes, reduce, convert to u8, bitcast -> s8[10,4]
  ```

### `CholeskyExpander`
- **Path:** `xla/hlo/transforms/expanders/cholesky_expander.h` (impl `cholesky_expander.cc`)
- **Description:** Expands the `cholesky` op into an equivalent blocked Cholesky factorization built from primitive HLOs.
- **Example:**
  ```
  // before
  ROOT chol = f32[4,4] cholesky(f32[4,4] a), lower=true
  // after (conceptual): blocked Cholesky — per-block unblocked factorization + triangular-solve + dot updates
  ```

### `ComparisonExpander`
- **Path:** `xla/hlo/transforms/expanders/comparison_expander.h` (impl `comparison_expander.cc`)
- **Description:** Expands floating-point `compare` ops requiring total-order semantics into integer comparisons by bitcasting floats to a signed integral representation (handling NaN and sign).
- **Example:**
  ```
  // before
  ROOT lt = pred[N] compare(f32[N] a, f32[N] b), direction=LT, type=TOTALORDER
  // after (conceptual): bitcast a,b to s32; map to monotonic total-order ints; compare the ints with LT
  ```

### `Convolution4DExpander`
- **Path:** `xla/hlo/transforms/expanders/convolution_4d_expander.h` (impl `convolution_4d_expander.cc`)
- **Description:** Expands convolutions with 4 (or more) spatial dimensions into convolutions with fewer spatial dimensions by collapsing trivial/contractible spatial dims via reshapes.
- **Example:**
  ```
  // before
  conv = ... convolution(input, filter)  // 4 spatial dims, some trivial
  // after (conceptual): reshape to merge collapsible spatial dims -> conv with fewer spatial dims -> reshape back
  ```

### `ConvolutionPredExpander`
- **Path:** `xla/hlo/transforms/expanders/convolution_pred_expander.h` (impl `convolution_pred_expander.cc`)
- **Description:** Rewrites boolean (`pred`) convolutions to floating point and converts the result back to boolean, since cuDNN convolutions only support FP and S8 inputs.
- **Example:**
  ```
  // before
  ROOT conv = pred[...] convolution(pred[...] lhs, pred[...] rhs)
  // after: convert lhs,rhs pred->f16 ; conv in f16 ; convert result f16->pred
  ```

### `ConvolutionTypeCanonicalizer`
- **Path:** `xla/hlo/transforms/expanders/convolution_type_canonicalizer.h` (impl `convolution_type_canonicalizer.cc`)
- **Description:** Canonicalizes convolutions with integral operand and result types by computing the convolution in F32 and converting the result back to the original integral type.
- **Example:**
  ```
  // before
  ROOT conv = s32[...] convolution(s32[...] lhs, s32[...] rhs)
  // after: conv computed in f32[...] ; ROOT = s32[...] convert(conv_f32)
  ```

### `DotDecomposer`
- **Path:** `xla/hlo/transforms/expanders/dot_decomposer.h` (impl `dot_decomposer.cc`)
- **Description:** Converts dots into a canonical form where non-contracting and contracting dimensions are each reshaped into a single dimension and batch dimensions are the most-major dimensions.
- **Example:**
  ```
  // before
  ROOT dot = f32[2,3,5] dot(f32[2,4,3] a, f32[2,4,5] b),
             lhs_batch={0}, rhs_batch={0}, lhs_contract={1}, rhs_contract={1}
  // after (conceptual): transpose/reshape so batch is major + single contracting/non-contracting dim, dot, reshape back
  ```

### `DynamicIndexSplitter`
- **Path:** `xla/hlo/transforms/expanders/dynamic_index_splitter.h` (impl `dynamic_index_splitter.cc`)
- **Description:** Converts the single R1 index operand of `dynamic-slice` / `dynamic-update-slice` into separate scalar index operands (one per dimension).
- **Example:**
  ```
  // before
  ROOT ds = s32[1,1,1] dynamic-slice(s32[4,5,6] operand, s32[3] indices), dynamic_slice_sizes={1,1,1}
  // after
  ROOT ds = s32[1,1,1] dynamic-slice(operand, reshape(slice(indices)), reshape(slice(indices)), reshape(slice(indices)))
  ```

### `EighExpander`
- **Path:** `xla/hlo/transforms/expanders/eigh_expander.h` (impl `eigh_expander.cc`)
- **Description:** Expands the symmetric/Hermitian eigendecomposition (`Eigh`) custom call into an iterative Jacobi algorithm built from primitive HLOs, optionally sorting eigenvalues.
- **Example:**
  ```
  // before
  (v, w) = Eigh(f32[4,4] a), lower=true   // eigenvectors v, eigenvalues w
  // after (conceptual): Jacobi rotation sweeps in a while-loop until off-diagonal converges
  ```

### `LogisticExpander`
- **Path:** `xla/hlo/transforms/expanders/logistic_expander.h` (impl `logistic_expander.cc`)
- **Description:** Expands the `logistic` op into its elementwise definition using primitive ops.
- **Example:**
  ```
  // before
  ROOT y = f32[N] logistic(f32[N] x)
  // after
  y = 1 / (1 + exp(-x))
  ```

### `OptimizationBarrierExpander`
- **Path:** `xla/hlo/transforms/expanders/optimization_barrier_expander.h` (impl `optimization_barrier_expander.cc`)
- **Description:** Removes the `opt-barrier` operation, which is functionally a no-op (a CSE barrier), forwarding its operand to users.
- **Example:**
  ```
  // before
  b = f32[N] opt-barrier(f32[N] x) ; ... uses(b)
  // after
  ... uses(x)   // opt-barrier removed
  ```

### `PermutationSortExpander`
- **Path:** `xla/hlo/transforms/expanders/permutation_sort_expander.h` (impl `permutation_sort_expander.cc`)
- **Description:** Replaces key-value sorts whose key operand is provably a permutation of indices `0..dim-1` with a more efficient `Scatter` (computing an inverse permutation).
- **Example:**
  ```
  // before
  values = s32[64,8732] iota(), iota_dimension=1
  ROOT sort2 = (...) sort(indices, values), dimensions={1}, to_apply=lt_s32
  // after (conceptual): scatter values into positions given by the index keys (inverse permutation)
  ```

### `QrExpander`
- **Path:** `xla/hlo/transforms/expanders/qr_expander.h` (impl `qr_expander.cc`)
- **Description:** Expands the `QrDecomposition` custom call into a blocked Householder QR algorithm built from primitive HLOs, producing the WY representation and Q/R factors.
- **Example:**
  ```
  // before
  (q, r) = Qr(f32[m,n] a)
  // after (conceptual): blocked Householder reflections over column blocks; form Q from reflectors, R = triangular part
  ```

### `RaggedDotRewriter`
- **Path:** `xla/hlo/transforms/expanders/ragged_dot_rewriter.h` (impl `ragged_dot_rewriter.cc`)
- **Description:** Converts `ragged-dot` ops into ordinary (general) `dot` ops through expansion, using group offsets/masks to emulate the ragged grouping.
- **Example:**
  ```
  // before
  ROOT rd = ragged-dot(lhs, rhs, group_sizes)
  // after (conceptual): build per-group masks from cumsum(group_sizes), mask operands, compute a standard dot
  ```

### `ReduceDecomposer`
- **Path:** `xla/hlo/transforms/expanders/reduce_decomposer.h` (impl `reduce_decomposer.cc`)
- **Description:** Ensures a reduction's output layout matches its input layout (modulo removed dims) unless `custom_layout_allowed`; decomposes layout-mutating reductions into a reduction plus copy.
- **Example:**
  ```
  // before
  reduce{L'}            // output layout differs from input layout L
  // after
  reduce{E(L)} -> copy{L'}
  ```

### `ReshapeDecomposer`
- **Path:** `xla/hlo/transforms/expanders/reshape_decomposer.h` (impl `reshape_decomposer.cc`)
- **Description:** Decomposes any reshape that is not a bitcast into a bitcast plus a copy (physical transposition). Postcondition: all reshapes are bitcasts.
- **Example:**
  ```
  // before
  ROOT r = f32[...] reshape(x)    // not a bitcast (changes physical layout)
  // after: copy(x) (physical transpose) then bitcast -> reshape result
  ```

### `RngBitGeneratorExpander`
- **Path:** `xla/hlo/transforms/expanders/rng_bit_generator_expander.h` (impl `rng_bit_generator_expander.cc`)
- **Description:** Expands `rng-bit-generator` into a concrete counter-based RNG computation (ThreeFry or Philox) selected by the algorithm, producing the updated state and random bits.
- **Example:**
  ```
  // before
  ROOT rbg = (state, bits) rng-bit-generator(state), algorithm=RNG_DEFAULT
  // after (conceptual): call a generated ThreeFry/Philox computation mapping (key,state) -> (new_state, bits)
  ```

### `RngExpander`
- **Path:** `xla/hlo/transforms/expanders/rng_expander.h` (impl `rng_expander.cc`)
- **Description:** Expands `rng` ops (uniform/normal distributions) into a computation built on an underlying bit generator and the distribution's transform.
- **Example:**
  ```
  // before
  ROOT r = f32[N] rng(f32[] a, f32[] b), distribution=rng_uniform
  // after (conceptual): generate random bits, convert to f32 in [0,1), scale/shift by (a,b)
  ```

### `StableSortExpander`
- **Path:** `xla/hlo/transforms/expanders/stable_sort_expander.h` (impl `stable_sort_expander.cc`)
- **Description:** Expands `sort` ops with `is_stable=true` into equivalent sorts that guarantee stable ordering without relying on the `is_stable` field (by adding an iota tiebreaker operand).
- **Example:**
  ```
  // before
  ROOT s = sort(keys), dimensions={0}, is_stable=true, to_apply=cmp
  // after (conceptual): sort (keys, iota) with comparator breaking ties on the iota index, then drop iota
  ```

### `StochasticConvertDecomposer`
- **Path:** `xla/hlo/transforms/expanders/stochastic_convert_decomposer.h` (impl `stochastic_convert_decomposer.cc`)
- **Description:** Replaces the unsupported `stochastic-convert` op with multiple HLOs that perform stochastic rounding (comparing the random bits against the fractional part).
- **Example:**
  ```
  // before
  ROOT sc = s8[N] stochastic-convert(f32[N] operand, u32[N] random)
  // after (conceptual): truncate operand; compare scaled random bits against fractional part; select-add 1 to round up
  ```

---

## 3. `xla/hlo/transforms/collectives/` — Collective-communication optimizations

### `AllGatherBroadcastReorder`
- **Path:** `xla/hlo/transforms/collectives/all_gather_broadcast_reorder.h` (impl `all_gather_broadcast_reorder.cc`)
- **Description:** Reorders `all-gather(broadcast(x))` into `broadcast(all-gather(x))` so the all-gather operates on the smaller pre-broadcast data.
- **Example:**
  ```
  // before
  %bcast = broadcast(%x) ; %ag = all-gather(%bcast)
  // after
  %ag = all-gather(%x) ; %bcast = broadcast(%ag)
  ```

### `AllGatherCombiner`
- **Path:** `xla/hlo/transforms/collectives/all_gather_combiner.h` (impl `all_gather_combiner.cc`)
- **Description:** Combines small, non-dependent all-gather ops into a single larger all-gather, amortizing per-op latency. Grouped by key (gather dim, channel id, replica groups, dtype, ...).
- **Example:**
  ```
  // before
  %ag0 = all-gather(%a), dimensions={0} ; %ag1 = all-gather(%b), dimensions={0}
  // after
  %ag = all-gather(%a, %b), dimensions={0}   // tuple-shaped combined all-gather
  ```

### `AllGatherCSE`
- **Path:** `xla/hlo/transforms/collectives/all_gather_cse.h` (impl `all_gather_cse.cc`)
- **Description:** CSEs duplicate all-gathers of the same parameter so there is only one all-gather per parameter, setting up later collective code-motion passes.
- **Example:**
  ```
  // before
  %ag1 = all-gather(%param_0) ; %ag2 = all-gather(%param_0)
  // after
  %ag0 = all-gather(%param_0)   // both uses now reference %ag0
  ```

### `AllGatherRemoveDegenerateDims`
- **Path:** `xla/hlo/transforms/collectives/all_gather_remove_degenerate_dims.h` (impl `all_gather_remove_degenerate_dims.cc`)
- **Description:** Removes size-1 (degenerate) dimensions from all-gathers by wrapping them in reshapes, which helps layout assignment (the reshapes become bitcasts).
- **Example:**
  ```
  // before
  %ag = f32[1,64,8192] all-gather(%in), dimensions={1}
  // after
  %reshape = f32[64,8192] reshape(%in)
  %ag.r   = f32[64,8192] all-gather(%reshape), dimensions={0}
  %ag     = f32[1,64,8192] reshape(%ag.r)
  ```

### `AllReduceCombiner`
- **Path:** `xla/hlo/transforms/collectives/all_reduce_combiner.h` (impl `all_reduce_combiner.cc`)
- **Description:** Combines small, non-dependent all-reduce ops into a single larger all-reduce, amortizing per-op latency. Grouped by an all-reduce key.
- **Example:**
  ```
  // before
  %ar0 = all-reduce(%a), to_apply=add ; %ar1 = all-reduce(%b), to_apply=add
  // after
  %ar = all-reduce(%a, %b), to_apply=add   // tuple-shaped combined all-reduce
  ```

### `AllReduceContiguous`
- **Path:** `xla/hlo/transforms/collectives/all_reduce_contiguous.h` (impl `all_reduce_contiguous.cc`)
- **Description:** Concatenates the operands of a multi-operand all-reduce into one contiguous buffer so the all-reduce runs over a single buffer, slicing the result back out afterward.
- **Example:**
  ```
  // before
  %ar = (f32[4], f32[8]) all-reduce(%a, %b), to_apply=add
  // after (conceptual)
  %cat = f32[12] concatenate(flatten(%a), flatten(%b))
  %ar  = f32[12] all-reduce(%cat), to_apply=add ; slice/reshape back to %a',%b'
  ```

### `AsyncCollectiveCreator`
- **Path:** `xla/hlo/transforms/collectives/async_collective_creator.h` (impl `async_collective_creator.cc`)
- **Description:** Converts synchronous collectives (selected via per-op predicates) into async start/done pairs so compute can overlap them.
- **Example:**
  ```
  // before
  %ar = all-reduce(%x), to_apply=add
  // after
  %start = all-reduce-start(%x), to_apply=add ; %ar = all-reduce-done(%start)
  ```

### `AsyncCollectiveReplacer`
- **Path:** `xla/hlo/transforms/collectives/async_collective_replacer.h` (impl `async_collective_replacer.cc`)
- **Description:** Inverse of the creator (run before scheduling): replaces async start/done pairs with their synchronous form based on per-op predicates. (Distinct from `ConvertAsyncCollectivesToSync`, which runs after scheduling.)
- **Example:**
  ```
  // before
  %start = all-reduce-start(%x), to_apply=add ; %ar = all-reduce-done(%start)
  // after
  %ar = all-reduce(%x), to_apply=add
  ```

### `CollectivePermuteCombiner`
- **Path:** `xla/hlo/transforms/collectives/collective_permute_combiner.h` (impl `collective_permute_combiner.cc`)
- **Description:** Combines small, non-dependent collective-permute ops into a single larger collective-permute to amortize per-op overhead.
- **Example:**
  ```
  // before
  %cp0 = collective-permute(%a), source_target_pairs={{0,1},{1,0}}
  %cp1 = collective-permute(%b), source_target_pairs={{0,1},{1,0}}
  // after
  %cp = collective-permute(%a, %b), source_target_pairs={{0,1},{1,0}}
  ```

### `CollectivePermuteCSE`
- **Path:** `xla/hlo/transforms/collectives/collective_permute_cse.h` (impl `collective_permute_cse.cc`)
- **Description:** Removes redundant collective-permutes that share identical source-target pairs and operands, and can CSE a permute through a slice (`cp(slice(x))` → `slice(cp(x))`).
- **Example:**
  ```
  // before
  %cp0 = collective-permute(%input), source_target_pairs={{0,1}}
  %cp1 = collective-permute(%input), source_target_pairs={{0,1}}
  // after
  %cp0 = collective-permute(%input), source_target_pairs={{0,1}}  // %cp1 -> %cp0
  ```

### `CollectiveQuantizer`
- **Path:** `xla/hlo/transforms/collectives/collective_quantizer.h` (impl `collective_quantizer.cc`)
- **Description:** Cuts collective traffic by moving a following quantization/narrowing convert before the collective (and a preceding dequantization/widening convert after it).
- **Example:**
  ```
  // before
  %ag = bf16[...] all-gather(%x) ; %q = f8[...] convert(%ag)
  // after
  %q = f8[...] convert(%x) ; %ag = f8[...] all-gather(%q)   // gathers smaller f8 data
  ```

### `CollectiveTransformationReorder`
- **Path:** `xla/hlo/transforms/collectives/collective_transformation_reorderer.h` (impl `collective_transformation_reorderer.cc`)
- **Description:** Reorders `all-gather + reshape` into `reshape + all-gather`, and `reshape + all-reduce` into `all-reduce + reshape`, under single-user and layout-safety conditions.
- **Example:**
  ```
  // before
  %ag = [.., P*C_i, ..] all-gather(%input), dimensions={i} ; %r = reshape(%ag)
  // after
  %r = reshape(%input) ; %ag = all-gather(%r), dimensions={j}
  ```

### `CollectivesScheduleLinearizer`
- **Path:** `xla/hlo/transforms/collectives/collectives_schedule_linearizer.h` (impl `collectives_schedule_linearizer.cc`)
- **Description:** Enforces a total order on the collectives within each computation by adding control dependencies, preventing concurrent collective execution (no inter-computation deps).
- **Example:**
  ```
  // before (conceptual): %ar0, %ar1 independent -> may run concurrently
  // after: %ar1 gains a control dependency on %ar0, forcing serial order
  ```

### `ConvertAsyncCollectivesToSync`
- **Path:** `xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h` (impl `convert_async_collectives_to_sync.cc`)
- **Description:** Run after scheduling: converts async start/done pairs back to synchronous form when no compute op is scheduled to overlap between start and done.
- **Example:**
  ```
  // before (no compute between start and done)
  %start = all-gather-start(%x) ; %ag = all-gather-done(%start)
  // after
  %ag = all-gather(%x)
  ```

### `InfeedTokenPropagation`
- **Path:** `xla/hlo/transforms/collectives/infeed_token_propagation.h` (impl `infeed_token_propagation.cc`)
- **Description:** Finds dangling infeed/outfeed tokens inside nested computations (while bodies, conditional branches) and bubbles them up through callers to the entry computation, preserving ordering before inlining.
- **Example:**
  ```
  // before (conceptual): infeed token confined inside a while body
  // after: the infeed token is threaded through the while's operand/result tuple up to entry
  ```

### `ReorderReduceTranspose`
- **Path:** `xla/hlo/transforms/collectives/while_loop_all_reduce_code_motion_setup.h` (impl `while_loop_all_reduce_code_motion_setup.cc`)
- **Description:** Setup pass (OpExpander) that sinks a reduce-scatter past a following convert/transpose so it operates on the pre-transposed/converted operand, preparing for while-loop all-reduce code motion.
- **Example:**
  ```
  // before
  %rs = reduce-scatter(%operand) ; %t = transpose(convert(%rs)) ; %add = add(%t, gte(%param,0))
  // after
  %rs = reduce-scatter(transpose(convert(%operand))) ; %add = add(%rs, gte(%param,0))
  ```

### `ReorderConvertReduceAdd`
- **Path:** `xla/hlo/transforms/collectives/while_loop_all_reduce_code_motion_setup.h` (impl `while_loop_all_reduce_code_motion_setup.cc`)
- **Description:** Setup pass (OpExpander) that sinks a reduce-scatter/all-reduce past a following convert when the result feeds an add; `enable_reduce_scatter` gates the reduce-scatter case.
- **Example:**
  ```
  // before
  %ar = all-reduce(%operand) ; %add = add(convert(%ar), gte(%param,0))
  // after
  %add = add(all-reduce(convert(%operand)), gte(%param,0))
  ```

---

## 4. `xla/hlo/transforms/` (top level) — Misc HLO transforms

### `AddOriginalValue`
- **Path:** `xla/hlo/transforms/add_original_value.h` (impl `add_original_value.cc`)
- **Description:** Adds the `original_value` attribute to each op, used for HLO value tracking (mapping optimized instructions back to original-graph values).
- **Example:**
  ```
  // before
  v1 = f32[] parameter(0) ; t1 = (f32[], f32[3]) tuple(v1, v2)
  // after
  v1 = f32[] parameter(0), origin={{"v1"}} ; t1 = tuple(v1, v2), origin={({"v1"}, {"v2"})}
  ```

### `BFloat16Propagation`
- **Path:** `xla/hlo/transforms/bfloat16_propagation.h` (impl `bfloat16_propagation.cc`)
- **Description:** Reduces the precision of select HLO instructions to BF16 per a backend `FloatSupport` rule, without changing numerical output; run late (it introduces mixed precision, so `FloatNormalization` should follow).
- **Example:**
  ```
  // before
  add = f32[4] add(a, b) ; ROOT dot = f32[] dot(add, add)   // backend rounds dot inputs to bf16 anyway
  // after
  add = bf16[4] add(a, b)   // producer reduced to bf16; output bitwise identical
  ```

### `CallSplitter`
- **Path:** `xla/hlo/transforms/call_splitter.h` (impl `call_splitter.cc`)
- **Description:** Splits a single `kCall` into two calls to two computations across a predicate-defined boundary (`kDown`/`kUp`); also supports "vertical" splits of multi-output calls. (Marked not yet production-ready.)
- **Example:**
  ```
  // before
  (x) = call(a, b, c), to_apply={ mul(p0, add(p1, p2)) }   // boundary: opcode==kMultiply, kDown
  // after
  (t) = call(b, c), to_apply={ add(p0, p1) } ; (y) = call(a, t), to_apply={ mul(p0, p1) }
  ```

### `ConvertMemoryPlacementToInternalAnnotations`
- **Path:** `xla/hlo/transforms/convert_memory_placement_to_internal_annotations.h` (impl `convert_memory_placement_to_internal_annotations.cc`)
- **Description:** Converts external `annotate_device_placement` custom calls (`_xla_buffer_placement` attrs) into XLA's internal `MoveToHost`/`MoveToDevice` custom-call annotations used by host-offloading passes.
- **Example:**
  ```
  // before
  cc = f32[16] custom-call(x), custom_call_target="annotate_device_placement",
       frontend_attributes={_xla_buffer_placement="pinned_host"}
  // after
  cc = f32[16] custom-call(x), custom_call_target="MoveToHost"
  ```

### `Defuser`
- **Path:** `xla/hlo/transforms/defuser.h` (impl `defuser.cc`)
- **Description:** Replaces all fusion instructions with the equivalent un-fused instructions, inlining each fusion computation's body back into the parent computation.
- **Example:**
  ```
  // before
  f = f32[4] fusion(a, b), kind=kLoop, calls={ add(p0, mul(p0, p1)) }
  // after
  m = f32[4] multiply(a, b) ; f = f32[4] add(a, m)
  ```

### `Despecializer`
- **Path:** `xla/hlo/transforms/despecializer.h` (impl `despecializer.cc`)
- **Description:** A pass pipeline that "despecializes" an optimized module (runs `HloDescheduler`, `ControlDepRemover`, `Defuser`, `BFloat16MixedPrecisionRemoval`) so a module optimized for one platform can run on another with matching numerics. Also declares `AssumeGatherIndicesInBoundRewriteToCopy`, `DeconstructReduceWindowToReduceBroadcast`, `ControlDepRemover`.
- **Example:**
  ```
  // before (conceptual): platform-optimized module (bf16 fusions, control deps, fixed schedule)
  // after (conceptual): fusions inlined, control deps dropped, bf16-mixed-precision removed, unscheduled
  ```

### `DotDimensionNormalizer`
- **Path:** `xla/hlo/transforms/dot_dimension_normalizer.h` (impl `dot_dimension_normalizer.cc`)
- **Description:** Normalizes `dot` operations to have at most one contracting dimension per operand (and optionally at most one non-contracting dimension), inserting reshapes/transposes.
- **Example:**
  ```
  // before
  ROOT d = bf16[12,44] dot(p0, p1), lhs_contracting_dims={1,2}, rhs_contracting_dims={0,1}
  // after
  r0 = bf16[12,8] reshape(p0) ; r1 = bf16[8,44] reshape(p1)
  ROOT d = bf16[12,44] dot(r0, r1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ```

### `HloModuleSplitter`
- **Path:** `xla/hlo/transforms/hlo_module_splitter.h` (impl `hlo_module_splitter.cc`)
- **Description:** Identifies non-inlineable computations, extracts each into a separate submodule, and replaces the calls in the main module with `kCustomCall` instructions (paired with `HloModuleStitcher`).
- **Example:**
  ```
  // before (conceptual): main: ... = call(args), to_apply=non_inlineable_comp
  // after (conceptual): main: ... = custom-call(args), custom_call_target="_xla_multi_module_call" + extracted submodule
  ```

### `HloModuleStitcher`
- **Path:** `xla/hlo/transforms/hlo_module_stitcher.h` (impl `hlo_module_stitcher.cc`)
- **Description:** Inverse of `HloModuleSplitter`: stitches optimized submodules back into the main module, replacing `_xla_multi_module_call` custom calls with standard `kCall` instructions.
- **Example:**
  ```
  // before (conceptual): custom-call "_xla_multi_module_call" + separate optimized submodules
  // after (conceptual): main: ... = call(args), to_apply=<inlined optimized submodule>
  ```

### `HostOffloadLegalize`
- **Path:** `xla/hlo/transforms/host_offload_legalize.h` (impl `host_offload_legalize.cc`)
- **Description:** Legalizes the graph so the host-memory-offloading pass can correctly track buffers meant to move to host, removing constructs (e.g. mispositioned copies/transposes) that would block offload tracking.
- **Example:**
  ```
  // before (conceptual): MoveToHost path with copy/transpose interleaved so offload tracking breaks
  // after (conceptual): same computation, reordered so the MoveToHost-to-consumer path is cleanly traceable
  ```

### `HostOffloader`
- **Path:** `xla/hlo/transforms/host_offloader.h` (impl `host_offloader.cc`)
- **Description:** Performs host-memory offloading: walks from `MoveToHost`/host params, sets host memory space along each path, inserts device-to-host copies (or DUS writes), errors on compute over offloaded tensors, and removes all `MoveToHost`/`MoveToDevice` annotations.
- **Example:**
  ```
  // before (conceptual)
  h = custom-call(x), target="MoveToHost" ; ... = custom-call(h), target="MoveToDevice"
  // after (conceptual): copy to host-memory buffer (or DUS write), annotations removed, layouts get host memory space
  ```

### `HostOffloadingPrepare`
- **Path:** `xla/hlo/transforms/host_offloading_prepare.h` (impl `host_offloading_prepare.cc`)
- **Description:** A configurable set of rewrites preparing a module for host offloading; `kElideMoveToHost` removes `MoveToHost` feeding directly into a host computation, `kConvertToCustomCall` rewrites calls to host-offloaded computations into custom calls.
- **Example:**
  ```
  // before (kElideMoveToHost, conceptual)
  h = custom-call(x), target="MoveToHost" ; out = async-call(h), to_apply=host_computation
  // after (conceptual)
  out = async-call(x), to_apply=host_computation   // MoveToHost elided
  ```

### `LiteralCanonicalizer`
- **Path:** `xla/hlo/transforms/literal_canonicalizer.h` (impl `literal_canonicalizer.cc`)
- **Description:** Canonicalizes constant literals larger than `min_size_bytes` using a shared `LiteralPool`, deduplicating identical large literals to reduce module memory.
- **Example:**
  ```
  // before (conceptual): c1 = f32[1024] constant({...big...}) ; c2 = f32[1024] constant({...same...})
  // after (conceptual): c1, c2 share one pooled literal instance
  ```

### `MemorySpacePropagation`
- **Path:** `xla/hlo/transforms/memory_space_propagation.h` (impl `memory_space_propagation.cc`)
- **Description:** Legalization pass that propagates the memory space (and split config) from an instruction's layout into the corresponding parameters/root of its fusion computations.
- **Example:**
  ```
  // before
  fusion = s32[6]{T(128)S(1)} fusion(arg0{T(128)S(1)}, ...)   // fused params still {T(128)}
  // after (inside fused_computation)
  param_0.1 = s32[6]{T(128)S(1)} parameter(0)   // memory space S(1) propagated
  ```

### `offloader_util::FindAndWrapOffloadedComputations`
- **Path:** `xla/hlo/transforms/offloaded_instruction_wrapper.h` (impl `offloaded_instruction_wrapper.cc`)
- **Description:** Utility (in `xla::offloader_util`, not a pass class) that finds connected sets of instructions satisfying a `should_offload` predicate, wraps each into its own "island" computation, and replaces them with a `kCall`.
- **Example:**
  ```
  // before (conceptual): a = op_to_offload(...) ; b = op_to_offload(a)
  // after (conceptual): call = call(...), to_apply={ a; b }   // wrapped into an offloaded island
  ```

### `OperandUpcaster`
- **Path:** `xla/hlo/transforms/operand_upcaster.h` (impl `operand_upcaster.cc`)
- **Description:** An `OpExpanderPass` that inserts `Convert` ops on operands so result accumulation happens in a wider type matching the instruction's result type.
- **Example:**
  ```
  // before
  ROOT dot = s32[2,2] dot(s8[2,3] p0, s8[3,2] p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  // after
  c0 = s32[2,3] convert(p0) ; c1 = s32[3,2] convert(p1) ; ROOT dot = s32[2,2] dot(c0, c1)
  ```

### `PropagateCallMetadata`
- **Path:** `xla/hlo/transforms/propagate_call_metadata.h` (impl `propagate_call_metadata.cc`)
- **Description:** Propagates metadata (`op_name` prefix and `stack_frame_id`) from `kCall` instructions into their called computations, recursing through nested calls/control-flow; run late to annotate remaining non-inlined calls.
- **Example:**
  ```
  // before (conceptual): call metadata={op_name="outer"}, to_apply=comp   // comp ops have bare op_names
  // after (conceptual): inside comp, op_name="outer/<inner>", stack frame chained to caller
  ```

### `ShapeCanonicalizer`
- **Path:** `xla/hlo/transforms/shape_canonicalizer.h` (impl `shape_canonicalizer.cc`)
- **Description:** Canonicalizes HLO instruction shapes using a shared `ShapePool`, so instructions with identical shapes share one shape object to reduce module memory.
- **Example:**
  ```
  // before (conceptual): a = f32[1024,1024] add(...) ; b = f32[1024,1024] mul(...)   // separate identical Shapes
  // after (conceptual): a, b reference one pooled Shape (no IR/semantic change)
  ```

### `StripMemoryPlacementAnnotations`
- **Path:** `xla/hlo/transforms/strip_memory_placement_annotations.h` (impl `strip_memory_placement_annotations.cc`)
- **Description:** Strips memory placement annotation custom calls (`annotate_device_placement`, `MoveToHost`, `MoveToDevice`), replacing each with its input operand.
- **Example:**
  ```
  // before
  cc = f32[16] custom-call(x), custom_call_target="MoveToHost" ; ROOT r = add(cc, cc)
  // after
  ROOT r = f32[16] add(x, x)
  ```

### `WhileLoopTripCountAnnotator`
- **Path:** `xla/hlo/transforms/while_loop_trip_count_annotator.h` (impl `while_loop_trip_count_annotator.cc`)
- **Description:** Annotates `while` loops with `WhileLoopBackendConfig` metadata (induction-variable tuple index, trip count, init/step) by pattern-matching loop bodies/conditions; run after loop-modifying passes but before fusion/layout assignment.
- **Example:**
  ```
  // before
  ROOT w = (s32[],...) while(init), condition=cond, body=body   // cond: counter < 5
  // after
  ROOT w = while(init), ..., backend_config={"known_trip_count":{"n":"5"}, "known_induction_variable":{"tuple_index":"0"}}
  ```

---

## 5. `xla/service/` — Optimization / transformation passes

### `AllGatherDecomposer`
- **Path:** `xla/service/all_gather_decomposer.h` (impl `all_gather_decomposer.cc`)
- **Description:** Converts unsupported all-gathers into dynamic-update-slices and all-reduces. Legalization pass for backends without native all-gather.
- **Example:**
  ```
  // before
  ag = f32[8] all-gather(x), dimensions={0}
  // after (conceptual)
  dus = f32[8] dynamic-update-slice(zeros, x, offset) ; ag = f32[8] all-reduce(dus), to_apply=add
  ```

### `AllGatherSimplifier`
- **Path:** `xla/service/all_gather_simplifier.h` (impl `all_gather_simplifier.cc`)
- **Description:** Detects unnecessary all-gathers and replaces them with their operands (e.g. a trivial all-gather, or an all-gather feeding a single dynamic-slice that reproduces the input).
- **Example:**
  ```
  // before
  ag = f32[8] all-gather(x), dimensions={0} ; ds = f32[2] dynamic-slice(ag, offset)
  // after
  ds = f32[2] x
  ```

### `AllReducePromotion`
- **Path:** `xla/service/all_reduce_promotion.h` (impl `all_reduce_promotion.cc`)
- **Description:** Promotes the data type of all-reduce (and optionally reduce-scatter) from one primitive type to another (e.g. for higher-precision accumulation); a wrapper around `ChangeOpDataType`.
- **Example:**
  ```
  // before (bf16 -> f32)
  ar = bf16[8] all-reduce(x), to_apply=add
  // after
  ar = bf16[8] convert(f32[8] all-reduce(f32[8] convert(x)), to_apply=add)
  ```

### `AllReduceReassociate`
- **Path:** `xla/service/all_reduce_reassociate.h` (impl `all_reduce_reassociate.cc`)
- **Description:** Reassociates all-reduces feeding into a compatible elementwise op, e.g. `add(all-reduce(x), all-reduce(y))` → `all-reduce(add(x,y))`.
- **Example:**
  ```
  // before
  ar0 = all-reduce(x) ; ar1 = all-reduce(y) ; s = add(ar0, ar1)
  // after
  s = all-reduce(add(x, y)), to_apply=add
  ```

### `AllReduceReduceScatterReorder`
- **Path:** `xla/service/all_reduce_reduce_scatter_reorder.h` (impl `all_reduce_reduce_scatter_reorder.cc`)
- **Description:** Rewrites all-reduce followed by reduce-scatter into reduce-scatter followed by all-reduce, so the all-reduce operates on a smaller (post-scatter) tensor.
- **Example:**
  ```
  // before
  ar = f32[8] all-reduce(x) ; rs = f32[2] reduce-scatter(ar)
  // after
  rs = f32[2] reduce-scatter(x) ; ar = f32[2] all-reduce(rs)
  ```

### `AllReduceSimplifier`
- **Path:** `xla/service/all_reduce_simplifier.h` (impl `all_reduce_simplifier.cc`)
- **Description:** Uses replication analysis to detect all-reduces whose inputs are already identical across replicas and replaces them with local computations (e.g. sum all-reduce on replicated input → multiply by replica count).
- **Example:**
  ```
  // before (x replicated across N replicas)
  ar = f32[8] all-reduce(x), to_apply=add
  // after
  ar = f32[8] multiply(x, broadcast(N))
  ```

### `AllToAllDecomposer`
- **Path:** `xla/service/all_to_all_decomposer.h` (impl `all_to_all_decomposer.cc`)
- **Description:** Converts unsupported array all-to-all into tuple all-to-all (or into an array all-to-all of a minimum rank). Legalization `OpExpanderPass`.
- **Example:**
  ```
  // before
  a2a = f32[8] all-to-all(x), dimensions={0}   // unsupported array form
  // after (conceptual): reshape to higher-rank/tuple form, run all-to-all, reshape back
  ```

### `BatchedGatherScatterNormalizer`
- **Path:** `xla/service/batched_gather_scatter_normalizer.h` (impl `batched_gather_scatter_normalizer.cc`)
- **Description:** Rewrites batched gather and scatter operations into a non-batch (canonical) version. Normalization `OpExpanderPass`.
- **Example:**
  ```
  // before
  g = gather(operand, indices), operand_batching_dims={0}, start_indices_batching_dims={0}
  // after (conceptual): batch dims folded into explicit iota start-index operands; no *_batching_dims
  ```

### `BatchNormExpander`
- **Path:** `xla/service/batchnorm_expander.h` (impl `batchnorm_expander.cc`)
- **Description:** Rewrites batch-norm (training/inference/grad) into smaller primitive operations so generic fusion can handle them; can optionally emit a multi-output fusion node.
- **Example:**
  ```
  // before
  out = (..) batch-norm-training(x, scale, offset), epsilon=0.001, feature_index=1
  // after (conceptual): mean/variance via reduces, then (x - mean) * rsqrt(var + eps) * scale + offset
  ```

### `CallInliner`
- **Path:** `xla/service/call_inliner.h` (impl `call_inliner.cc`)
- **Description:** For every `kCall`, inlines the body of the called computation into the caller, recursively (with policy hooks for inlineable call sites).
- **Example:**
  ```
  // before
  c = f32[] call(a, b), to_apply={ ROOT add = add(p0, p1) }
  // after
  c = f32[] add(a, b)
  ```

### `CallMarker`
- **Path:** `xla/service/call_marker.h` (impl `call_marker.cc`)
- **Description:** Wraps `kCall` instructions with `__xla_internal_call_marker_before`/`_after` custom-calls so the later `CallOutliner` can identify them. Infrastructure paired with CallInliner/CallOutliner.
- **Example:**
  ```
  // before
  call = call(p0, p1), to_apply=my_computation
  // after
  before = custom-call(p0,p1), target="__xla_internal_call_marker_before", frontend_attributes={xla_call_marked_computation="my_computation"}
  call  = call(gte(before,0), gte(before,1)), to_apply=my_computation ; after = custom-call(call), target="..._after"
  ```

### `CallOutliner`
- **Path:** `xla/service/call_outliner.h` (impl `call_outliner.cc`)
- **Description:** Uses `CallMarker` markers to outline previously inlined blocks back into separate `kCall` computations. Inverse of inlining.
- **Example:**
  ```
  // before: marker_before ... body ... marker_after
  // after (conceptual): body extracted into outlined_computation; markers removed; c = call(operands), to_apply=outlined_computation
  ```

### `ChangeOpDataType`
- **Path:** `xla/service/change_op_data_type.h` (impl `change_op_data_type.cc`)
- **Description:** Rewrites `from_ty op(from_ty a, from_ty b)` into `from_ty convert(op(to_ty a, to_ty b))` for ops matching a predicate (e.g. run an fp16 dot/conv in fp32). Somewhat backend-oriented (used by XLA:CPU).
- **Example:**
  ```
  // before (f16 -> f32)
  d = f16[..] dot(a, b)
  // after
  d = f16[..] convert(f32[..] dot(f32[..] convert(a), f32[..] convert(b)))
  ```

### `CollectivePermuteDecomposer`
- **Path:** `xla/service/collective_permute_decomposer.h` (impl `collective_permute_decomposer.cc`)
- **Description:** Converts cycle-free collective-permute ops into Send/Recv pairs and annotates them for pipelining. Legalization tied to pipeline-parallelism lowering.
- **Example:**
  ```
  // before
  cp = <rt> collective-permute(data), source_target_pairs={...}
  // after (conceptual): after-all -> recv/send (+ control deps) -> recv-done/send-done -> gte(recv-done)
  ```

### `CollectivePipeliner`
- **Path:** `xla/service/collective_pipeliner.h` (impl `collective_pipeliner.cc`)
- **Description:** Peels loop iterations of stacked-layer models so collectives (reduce-scatter/all-reduce/all-gather) are pushed to the next iteration where they overlap with that iteration's compute. Supports forward / backward / forward-sink directions.
- **Example:**
  ```
  // before
  while (i<L) { x = comp(p0); xg = all-reduce(x); }
  // after (conceptual)
  x_prev = comp(p0) ; while (i<L, x_prev) { xg = all-reduce(x_prev); x = comp(p0); x_prev = x; }
  ```

### `ConditionalCodeMotion`
- **Path:** `xla/service/conditional_code_motion.h` (impl `conditional_code_motion.cc`)
- **Description:** Moves identical ops into or out of `kConditional` branches (governed by a tunable cost model); identical ops with non-shared operands can be hoisted out.
- **Example:**
  ```
  // before
  cond = conditional(p, x, y), true={ROOT add(a,b)}, false={ROOT add(c,d)} ; r = negate(cond)
  // after (conceptual): hoist the common 'negate' out of both branches (single shared op outside)
  ```

### `ConditionalSimplifier`
- **Path:** `xla/service/conditional_simplifier.h` (impl `conditional_simplifier.cc`)
- **Description:** Removes `kConditional` instructions whose predicate is a constant, replacing them with the chosen branch (inlined).
- **Example:**
  ```
  // before
  cond = conditional(true_const, x, y), true=T, false=F
  // after
  cond = call(x), to_apply=T   // false branch dropped
  ```

### `ConditionalToSelect`
- **Path:** `xla/service/conditional_to_select.h` (impl `conditional_to_select.cc`)
- **Description:** Transforms conditionals into selects where conditionals are legal but unsupported by the backend (e.g. inside `kMap`).
- **Example:**
  ```
  // before (inside a kMap body)
  c = conditional(pred, a, b), true=T, false=F
  // after
  c = select(pred, T(a), F(b))
  ```

### `CopyInsertion`
- **Path:** `xla/service/copy_insertion.h` (impl `copy_insertion.cc`)
- **Description:** Legalization pass that inserts `kCopy` to fix buffer-aliasing problems (params/constants live-out, simultaneously-live values forced to share a buffer, ambiguous entry roots). Correctness infrastructure.
- **Example:**
  ```
  // before
  ENTRY { p = f32[4] parameter(0); ROOT t = (f32[4]) tuple(p) }
  // after
  ENTRY { p = f32[4] parameter(0); c = f32[4] copy(p); ROOT t = (f32[4]) tuple(c) }
  ```

### `DynamicPadder`
- **Path:** `xla/service/dynamic_padder.h` (impl `dynamic_padder.cc`)
- **Description:** Uses `DynamicDimensionInference` to find dynamic shapes and inserts instructions to reset padding to an identity value before affected ops (e.g. reduce), plus PadToStatic/SliceToDynamic conversions. Dynamic-shape infrastructure.
- **Example:**
  ```
  // before (x has dynamic size; padding garbage)
  r = f32[] reduce(x, 0), to_apply=add
  // after (conceptual)
  masked = select(in_bounds_mask, x, 0) ; r = f32[] reduce(masked, 0), to_apply=add
  ```

### `GatherExpander`
- **Path:** `xla/service/gather_expander.h` (impl `gather_expander.cc`)
- **Description:** Rewrites gathers into (roughly) while loops of dynamic-slices; `kEliminateAllGathers` expands every gather, `kEliminateSimpleGathers` strength-reduces loop-free "simple" gathers to dynamic-slices.
- **Example:**
  ```
  // before (simple gather)
  g = f32[3] gather(operand, indices), ...
  // after (conceptual, kEliminateSimpleGathers)
  g = f32[3] dynamic-slice(operand, indices)   // general gathers -> while loop of dynamic-slices
  ```

### `HloCSE`
- **Path:** `xla/service/hlo_cse.h` (impl `hlo_cse.cc`)
- **Description:** Common-subexpression elimination: commons identical constants and identical instructions with the same operands, iterating in topological order so arbitrarily large common expressions are found. (XLA's GVN equivalent.)
- **Example:**
  ```
  // before
  a = f32[4] add(x, y) ; b = f32[4] add(x, y) ; c = f32[4] multiply(a, b)
  // after
  a = f32[4] add(x, y) ; c = f32[4] multiply(a, a)
  ```

### `HloModuleDCE`
- **Path:** `xla/service/hlo_module_dce.h` (impl `hlo_module_dce.cc`)
- **Description:** Removes dead code using module-scoped liveness analysis, sweeping live instructions that cross computation boundaries (`kWhile`) and removing code at dead shape indices.
- **Example:**
  ```
  // before: while loops over tuple(a, b) but only element 0 (a) is ever used
  // after (conceptual): dead tuple element b and its feeding instructions removed from body/condition/state
  ```

### `InstructionFusion`
- **Path:** `xla/service/instruction_fusion.h` (impl `instruction_fusion.cc`)
- **Description:** "Vertical" instruction fusion, fusing producers into consumers so their loops fuse in codegen; derived classes define `ShouldFuse` heuristics. Base-class infrastructure for backend fusion.
- **Example:**
  ```
  // before
  a = f32[4] add(x, y) ; m = f32[4] multiply(a, z)
  // after
  m = f32[4] fusion(x, y, z), kind=kLoop, calls={ add then multiply }
  ```

### `LayoutAssignment`
- **Path:** `xla/service/layout_assignment.h` (impl `layout_assignment.cc`)
- **Description:** Assigns layouts to all instructions while satisfying invariants (entry/result layout constraints) and minimizing cost. Infrastructure; backends subclass to add target-specific constraints.
- **Example:**
  ```
  // before
  t = f32[2,3] transpose(p), dimensions={1,0}   // layouts unassigned
  // after (conceptual): physical layouts chosen/propagated; kCopy inserted where a layout change is unavoidable
  ```

### `LayoutNormalization`
- **Path:** `xla/service/layout_normalization.h` (impl `layout_normalization.cc`)
- **Description:** Converts shapes to physically-equivalent "normalized" form (descending layout, no degenerate dims); a `CustomCallTransformer` lets backends supply custom-call rules.
- **Example:**
  ```
  // before
  x = f32[5,1,4]{0,1,2}
  // after
  x = f32[4,5]{1,0}   // descending layout, degenerate dim removed
  ```

### `MapInliner`
- **Path:** `xla/service/map_inliner.h` (impl `map_inliner.cc`)
- **Description:** Performs map inlining, replacing `kMap` instructions with the equivalent sequence of array operations.
- **Example:**
  ```
  // before
  m = f32[4] map(X, Y), to_apply={ ROOT add(p0,p1) }
  // after
  m = f32[4] add(X, Y)
  ```

### `MultiOutputFusion`
- **Path:** `xla/service/multi_output_fusion.h` (impl `multi_output_fusion.cc`)
- **Description:** Fuses sibling fusion instructions that share common operands (without duplicating the producer), via a profit-scored worklist and reachability map. Base-class infrastructure subclassed by backends.
- **Example:**
  ```
  // before
  f0 = f32[4] fusion(x) ; f1 = f32[4] fusion(x)   // share operand x
  // after (conceptual)
  mof = (f32[4], f32[4]) fusion(x), kind=kInput   // both outputs from one fusion
  ```

### `ScanExpander`
- **Path:** `xla/service/scan_expander.h` (impl `scan_expander.cc`)
- **Description:** Rewrites scan operations into a while loop (optionally expanding associative scans). Legalization `OpExpanderPass`.
- **Example:**
  ```
  // before
  s = scan(init, xs), to_apply=body
  // after (conceptual)
  s = while(tuple(i, init, xs, ys)) { slice xs[i], apply, write ys[i], i++ }
  ```

### `ScanLoopAccumulatorInputUnification`
- **Path:** `xla/service/scan_loop_accumulator_input_unification.h` (impl `scan_loop_accumulator_input_unification.cc`)
- **Description:** Finds nested-loop accumulator patterns (commonly from `jax.scan`) and unifies the accumulation buffer with its input, removing the unnecessary copy of the accumulation buffer.
- **Example:**
  ```
  // before
  acc = allocate-buffer() ; acc' = dus(acc, slice', i) ; copy_acc = copy(new_acc)
  // after
  acc' = dus(input, slice', i)   // accumulate directly into input; copy removed
  ```

### `ReduceScatterCombiner`
- **Path:** `xla/service/reduce_scatter_combiner.h` (impl `reduce_scatter_combiner.cc`)
- **Description:** Combines small, non-dependent reduce-scatter ops into a single larger combined reduce-scatter, amortizing per-op latency.
- **Example:**
  ```
  // before
  rs0 = reduce-scatter(a) ; rs1 = reduce-scatter(b)
  // after
  rs = reduce-scatter(a, b)   // single combined op, results unpacked via GTE
  ```

### `ReduceScatterDecomposer`
- **Path:** `xla/service/reduce_scatter_decomposer.h` (impl `reduce_scatter_decomposer.cc`)
- **Description:** Decomposes a reduce-scatter into an all-reduce followed by a dynamic-slice (driven by an optional `should_decompose` predicate). For backends lacking native reduce-scatter.
- **Example:**
  ```
  // before
  rs = f32[4] reduce-scatter(x), to_apply=add
  // after
  ar = f32[8] all-reduce(x), to_apply=add ; rs = f32[4] dynamic-slice(ar, partition_id*4)
  ```

### `ReduceScatterReassociate`
- **Path:** `xla/service/reduce_scatter_reassociate.h` (impl `reduce_scatter_reassociate.cc`)
- **Description:** Reassociates two reduce-scatters feeding a compatible elementwise op into a single reduce-scatter applied to the combined operands.
- **Example:**
  ```
  // before
  add(reduce-scatter(x), reduce-scatter(y))
  // after
  reduce-scatter(add(x, y))
  ```

### `ScatterExpander`
- **Path:** `xla/service/scatter_expander.h` (impl `scatter_expander.cc`)
- **Description:** Rewrites scatter ops into while loops of dynamic-update-slices; configurable to eliminate all scatters, only "simple" ones, or only potentially indeterministic ones.
- **Example:**
  ```
  // before
  s = scatter(operand, indices, updates), to_apply=add
  // after (conceptual)
  s = while { operand = dynamic-update-slice(operand, add(slice(operand), update_i), index_i) }
  ```

### `ScatterSimplifier`
- **Path:** `xla/service/scatter_simplifier.h` (impl `scatter_simplifier.cc`)
- **Description:** Normalizes a general scatter into a canonical "simple" scatter via surrounding transposes/reshapes (making `scatter_dims_to_operand_dims` the identity), without lowering to a loop.
- **Example:**
  ```
  // before (conceptual): scatter(operand, indices, updates) with arbitrary dim numbers
  // after: t = transpose/reshape(...) ; s = scatter(t...)   // canonical: identity dim mapping, single index vector dim
  ```

### `SelectAndScatterExpander`
- **Path:** `xla/service/select_and_scatter_expander.h` (impl `select_and_scatter_expander.cc`)
- **Description:** Rewrites select-and-scatter into an explicit windowed reduction (the "select" step) followed by a scatter (the "scatter" step).
- **Example:**
  ```
  // before (conceptual)
  out = select-and-scatter(operand, source), select=GE, scatter=add
  // after: reduce-window picks argmax per window, then scatter adds each source value into that position
  ```

### `ShardingPropagation`
- **Path:** `xla/service/sharding_propagation.h` (impl `sharding_propagation.cc`)
- **Description:** Propagates `sharding` annotations across the graph via a local greedy fixpoint heuristic from operands/users. SPMD-partitioning infrastructure (not a target-independent compute optimization).
- **Example:**
  ```
  // before
  p0 = f32[8,8] parameter(0), sharding={devices=[2,1]<=[2]} ; add = add(p0, p0)   // no sharding
  // after
  add = f32[8,8] add(p0, p0), sharding={devices=[2,1]<=[2]}   // inferred
  ```

### `ShardingRemover`
- **Path:** `xla/service/sharding_remover.h` (impl `sharding_remover.cc`)
- **Description:** Removes `Sharding` custom-call instructions by rewiring users directly to the operand; useful when `partition_count == 1`.
- **Example:**
  ```
  // before
  s = f32[8] custom-call(x), custom_call_target="Sharding" ; y = add(s, s)
  // after
  y = f32[8] add(x, x)
  ```

### `SpaceToBatchConverter`
- **Path:** `xla/service/space_to_batch_converter.h` (impl `space_to_batch_converter.cc`)
- **Description:** Rewrites convolutions so a spatial dimension is folded into the batch dimension (space-to-batch), propagating through chains of conv layers to improve utilization on small spatial sizes.
- **Example:**
  ```
  // before (conceptual): conv(activations[N,H,W,C], kernel)   // small spatial dim
  // after: split spatial dim into batch -> [N*splits, H, W/splits, C] -> conv -> batch-to-space
  ```

### `TopkRewriter` / `TopkDecomposer`
- **Path:** `xla/service/topk_rewriter.h` (impl `topk_rewriter.cc`)
- **Description:** `TopkRewriter` pattern-matches a sort+iota+slice "soup" implementing a TopK and replaces it with a `TopK` custom-call when supported/profitable; `TopkDecomposer` does the inverse, lowering a TopK custom-call back to sort-based HLO.
- **Example:**
  ```
  // before
  s = sort(values, iota), dimension=1, is_stable, comparator=GT
  topv = slice(gte(s,0), [0:k]) ; topi = slice(gte(s,1), [0:k])
  // after
  ck = (f32[..,k], s32[..,k]) custom-call(values), custom_call_target="TopK", k=k
  ```

### `TransposeFolding`
- **Path:** `xla/service/transpose_folding.h` (impl `transpose_folding.cc`)
- **Description:** Folds transpose operands into dot (and optionally convolution) ops, relying on the GEMM kernel's ability to transpose its inputs, removing the explicit transpose.
- **Example:**
  ```
  // before
  t = f32[4,3] transpose(a), dimensions={1,0} ; d = f32[4,5] dot(t, b), lhs_contracting={1}, rhs_contracting={0}
  // after
  d = f32[4,5] dot(a, b), lhs_contracting={0}, rhs_contracting={0}   // transpose folded into dot dims
  ```

### `TriangularSolveExpander`
- **Path:** `xla/service/triangular_solve_expander.h` (impl `triangular_solve_expander.cc`)
- **Description:** Expands `triangular-solve` into elementary HLO: a blocked algorithm (MAGMA-style) that inverts diagonal blocks and combines them with matrix multiplies, or a direct solve for small matrices.
- **Example:**
  ```
  // before (conceptual)
  x = triangular-solve(a, b), left_side, lower, unit_diagonal
  // after: partition a into diagonal blocks, invert each block, propagate via dot across blocks
  ```

### `WhileLoopAllReduceCodeMotion`
- **Path:** `xla/service/while_loop_all_reduce_code_motion.h` (impl `while_loop_all_reduce_code_motion.cc`)
- **Description:** Sinks all-reduce (optionally reduce-scatter) ops out of while-loop bodies when their result is only accumulated into a buffer, so a single collective runs after the loop.
- **Example:**
  ```
  // before (conceptual)
  while { b = ...; c = all-reduce(b); a += c }
  // after
  d = 0 ; while { b = ...; d += b } ; a += all-reduce(d)
  ```

### `WhileLoopConcatCodeMotion`
- **Path:** `xla/service/while_loop_concat_code_motion.h` (impl `while_loop_concat_code_motion.cc`)
- **Description:** Lifts concatenation out of a while loop, rewriting piecewise per-element subcomputations in the body to operate on the single concatenated shape, slicing results back out after the loop.
- **Example:**
  ```
  // before (conceptual)
  while (a, b) { e = concat(a,b); f = op(e); a' = slice(f); b' = slice(f) }
  // after
  ab = concat(a, b) ; while (ab) { f = op(ab); ab' = ... } ; a' = slice(ab') ; b' = slice(ab')
  ```

### `WhileLoopConstantSinking`
- **Path:** `xla/service/while_loop_constant_sinking.h` (impl `while_loop_constant_sinking.cc`)
- **Description:** Sinks loop-invariant constants from the loop's input tuple into the body/condition (replacing the tuple-element use with the constant), unlocking constant folding. Leaves the tuple element for `WhileLoopSimplifier` to remove.
- **Example:**
  ```
  // before
  state = (..., const, ...) ; while { (..., v, ...) = state; use(v) }
  // after
  while { (..., v, ...) = state; use(const) }   // v still carried; later removed
  ```

### `WhileLoopExpensiveInvariantCodeMotion`
- **Path:** `xla/service/while_loop_expensive_invariant_code_motion.h` (impl `while_loop_expensive_invariant_code_motion.cc`)
- **Description:** Hoists groups of expensive, non-size-inflating loop-invariant instructions out of a while body, gated by a user `worth_hoisting_individually` predicate.
- **Example:**
  ```
  // before
  while { inv = expensive-op(loop_invariant_param); ... use(inv) }
  // after
  inv = expensive-op(loop_invariant_param) ; while { ... use(inv) }
  ```

### `WhileLoopFusibleSinking`
- **Path:** `xla/service/while_loop_fusible_sinking.h` (impl `while_loop_fusible_sinking.cc`)
- **Description:** Sinks loop-invariant fusible subgraphs (and broadcast-of-constant initializers) into the while body so they can later fuse; for fully-overwritten constant buffers replaces the broadcast with a free `AllocateBuffer` + first-iteration select.
- **Example:**
  ```
  // before
  state = (..., fusible_graph, ...) ; while { (..., v, ...) = state; use(v) }
  // after
  while { ... use(fusible_graph) }   // recomputed inside, fuses later
  ```

### `WhileLoopInvariantCodeMotion`
- **Path:** `xla/service/while_loop_invariant_code_motion.h` (impl `while_loop_invariant_code_motion.cc`)
- **Description:** Classic loop-invariant code motion for while loops: hoists invariant instructions from the body into the enclosing computation, with knobs for constants, reshapes, and a size-inflation limit.
- **Example:**
  ```
  // before
  while { inv = add(p_invariant, p_invariant); x = mul(inv, loop_var); ... }
  // after
  inv = add(p_invariant, p_invariant) ; while { x = mul(inv, loop_var); ... }
  ```

### `WhileLoopPipelineUnroller`
- **Path:** `xla/service/while_loop_pipeline_unroller.h` (impl `while_loop_pipeline_unroller.cc`)
- **Description:** Unrolls a pipelined while loop exactly enough times (its "pipeline depth") so input lifetimes end before outputs are materialized, removing aliasing interference and the copies copy-insertion would otherwise add.
- **Example:**
  ```
  // before (conceptual): pipelined inputs shift position each iter -> forced copies
  // after: while { body; body; ... }   // unrolled pipeline_depth times so lifetimes close, no interference copies
  ```

### `WhileLoopSimplifier`
- **Path:** `xla/service/while_loop_simplifier.h` (impl `while_loop_simplifier.cc`)
- **Description:** Simplifies while loops: deletes trip-count-0 loops, inlines trip-count-1 loops, removes unused tuple elements, flattens nested-tuple params, optionally folds trivially-true/false body compares.
- **Example:**
  ```
  // before
  w = while(init), condition=cond, body=body   // static trip count == 1
  // after
  w = body(init)   // loop removed, body inlined once
  ```

### `WhileLoopUnroller`
- **Path:** `xla/service/while_loop_unroller.h` (impl `while_loop_unroller.cc`)
- **Description:** Unrolls while loops with a known trip count (currently full unrolling), subject to trip-count/instruction-count/expansion thresholds.
- **Example:**
  ```
  // before
  while (i=0; i<3) { body(i); i++ }
  // after
  body(0); body(1); body(2)   // fully unrolled
  ```

---

## 6. `xla/service/` — Infrastructure / scheduling / verification

These are included for completeness; they are **not** target-independent compute optimizations.

### `ControlDepRewriter`
- **Path:** `xla/service/control_dep_rewriter.h` (impl `control_dep_rewriter.cc`)
- **Description:** Converts `"control_dep"` custom-calls into real HLO control-dependency edges between the named operands, then erases the custom-calls.
- **Example:**
  ```
  // before
  exp = exponential(x) ; cos = cosine(y) ; _ = custom-call(exp, cos), custom_call_target="control_dep"
  // after
  exp = exponential(x) ; cos = cosine(y), control-predecessors={exp}
  ```

### `HloCycleDetection`
- **Path:** `xla/service/hlo_cycle_detection.h` (impl `hlo_cycle_detection.cc`)
- **Description:** Verification pass that walks each computation via post-order DFS to detect cycles; never modifies instructions.
- **Example:** _No IR transformation — analysis/verification only._

### `HloDomainIsolator`
- **Path:** `xla/service/hlo_domain_isolator.h` (impl `hlo_domain_isolator.cc`)
- **Description:** Inserts `kDomain` instructions on graph edges connecting instructions with differing attributes (e.g. sharding), partitioning the graph into uniform-attribute domains. Sharding/domain infrastructure.
- **Example:**
  ```
  // before (conceptual): a = op(...), sharding=S1 ; b = op(a), sharding=S2
  // after: a (S1) ; d = domain(a) ; b = op(d), sharding=S2
  ```

### `HloDomainRemover`
- **Path:** `xla/service/hlo_domain_remover.h` (impl `hlo_domain_remover.cc`)
- **Description:** Removes all `kDomain` instructions of a given kind, invoking a normalizer to propagate the domain's attributes onto exposed instructions. Inverse of `HloDomainIsolator`.
- **Example:**
  ```
  // before (conceptual): d = domain(a) ; b = op(d)
  // after: b = op(a)   // kDomain removed; sharding normalized
  ```

### `HloDomainVerifier`
- **Path:** `xla/service/hlo_domain_verifier.h` (impl `hlo_domain_verifier.cc`)
- **Description:** Verification pass checking that `kDomain` instructions are consistent and each domain is bounded by matching metadata.
- **Example:** _No IR transformation — analysis/verification only._

### `HloVerifier`
- **Path:** `xla/service/hlo_verifier.h` (impl `hlo_verifier.cc`)
- **Description:** Core verification pass (driven by `ShapeVerifier` + `HloVerifierOpts`) checking shape/layout consistency, operand counts, mixed-precision rules, and structural invariants.
- **Example:** _No IR transformation — analysis/verification only._

### `LatencyHidingScheduler`
- **Path:** `xla/service/latency_hiding_scheduler.h` (impl `latency_hiding_scheduler.cc`)
- **Description:** Reorders instructions to overlap long-latency (especially async collective) operations with independent compute. Produces a schedule; does not change the dataflow graph.
- **Example:**
  ```
  // before (schedule): ar-start; ar-done; compute   // collective blocks compute
  // after: ar-start; compute; ar-done               // compute scheduled in the latency gap
  ```

### `LegalizeSchedulingAnnotations` / `CheckNoDataDependencyInSchedulingAnnotations`
- **Path:** `xla/service/legalize_scheduling_annotations.h` (impl `legalize_scheduling_annotations.cc`)
- **Description:** Legalizes/propagates scheduling-annotation groups used by `LatencyHidingScheduler` (filling gaps, removing trivial groups, verifying start/done consistency); the companion pass checks for direct data dependencies between annotated instructions.
- **Example:** _No IR transformation — annotation legalization/verification only._

### `LoopScheduleLinearizer`
- **Path:** `xla/service/loop_schedule_linearizer.h` (impl `loop_schedule_linearizer.cc`)
- **Description:** Adds control-dependency edges from value "writers" to "readers" inside a loop to avoid extraneous copies, best-effort and only when it can prove no cycle is introduced.
- **Example:**
  ```
  // before (conceptual): write_buf, read_buf unordered -> copy-insertion adds copies
  // after: read_buf = ..., control-predecessors={write_buf}   // ordered so buffers can be shared
  ```

### `P2PSchedulePreparation`
- **Path:** `xla/service/p2p_schedule_preparation.h` (impl `p2p_schedule_preparation.cc`)
- **Description:** Linearizes point-to-point Send/Recv chains by adding control dependencies so any later scheduler avoids P2P deadlocks. (Pass name string: `latency-hiding-scheduler-preparation`.)
- **Example:**
  ```
  // before (conceptual): recv; send; recv-done; send-done   // unordered
  // after: recv; send (ctrl recv); recv-done (ctrl send); send-done (ctrl recv-done)
  ```

### `XlaTransformBase` / `HloXlaTransform` / `ApplyXlaTransforms`
- **Path:** `xla/service/xla_transform.h` (impl `xla_transform.cc`)
- **Description:** A registry and `HloModulePass` (`ApplyXlaTransforms`) that runs user-registered `HloXlaTransform` callbacks at a chosen pipeline stage. Generic extensibility hook with no fixed transformation of its own.
- **Example:** _No IR transformation — infrastructure hook; effect depends on the registered transform._

---

## 7. `xla/service/spmd/` and `xla/service/debug/` — SPMD & debug

These are multi-device (SPMD) passes plus one debug pass.

### `SpmdPartitioner`
- **Path:** `xla/service/spmd/spmd_partitioner.h` (impl `spmd_partitioner.cc`)
- **Description:** The core SPMD partitioning pass: rewrites a sharding-annotated whole-module program into per-device (per-partition) computations, inserting the necessary collectives (all-gather, all-reduce, collective-permute, ...) to preserve semantics across `num_partitions` devices.
- **Example:**
  ```
  // before
  ENTRY e { p = f32[8,8] parameter(0), sharding={devices=[2,1]0,1} ; ROOT add = f32[8,8] add(p, p), sharding={devices=[2,1]0,1} }
  // after (conceptual: per-partition on a [4,8] local shard)
  ENTRY e { p = f32[4,8] parameter(0) ; ROOT add = f32[4,8] add(p, p) }
  ```

### `SpmdPrepare`
- **Path:** `xla/service/spmd/spmd_prepare.h` (impl `spmd_prepare.cc`)
- **Description:** Preparation pass that rewrites sharded ops into forms that partition better (e.g. merging concatenated index/update operands feeding a scatter), interleaved with other sharding optimizations before the main partitioner.
- **Example:**
  ```
  // before (two scatter operands built from separate concatenates) -> scatter(add, concat(c0,c1), concat(p3,p4))
  // after (conceptual): s0 = scatter(add, c0, p3) ; ROOT scatter(s0, c1, p4)
  ```

### `CanonicalizeAllGatherForCSE`
- **Path:** `xla/service/spmd/canonicalize_all_gather_for_cse.h` (impl `canonicalize_all_gather_for_cse.cc`)
- **Description:** SPMD pass that canonicalizes `all-gather`s so equivalent gathers become syntactically identical (enabling CSE), by pushing degenerate-dimension reshapes through the all-gather.
- **Example:**
  ```
  // before
  resh = s32[1,8] reshape(param0) ; ROOT ag = s32[2,8] all-gather(resh), replica_groups={{0,1}}, dimensions={0}
  // after
  ROOT reshape(all-gather(param0))   // gather moved before reshape
  ```

### `CollectivePermuteMotion`
- **Path:** `xla/service/spmd/collective_permute_motion.h` (impl `collective_permute_motion.cc`)
- **Description:** SPMD loop optimization that moves a `collective-permute` from the end of a while-loop body to the beginning (overlapping it with compute), adding a compensating collective-permute after the loop.
- **Example:**
  ```
  // before: body { ... mul; cp = collective-permute(mul); ROOT tuple(..., cp) }
  // after (conceptual): body { cp = collective-permute(prev); ... } ; result = collective-permute(gte(while))
  ```

### `PartitionAssignment`
- **Path:** `xla/service/spmd/partition_assignment.h` (impl `partition_assignment.cc`)
- **Description:** SPMD pass that assigns shardings to a select set of the costliest HLOs (leaving propagation to spread them), via a pluggable `PartitioningAlgorithm` (the only provided one, `NoopPartitioning`, makes no changes).
- **Example:**
  ```
  // (conceptual) default NoopPartitioning assigns no shardings:
  // before: unsharded module -> after: unchanged
  ```

### `ScheduleAwareCollectiveOpsCSE`
- **Path:** `xla/service/spmd/schedule_aware_collective_ops_cse.h` (impl `schedule_aware_collective_ops_cse.cc`)
- **Description:** SPMD CSE on collective ops (e.g. duplicate `all-gather`s), but only when duplicates lie within a configurable live-range `distance_threshold`; targets cross-replica or cross-partition collectives.
- **Example:**
  ```
  // before
  ag1 = s32[2,8] all-gather(param0) ; ag2 = s32[2,8] all-gather(param0) ; use = add(ag1, ag2)
  // after
  ag1 = s32[2,8] all-gather(param0) ; use = add(ag1, ag1)
  ```

### `ShardingFormatPicker`
- **Path:** `xla/service/spmd/sharding_format_picker.h` (impl `sharding_format_picker.cc`)
- **Description:** Test-only SPMD pass that re-encodes every instruction's `HloSharding` to a selected internal format (V1, V2, or V3/Named) without changing the logical sharding. Used to exercise the partitioner.
- **Example:**
  ```
  // before / after: same logical sharding, re-encoded — e.g. sharding={devices=[2,2]0,1,2,3} -> equivalent V2/Named form
  ```

### `WholeGraphManualPass`
- **Path:** `xla/service/spmd/whole_graph_manual_pass.h` (impl `whole_graph_manual_pass.cc`)
- **Description:** SPMD pass that stamps every instruction with a `{manual}` sharding, marking the whole graph as manually sharded so a subsequent partitioner run leaves it unpartitioned.
- **Example:**
  ```
  // before
  resh1 = s32[1,8] reshape(g1) ; ROOT t = tuple(...), sharding={{devices=[1,4]0,1,2,3}, ..., {replicated}}
  // after
  resh1 = s32[1,8] reshape(g1), sharding={manual} ; ROOT t = tuple(...), sharding={{manual}, {manual}, {manual}}
  ```

### `UnstableReductionDetector`
- **Path:** `xla/service/debug/unstable_reduction_detector.h` (impl `unstable_reduction_detector.cc`)
- **Description:** Debug-only pass that scans for numerically unstable reductions (a `reduce` accumulating something other than max/min into a sub-f32 float type) and, per `xla_detect_unstable_reductions`, either logs or crashes.
- **Example:** _No IR transformation — analysis only._
