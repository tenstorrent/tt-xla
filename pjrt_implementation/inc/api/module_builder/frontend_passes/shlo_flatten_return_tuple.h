// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_FLATTEN_RETURN_TUPLE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_FLATTEN_RETURN_TUPLE_H_

// llvm mlir includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

// tt-xla includes
#include "utils/status.h"

namespace tt::pjrt::module_builder::frontend_passes {

// Flattens tuple-typed results of public (entry) functions.
//
// XLA computations use a tuple root at the StableHLO boundary, so the framework
// (e.g. torch-xla) hands us modules whose entry function looks like:
//   func.func @main(...) -> tuple<tensor<..>, ..> {
//     %t = stablehlo.tuple %a, %b : tuple<..>
//     return %t : tuple<..>
//   }
// The rest of the plugin (PJRT output metadata collection) and the SHLO->TTIR
// lowering expect one flat tensor result per output, not a single tuple. This
// pass rewrites each public function returning a tuple into one returning the
// tuple's leaf tensors:
//   func.func @main(...) -> (tensor<..>, ..) {
//     ...
//     return %a, %b : tensor<..>, ..
//   }
// inserting stablehlo.get_tuple_element ops as needed and recursing through
// nested tuples. The now-redundant stablehlo.tuple / get_tuple_element ops are
// cleaned up later by the tt-mlir DecomposeCustomCallTuples pass. Functions
// whose results are already flat tensors are left unchanged.
//
// Must run before any PJRT output-metadata collection and before the tt-mlir
// StableHLO compiler pipeline.
tt_pjrt_status
flattenReturnTuple(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_FLATTEN_RETURN_TUPLE_H_
