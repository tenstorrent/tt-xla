// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_CLEAN_FOR_XLA_INGESTION_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_CLEAN_FOR_XLA_INGESTION_H_

// llvm mlir includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

// tt-xla includes
#include "utils/status.h"

namespace tt::pjrt::module_builder::frontend_passes {

// Sanitizes a module for XLA ingestion via PJRT_Executable_OptimizedProgram.
// XLA cannot parse sdy dialect ops, ttcore/ttir attributes, or debug locations,
// so these must be stripped before the module is returned as optimized code.
//
// Always performed (both shardy and non-shardy paths):
// 1. ttcore and ttir dialect attributes are stripped from function arguments
//    and results.
// 2. Location information is stripped from the module.
// 3. sdy.mesh operations are removed.
//
// Performed only when a ManualComputationOp is present (shardy path):
// 4. The sdy.manual_computation op is stripped by deleting its body and
//    replacing its results with dummy outputs in the correct shape and order.
// 5. Output shardings are injected as a moduleOp attr,
//    mhlo.spmd_output_shardings. This is required by XLA to correctly parse
//    the output shardings of the module.
//
// For an example of the shardy transformation, see
// https://github.com/openxla/xla/issues/34830#issuecomment-3706288785
tt_pjrt_status
cleanForXlaIngestion(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_CLEAN_FOR_XLA_INGESTION_H_
