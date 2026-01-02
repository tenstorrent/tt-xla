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

// Strips ttcore dialect attributes from function arguments and results.
// This is necessary before passing MLIR to XLA ingestion, as XLA does not
// understand ttcore-specific attributes like argument_type and shard_status.
tt_pjrt_status cleanForXlaIngestion(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_CLEAN_FOR_XLA_INGESTION_H_
