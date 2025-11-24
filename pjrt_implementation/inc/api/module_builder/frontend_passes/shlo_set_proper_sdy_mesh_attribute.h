// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_SET_PROPER_SDY_MESH_ATTRIBUTE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_SET_PROPER_SDY_MESH_ATTRIBUTE_H_

// llvm mlir includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

// tt-xla includes
#include "utils/status.h"

namespace tt::pjrt::module_builder::frontend_passes {
// In torch using SPMD mode, a fully replicated graph may default to a
// degenerate mesh [1,1], which is not what we want—every device should run with
// the same inputs/weights. This function detects that case and rewrites the
// mesh to [1, num_devices] so fully replicated graphs execute as intended.
tt_pjrt_status setProperSdyMeshAttributeInSpmdMode(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

namespace internal {
// Checks whether the graph is in SPMD mode.
// If its arguments contain the "mhlo.sharding" attribute, it is considered to
// be in SPMD mode.
bool isSpmdMode(const mlir::OwningOpRef<mlir::ModuleOp> &module);

} // namespace internal

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_FRONTEND_PASSES_SHLO_SET_PROPER_SDY_MESH_ATTRIBUTE_H_
