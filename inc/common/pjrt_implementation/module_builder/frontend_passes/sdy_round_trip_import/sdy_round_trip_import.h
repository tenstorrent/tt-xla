// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SDY_ROUND_TRIP_IMPORT_SDY_ROUND_TRIP_IMPORT_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SDY_ROUND_TRIP_IMPORT_SDY_ROUND_TRIP_IMPORT_H_

// llvm mlir includes
#include "mlir/Pass/PassManager.h"

namespace tt::pjrt::module_builder::frontend_passes::sdy_round_trip_import {

// Adds the complete SDY round trip import pipeline to the pass manager.
// This pipeline imports SDY sharding attributes and handles mesh operations.
void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport,
                                   bool importFuncCalls,
                                   bool liftAndDedupMeshes);

namespace internal {

// Adds common pre-import passes to the pass manager.
void addCommonPreImportPasses(mlir::OpPassManager& pm, bool enableConstantImport);

// Adds common post-import passes to the pass manager.
void addCommonPostImportPasses(mlir::OpPassManager& pm, bool importFuncCalls);

// Creates a pass that clones manual computation calls so each call gets its own unique function.
std::unique_ptr<mlir::Pass> createSdyRoundTripCloneManualComputationCallsPass();

// Creates a pass that deduplicates meshes with identical device configurations but different names.
std::unique_ptr<mlir::Pass> createSdyRoundTripDedupMeshesPass();

} // namespace internal

} // namespace tt::pjrt::module_builder::frontend_passes::sdy_round_trip_import

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SDY_ROUND_TRIP_IMPORT_SDY_ROUND_TRIP_IMPORT_H_