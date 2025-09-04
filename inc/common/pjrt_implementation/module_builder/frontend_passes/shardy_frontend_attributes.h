// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// llvm mlir includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace tt::pjrt::module_builder::frontend_passes {

// Frontend pass for consolidating Shardy sharding attributes from custom calls
// to CallOps. Transfers input/output sharding specifications and manual axes
// from global-to-local and local-to-global custom calls onto the central manual
// computation CallOp, then applies Shardy round-trip import passes.
void applyShardyFrontendAttributesPasses(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    mlir::PassManager &pipeline_pm);

} // namespace tt::pjrt::module_builder::frontend_passes
