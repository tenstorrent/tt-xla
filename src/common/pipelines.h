// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_SRC_COMMON_PIPELINES_H_
#define TT_XLA_SRC_COMMON_PIPELINES_H_

// llvm mlir includes
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

namespace tt::pjrt::pipelines {

const std::string kInputRoleAttrString = "tt.input_role";

// Runs TT-XLA specific pipelines on the MLIR module.
void runTTXLAPipelines(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Propagates tt.input_role attributes from func.call operations to function
// arguments.
void propagateRoleAttributes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Helper function to recursively propagate tt.input_role attribute upward
// through call chain.
void propagateRoleAttribute(mlir::ModuleOp module, mlir::Value value,
                            mlir::StringAttr roleAttr);

// Inlines all private tt.mark_* functions to eliminate unnecessary function
// calls.
void inlineTTMarkFunctions(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Check is the function is tt_mark.
bool isTTMarkFunction(const std::string &function_name);

void legalizeStablehloMarkCompositeToCall(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

} // namespace tt::pjrt::pipelines

#endif // TT_XLA_SRC_COMMON_PIPELINES_H_
