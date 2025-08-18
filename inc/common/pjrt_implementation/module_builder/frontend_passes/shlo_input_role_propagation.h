// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SHLO_INPUT_ROLE_PROPAGATION_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SHLO_INPUT_ROLE_PROPAGATION_H_

// c++ standard library includes
#include <cstring>

// llvm mlir includes
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

namespace tt::pjrt::module_builder::frontend_passes {

extern const std::string c_input_role_attr_name;

// Propagates tt.input_role attributes from func.call operations to function
// arguments.
void propagateInputRoleAttributes(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

namespace internal {

// Helper function to recursively propagate tt.input_role attribute upward
// through call chain.
void propagateRoleAttribute(mlir::ModuleOp module, mlir::Value value,
                            mlir::StringAttr roleAttr);

// Inlines all private tt.mark_* functions to eliminate unnecessary function
// calls.
void inlineTTMarkFunctions(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Check is the function is tt_mark.
bool isTTMarkFunction(const std::string &function_name);

} // namespace internal

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SHLO_INPUT_ROLE_PROPAGATION_H_
