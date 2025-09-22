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

// tt-xla includes
#include "common/status.h"

namespace tt::pjrt::module_builder::frontend_passes {

// Annotates the attributes of the function arguments (argument type, name) via
// observation of the annotation ops inserted by the frontend(s).
tt_pjrt_status
annotateArgumentAttributes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

namespace internal {

// Propagates ttcore.argument_type attributes from tt.mark func.call operations
// upwards to the module root public function arguments.
void propagateInputRoleAttributes(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Helper function to recursively propagate ttcore.argument_type attribute
// upward through call chain.
void propagateRoleAttribute(mlir::ModuleOp module, mlir::Value argument,
                            mlir::StringAttr roleAttr);

// Inlines all private tt.mark_* functions to eliminate unnecessary function
// calls.
void inlineTTMarkFunctions(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Check is the function is tt_mark.
bool isTTMarkFunction(const std::string &function_name);

// Annotates the attributes of the function arguments.
tt_pjrt_status annotateArgumentAttributesFromCustomCall(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

// Sets default role for function arguments that have not been annotated.
// Currently the default role is Input.
void setDefaultRoleForUnannotatedArguments(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

} // namespace internal

} // namespace tt::pjrt::module_builder::frontend_passes

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_FRONTEND_PASSES_SHLO_INPUT_ROLE_PROPAGATION_H_
