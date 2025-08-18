// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/module_builder/frontend_passes/shlo_input_role_propagation.h"

// c++ standard library includes
#include <cassert>

// loguru includes
#include "loguru/loguru.hpp"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

// tt-xla includes
#include "common/status.h"

namespace tt::pjrt::module_builder::frontend_passes {

const std::string c_input_role_attr_name = "tt.input_role";

void propagateInputRoleAttributes(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::ModuleOp module = mlir_module.get();

  // Propagate role attributes through the call graph
  module.walk([&](mlir::func::CallOp call_op) {
    auto role_attr =
        call_op->getAttrOfType<mlir::StringAttr>(c_input_role_attr_name);
    if (role_attr) {
      for (mlir::Value operand : call_op.getOperands()) {
        internal::propagateRoleAttribute(module, operand, role_attr);
      }
    }
  });

  // Inline all private tt.mark_* functions to eliminate unnecessary calls.
  internal::inlineTTMarkFunctions(mlir_module);
}

namespace internal {

const std::string c_tt_mark_function_prefix = "tt.mark_";

void propagateRoleAttribute(mlir::ModuleOp module, mlir::Value value,
                            mlir::StringAttr role_attr) {
  // We are marking only function arguments so we expect call op operands to be
  // block arguments of the FuncOp.
  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  assert(block_arg && "CallOp operand not a BlockArgument");
  auto *parent_op = block_arg.getOwner()->getParentOp();
  auto arg_index = block_arg.getArgNumber();
  auto parent_func_op = mlir::dyn_cast<mlir::func::FuncOp>(parent_op);
  assert(parent_func_op && "CallOp operand not a BlockArgument of FuncOp");

  parent_func_op.setArgAttr(arg_index, c_input_role_attr_name, role_attr);

  // Find all call sites of this function and propagate upward.
  auto funcName = parent_func_op.getSymName();
  module.walk([&](mlir::func::CallOp call_op) {
    if (call_op.getCallee() == funcName &&
        arg_index < call_op.getNumOperands()) {
      mlir::Value callerArg = call_op.getOperand(arg_index);
      propagateRoleAttribute(module, callerArg, role_attr);
    }
  });
}

void inlineTTMarkFunctions(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  std::vector<mlir::func::FuncOp> functions_to_remove;

  mlir_module->walk([&](mlir::func::FuncOp func_op) {
    std::string func_name = func_op.getSymName().str();
    if (isTTMarkFunction(func_name) && func_op.isPrivate()) {
      functions_to_remove.push_back(func_op);
    }
  });

  mlir_module->walk([&](mlir::func::CallOp call_op) {
    auto callee_attr = call_op.getCalleeAttr();
    if (!callee_attr) {
      return;
    }

    std::string callee_name = callee_attr.getValue().str();
    if (!isTTMarkFunction(callee_name) || call_op.getNumOperands() != 1 ||
        call_op.getNumResults() != 1) {
      return;
    }

    // This is a call to a tt.mark_* function - inline it.
    mlir::Value operand = call_op.getOperand(0);
    call_op.getResult(0).replaceAllUsesWith(operand);
    call_op.erase();
  });

  // Remove the now-unused tt.mark_* function definitions.
  for (auto func_op : functions_to_remove) {
    std::string func_name = func_op.getSymName().str();
    func_op.erase();
  }
}

bool isTTMarkFunction(const std::string &function_name) {
  // Check if function name starts with required prefix.
  return function_name.rfind(c_tt_mark_function_prefix, 0) == 0;
}

} // namespace internal

} // namespace tt::pjrt::module_builder::frontend_passes
