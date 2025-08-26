// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/module_builder/frontend_passes/shlo_input_role_propagation.h"

// c++ standard library includes
#include <cassert>

// llvm includes
#include "llvm/ADT/StringRef.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

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

void propagateRoleAttribute(mlir::ModuleOp module, mlir::Value argument,
                            mlir::StringAttr role_attr) {
  // We are marking only inputs to model's `apply` function, so we expect call
  // op arguments to be block arguments of the FuncOp corresponding to `apply`
  // function in the model graph. Sometimes model can be defined as a
  // composition of smaller model parts with their own `apply` function, and its
  // `apply` function can call `apply` functions of those model parts passing
  // them its inputs as arguments. That would lead to multiple mark calls being
  // produced for the same input, and those other calls could end up inlined
  // into main graph in which case the argument will not be a block argument but
  // instead a result of other ops which transform the original input. Since it
  // is enough to propagate the role for input only once (from the root `apply`
  // function) we can ignore the other marks coming from internal `apply` calls.
  auto block_argument = mlir::dyn_cast<mlir::BlockArgument>(argument);
  if (!block_argument) {
    return;
  }

  mlir::Operation *parent_op = block_argument.getOwner()->getParentOp();
  uint32_t arg_index = block_argument.getArgNumber();
  if (auto parent_func_op = mlir::dyn_cast<mlir::func::FuncOp>(parent_op)) {
    parent_func_op.setArgAttr(arg_index, c_input_role_attr_name, role_attr);

    // In case when graph parts are moved to separate private functions and mark
    // calls end up in some of them, we need to propagate the input role
    // attribute upwards through the call chain up to the module root public
    // function arguments.
    llvm::StringRef funcName = parent_func_op.getSymName();
    module.walk([&](mlir::func::CallOp call_op) {
      if (call_op.getCallee() == funcName &&
          arg_index < call_op.getNumOperands()) {
        mlir::Value callerArg = call_op.getOperand(arg_index);
        propagateRoleAttribute(module, callerArg, role_attr);
      }
    });
  } else {
    // Sometimes graph can be transformed after mark calls are inserted, where
    // they end up wrapped in some op body, for example wrapped by
    // `sdy.manual_computation` op. In that case we need to propagate the input
    // role attribute upwards through the call chain up to the module root
    // public function arguments.
    mlir::Value op_operand = parent_op->getOperand(arg_index);
    propagateRoleAttribute(module, op_operand, role_attr);
  }
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
