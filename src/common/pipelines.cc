// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pipelines.h"
#include "common/status.h"

// loguru includes
#include "loguru/loguru.hpp"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace tt::pjrt::pipelines {

bool isTTMarkFunction(const std::string &function_name) {
  return function_name.substr(0, 8) == "tt.mark_";
}

void runTTXLAPipelines(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  // Propagate role attributes through the call graph
  propagateRoleAttributes(mlir_module);

  // Inline all private tt.mark_* functions to eliminate unnecessary calls
  inlineTTMarkFunctions(mlir_module);
}

void propagateRoleAttributes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::ModuleOp module = mlir_module.get();

  module.walk([&](mlir::func::CallOp callOp) {
    auto roleAttr =
        callOp->getAttrOfType<mlir::StringAttr>(kInputRoleAttrString);
    if (roleAttr) {
      for (mlir::Value operand : callOp.getOperands()) {
        propagateRoleAttribute(module, operand, roleAttr);
      }
    }
  });
}

void propagateRoleAttribute(mlir::ModuleOp module, mlir::Value value,
                            mlir::StringAttr roleAttr) {
  if (auto *definingOp = value.getDefiningOp()) {
    definingOp->setAttr(kInputRoleAttrString, roleAttr);

    // If this is a call operation, propagate to its arguments
    if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(definingOp)) {
      for (mlir::Value operand : callOp.getOperands()) {
        propagateRoleAttribute(module, operand, roleAttr);
      }
    }
  } else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // If it's a block argument, the value is a function arguments, so
    // set the attribute on the parent function
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto argIndex = blockArg.getArgNumber();

    if (auto parentFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp)) {
      parentFuncOp.setArgAttr(argIndex, kInputRoleAttrString, roleAttr);

      // Find all call sites of this function and propagate upward
      auto funcName = parentFuncOp.getSymName();
      module.walk([&](mlir::func::CallOp walkCallOp) {
        if (walkCallOp.getCallee() == funcName &&
            argIndex < walkCallOp.getNumOperands()) {
          mlir::Value callerArg = walkCallOp.getOperand(argIndex);
          propagateRoleAttribute(module, callerArg, roleAttr);
        }
      });
    }
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
    if (auto callee_attr = call_op.getCalleeAttr()) {
      std::string callee_name = callee_attr.getValue().str();

      if (isTTMarkFunction(callee_name)) {
        // This is a call to a tt.mark_* function - inline it
        if (call_op.getNumOperands() == 1 && call_op.getNumResults() == 1) {
          mlir::Value operand = call_op.getOperand(0);
          call_op.getResult(0).replaceAllUsesWith(operand);

          // Preserve any tt.input_role attribute from the call
          if (auto role_attr = call_op->getAttrOfType<mlir::StringAttr>(
                  kInputRoleAttrString)) {
            // Set the attribute on the operand's defining operation if it
            // exists
            if (auto *defining_op = operand.getDefiningOp()) {
              defining_op->setAttr(kInputRoleAttrString, role_attr);
            }
          }
          call_op.erase();
        }
      }
    }
  });

  // Remove the now-unused tt.mark_* function definitions
  for (auto func_op : functions_to_remove) {
    std::string func_name = func_op.getSymName().str();
    func_op.erase();
  }
}

} // namespace tt::pjrt::pipelines
