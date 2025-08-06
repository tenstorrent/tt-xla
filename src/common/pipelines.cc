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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace tt::pjrt::pipelines {

bool isTTMarkFunction(const std::string &function_name) {
  return function_name.substr(0, 8) == "tt.mark_";
}

void runTTXLAPipelines(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  upliftMarkParametersCustomCall(mlir_module);
  // Propagate role attributes through the call graph - jax implementation
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

struct ReplaceMarkParameterWithCall final
    : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>::OpRewritePattern;

  ReplaceMarkParameterWithCall(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (op.getCallTargetName() != "tt.mark_argument") {
      return mlir::failure();
    }

    mlir::Value input = op.getOperand(0);
    auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(input);
    assert(blockArg && "Expected block argument as input to tt.mark_argument");

    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto argIndex = blockArg.getArgNumber();

    auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp);
    assert(funcOp && "Expected function as parent of block argument");

    mlir::DictionaryAttr frontendAttrs;
    if (mlir::Attribute frontendAttrs_ =
            op->getDiscardableAttr("mhlo.frontend_attributes")) {
      frontendAttrs = mlir::cast<mlir::DictionaryAttr>(frontendAttrs_);
    } else {
      return mlir::failure();
    }

    auto argumentType = frontendAttrs.get("argument_type");
    if (!argumentType) {
      return mlir::failure();
    }

    mlir::StringRef argumentTypeStr;
    if (mlir::StringAttr argumentTypeStrAttr =
            mlir::dyn_cast<mlir::StringAttr>(argumentType)) {
      argumentTypeStr = argumentTypeStrAttr.getValue();
    }

    auto nameAttr = frontendAttrs.get("name");
    if (!nameAttr) {
      return mlir::failure();
    }

    mlir::StringAttr nameStrAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    if (!nameStrAttr) {
      return mlir::failure();
    }

    mlir::tt::ttcore::ArgumentType argumentTypeEnum;
    if (argumentTypeStr == "input") {
      argumentTypeEnum = mlir::tt::ttcore::ArgumentType::Input;
    } else if (argumentTypeStr == "parameter") {
      argumentTypeEnum = mlir::tt::ttcore::ArgumentType::Parameter;
    } else if (argumentTypeStr == "constant") {
      argumentTypeEnum = mlir::tt::ttcore::ArgumentType::Constant;
    } else {
      return mlir::failure();
    }

    funcOp.setArgAttr(argIndex, "ttcore.argument_type",
                      mlir::tt::ttcore::ArgumentTypeAttr::get(
                          funcOp.getContext(), argumentTypeEnum));

    funcOp.setArgAttr(argIndex, "ttir.name", nameStrAttr);

    rewriter.replaceOp(op, input);
    return mlir::success();
  }
};

void upliftMarkParametersCustomCall(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {

  mlir::MLIRContext *context = mlir_module->getContext();
  context->loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  mlir::RewritePatternSet patterns(context);
  patterns.add<ReplaceMarkParameterWithCall>(context);

  if (failed(mlir::applyPatternsGreedily(mlir_module.get(),
                                         std::move(patterns)))) {
    LOG_F(ERROR, "Failed to uplift mark parameters custom call");
  }
}

} // namespace tt::pjrt::pipelines
