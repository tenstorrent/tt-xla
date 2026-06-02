// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_input_role_propagation.h"

// llvm includes
#include "llvm/ADT/StringRef.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// stablehlo mlir includes
#include "stablehlo/dialect/StablehloOps.h"

// tt-mlir includes
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

// tt-xla includes
#include "utils/assert.h"
#include "utils/logging.h"
#include "utils/status.h"

namespace tt::pjrt::module_builder::frontend_passes {

const std::string c_name_attr_name = "ttir.name";
const std::string c_mark_argument_function_name = "tt.mark_argument";
const std::string c_weight_dtype_override_function_name =
    "tt.weight_dtype_override";
const std::string c_weight_dtype_attr_name = "ttcore.weight_dtype";

namespace {

// Shardy attaches argument shardings via this discardable attribute name.
constexpr llvm::StringLiteral c_sdy_sharding_attr_name = "sdy.sharding";

// Propagate per-argument `sdy.sharding` annotations from a private callee onto
// the public caller arguments that it forwards.
//
// Large tensor-parallel models (e.g. Qwen3-32B at high layer counts) are
// emitted by the frontend with the prefill graph as a thin public `@main`
// wrapper whose arguments are bare, forwarding 1:1 to a private inner function
// (`@SyncTensorsGraph.N`) whose arguments carry the `sdy.sharding`. The later
// Inliner folds the inner function into `@main` but does NOT copy the callee's
// per-argument shardings onto `@main`'s arguments, so ApplyArgumentShardStatus
// subsequently marks every `@main` argument Unsharded -> the compiler emits a
// spurious replicate `distribute_tensor` for already-mesh-distributed inputs
// (runtime "single buffer from host storage" fatal) or trips the TTNNLayout
// assertion on the trace path. Copying the callee's arg sharding up to the
// caller's forwarded arguments (before inlining) makes the entry function carry
// the sharding, matching how smaller graphs are already emitted.
void propagateShardyArgShardings(mlir::ModuleOp module) {
  module.walk([&](mlir::func::CallOp call_op) {
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
            call_op, call_op.getCalleeAttr());
    if (!callee) {
      return;
    }

    for (unsigned operand_index = 0;
         operand_index < call_op.getNumOperands() &&
         operand_index < callee.getNumArguments();
         ++operand_index) {
      mlir::Attribute sharding =
          callee.getArgAttr(operand_index, c_sdy_sharding_attr_name);
      if (!sharding) {
        continue;
      }

      // Only propagate onto a value that is directly a caller block argument
      // (i.e. forwarded unchanged from the caller's signature).
      auto block_arg =
          mlir::dyn_cast<mlir::BlockArgument>(call_op.getOperand(operand_index));
      if (!block_arg) {
        continue;
      }
      auto caller_func = mlir::dyn_cast<mlir::func::FuncOp>(
          block_arg.getOwner()->getParentOp());
      if (!caller_func) {
        continue;
      }

      unsigned caller_arg_index = block_arg.getArgNumber();
      // Do not override an existing sharding on the caller argument.
      if (caller_func.getArgAttr(caller_arg_index, c_sdy_sharding_attr_name)) {
        continue;
      }
      caller_func.setArgAttr(caller_arg_index, c_sdy_sharding_attr_name,
                             sharding);
    }
  });
}

} // namespace

tt_pjrt_status
annotateArgumentAttributes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {

  // Register the ttcore dialect so that ArgumentTypeAttr objects can be
  // created.
  mlir::MLIRContext *context = mlir_module->getContext();
  context->loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  // Make sure the public entry function carries the argument shardings that the
  // frontend may have placed only on a private forwarded callee, before the
  // compiler pipeline inlines that callee away.
  propagateShardyArgShardings(mlir_module.get());
  // If the model being compiled originates from JAX then the argument types
  // will be annotated using function calls to empty functions, whose attributes
  // contain the argument type information. This function will handle that case.
  internal::propagateInputRoleAttributes(mlir_module);
  return internal::annotateArgumentAttributesFromCustomCall(mlir_module);
}

namespace internal {

const std::string c_tt_mark_function_prefix = "tt.mark_";

void propagateInputRoleAttributes(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::ModuleOp module = mlir_module.get();

  // Propagate role attributes through the call graph
  module.walk([&](mlir::func::CallOp call_op) {
    auto role_attr = call_op->getAttrOfType<mlir::StringAttr>(
        mlir::tt::ttcore::ArgumentTypeAttr::name);
    if (role_attr) {
      for (mlir::Value operand : call_op.getOperands()) {
        propagateRoleAttribute(module, operand, role_attr);
      }
    }
  });

  // Inline all private tt.mark_* functions to eliminate unnecessary calls.
  inlineTTMarkFunctions(mlir_module);
}

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

    std::optional<mlir::tt::ttcore::ArgumentType> argumentTypeEnum =
        mlir::tt::ttcore::ArgumentTypeStringToEnum(role_attr);
    if (!argumentTypeEnum) {
      return;
    }

    parent_func_op.setArgAttr(
        arg_index, mlir::tt::ttcore::ArgumentTypeAttr::name,
        mlir::tt::ttcore::ArgumentTypeAttr::get(parent_func_op.getContext(),
                                                *argumentTypeEnum));

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

// Traces a value to its root block arguments by walking the use-def chain.
mlir::SmallVector<mlir::BlockArgument> getBlockArguments(mlir::Value value) {
  mlir::SmallVector<mlir::BlockArgument> blockArgs;
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (blockArg) {
    blockArgs.push_back(blockArg);
  } else {
    auto definingOp = value.getDefiningOp();
    TT_FATAL(definingOp, "This value does not have a defining operation, nor "
                         "is it a block argument.");
    for (mlir::Value operand : definingOp->getOperands()) {
      blockArgs.append(getBlockArguments(operand));
    }
  }

  return blockArgs;
}

// This pattern is used to populate function argument attributes. It looks for
// calls to `tt.mark_argument` and populates the argument attributes using the
// attributes of the `tt.mark_argument` call. It then erases the
// `tt.mark_argument` call and replaces it with the input value.
//
// Note: If a `tt.mark_argument` call is not the first operation executed on a
// function argument, this is an error and the graph which was provided is not
// valid. This pattern will assert in that case.
struct PopulateArgumentAttrsFromTTMark final
    : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>::OpRewritePattern;

  PopulateArgumentAttrsFromTTMark(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (op.getCallTargetName() != c_mark_argument_function_name) {
      return mlir::failure();
    }

    TT_FATAL(op.getNumOperands() == 1, "Expected one operand to {}, got {}",
             c_mark_argument_function_name, op.getNumOperands());
    TT_FATAL(op.getNumResults() == 1, "Expected one result to {}, got {}",
             c_mark_argument_function_name, op.getNumResults());

    // Torch XLA allows us to populate a frontend_attributes dictionary to
    // custom call ops. This dictionary is used to populate the argument type
    // and name of the argument. We need to extract this information and set the
    // argument type and name of the argument in the function argument
    // attributes.
    mlir::DictionaryAttr frontendAttrs;
    if (mlir::Attribute frontendAttrs_ =
            op->getDiscardableAttr("mhlo.frontend_attributes")) {
      frontendAttrs = mlir::cast<mlir::DictionaryAttr>(frontendAttrs_);
    } else {
      return mlir::failure();
    }

    auto argumentType =
        frontendAttrs.get(mlir::tt::ttcore::ArgumentTypeAttr::name);
    if (!argumentType) {
      return mlir::failure();
    }

    mlir::StringRef argumentTypeStr;
    if (mlir::StringAttr argumentTypeStrAttr =
            mlir::dyn_cast<mlir::StringAttr>(argumentType)) {
      argumentTypeStr = argumentTypeStrAttr.getValue();
    }

    auto nameAttr = frontendAttrs.get(c_name_attr_name);
    if (!nameAttr) {
      return mlir::failure();
    }

    mlir::StringAttr nameStrAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    if (!nameStrAttr) {
      return mlir::failure();
    }

    // Determine the argument type enum from the argument type string
    std::optional<mlir::tt::ttcore::ArgumentType> argumentTypeEnum =
        mlir::tt::ttcore::ArgumentTypeStringToEnum(argumentTypeStr);
    if (!argumentTypeEnum) {
      return mlir::failure();
    }

    // Retrieve input and get its block arguments.
    // Occasionally some torch decompositions will place operations on the input
    // of the mark call. In that case the mark call will no longer be the first
    // operation executed on the argument(s). However, that means we may
    // populate all the roots of the input with the same attributes.
    mlir::Value input = op.getOperand(0);
    mlir::SmallVector<mlir::BlockArgument> blockArgs = getBlockArguments(input);

    for (auto blockArg : blockArgs) {
      auto *parentOp = blockArg.getOwner()->getParentOp();
      auto argIndex = blockArg.getArgNumber();

      // Assert that the input is a block argument to a function
      auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp);
      TT_FATAL(funcOp, "Expected function as parent of block argument");

      // Set argument type for this argument
      funcOp.setArgAttr(argIndex, mlir::tt::ttcore::ArgumentTypeAttr::name,
                        mlir::tt::ttcore::ArgumentTypeAttr::get(
                            funcOp.getContext(), *argumentTypeEnum));

      // Set argument name for this argument
      funcOp.setArgAttr(argIndex, c_name_attr_name, nameStrAttr);
    }
    // Remove the custom call op and replace it with the input
    // as the information is now embedded in the function argument attributes
    rewriter.replaceOp(op, input);

    return mlir::success();
  }
};

// This pattern extracts per-tensor weight dtype annotations from
// `tt.weight_dtype_override` custom calls and sets them as function argument
// attributes. The custom call is then replaced with its input value.
struct PopulateWeightDtypeFromCustomCall final
    : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>::OpRewritePattern;

  PopulateWeightDtypeFromCustomCall(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (op.getCallTargetName() != c_weight_dtype_override_function_name) {
      return mlir::failure();
    }

    TT_FATAL(op.getNumOperands() == 1,
             "Expected one operand to tt.weight_dtype_override");
    TT_FATAL(op.getNumResults() == 1,
             "Expected one result from tt.weight_dtype_override");

    // Extract frontend_attributes from the custom call.
    mlir::DictionaryAttr frontendAttrs;
    if (mlir::Attribute frontendAttrs_ =
            op->getDiscardableAttr("mhlo.frontend_attributes")) {
      frontendAttrs = mlir::cast<mlir::DictionaryAttr>(frontendAttrs_);
    } else {
      return mlir::failure();
    }

    // Extract the weight dtype string attribute.
    auto weightDtypeAttr = frontendAttrs.get(c_weight_dtype_attr_name);
    if (!weightDtypeAttr) {
      return mlir::failure();
    }

    mlir::StringAttr weightDtypeStrAttr =
        mlir::dyn_cast<mlir::StringAttr>(weightDtypeAttr);
    if (!weightDtypeStrAttr) {
      return mlir::failure();
    }

    // Trace the input to its root block arguments and set the weight dtype
    // attribute on each.
    mlir::Value input = op.getOperand(0);
    mlir::SmallVector<mlir::BlockArgument> blockArgs = getBlockArguments(input);

    for (auto blockArg : blockArgs) {
      auto *parentOp = blockArg.getOwner()->getParentOp();
      auto argIndex = blockArg.getArgNumber();

      auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(parentOp);
      TT_FATAL(funcOp, "Expected function as parent of block argument");

      funcOp.setArgAttr(argIndex, c_weight_dtype_attr_name, weightDtypeStrAttr);
    }

    // Remove the custom call op and replace it with the input value.
    rewriter.replaceOp(op, input);

    return mlir::success();
  }
};

tt_pjrt_status annotateArgumentAttributesFromCustomCall(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::MLIRContext *context = mlir_module->getContext();
  mlir::RewritePatternSet patterns(context);
  patterns.add<internal::PopulateArgumentAttrsFromTTMark>(context);
  patterns.add<internal::PopulateWeightDtypeFromCustomCall>(context);

  if (failed(mlir::applyPatternsGreedily(mlir_module.get(),
                                         std::move(patterns)))) {
    LOG_F(ERROR, "Failed to uplift mark parameters custom call");
    return tt_pjrt_status::kInternal;
  }

  setDefaultRoleForUnannotatedArguments(mlir_module);

  return tt_pjrt_status::kSuccess;
}

void setDefaultRoleForUnannotatedArguments(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  // In the event that some of the arguments have not been annotated,
  // we annotate them to default, which is Input.
  mlir_module->walk([&](mlir::func::FuncOp funcOp) {
    for (int64_t i = 0; i < funcOp.getNumArguments(); i++) {
      if (funcOp.getArgAttr(i, mlir::tt::ttcore::ArgumentTypeAttr::name)) {
        continue;
      }

      funcOp.setArgAttr(
          i, mlir::tt::ttcore::ArgumentTypeAttr::name,
          mlir::tt::ttcore::ArgumentTypeAttr::get(
              funcOp.getContext(), mlir::tt::ttcore::ArgumentType::Input));
    }
  });
}

} // namespace internal

} // namespace tt::pjrt::module_builder::frontend_passes
