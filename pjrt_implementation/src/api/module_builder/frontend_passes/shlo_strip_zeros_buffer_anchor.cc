// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_strip_zeros_buffer_anchor.h"

// c++ standard library includes
#include <string>

// llvm mlir includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// stablehlo includes
#include "stablehlo/dialect/StablehloOps.h"

// tt-xla includes
#include "utils/assert.h"
#include "utils/logging.h"

namespace tt::pjrt::module_builder::frontend_passes {

namespace internal {

const std::string c_zeros_buffer_function_name = "tt.zeros_buffer";

// Drops the anchor operand from `stablehlo.custom_call @tt.zeros_buffer`.
//
// The anchor exists purely because torch-xla refuses to build a custom call
// with an empty operand list (it derives the result tensors' device from the
// first input). tt-mlir wants the operand-free form, so the operand is removed
// here, before the module reaches the compiler. Once dropped, the constant
// feeding it has no uses left and the greedy driver erases it.
struct StripZerosBufferAnchor final
    : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  using mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>::OpRewritePattern;

  StripZerosBufferAnchor(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::stablehlo::CustomCallOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (op.getCallTargetName() != c_zeros_buffer_function_name) {
      return mlir::failure();
    }

    // Already in the form tt-mlir expects. Reporting a match here would make
    // the greedy driver reapply the pattern forever.
    if (op.getNumOperands() == 0) {
      return mlir::failure();
    }

    TT_FATAL(op.getNumResults() == 1, "Expected one result from {}, got {}",
             c_zeros_buffer_function_name, op.getNumResults());
    TT_FATAL(op.getOutputOperandAliases().empty(),
             "{} must not alias its operands, got {} aliases",
             c_zeros_buffer_function_name, op.getOutputOperandAliases().size());

    rewriter.modifyOpInPlace(op, [&]() {
      op.getInputsMutable().clear();
      // Layout attributes are verified against operand arity, and StableHLO
      // requires operand_layouts and result_layouts either both present or both
      // absent -- so removing one means removing the other.
      op.removeOperandLayoutsAttr();
      op.removeResultLayoutsAttr();
    });

    return mlir::success();
  }
};

} // namespace internal

tt_pjrt_status
stripZerosBufferAnchorOperands(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::MLIRContext *context = mlir_module->getContext();
  mlir::RewritePatternSet patterns(context);
  patterns.add<internal::StripZerosBufferAnchor>(context);

  if (failed(mlir::applyPatternsGreedily(mlir_module.get(),
                                         std::move(patterns)))) {
    LOG_F(ERROR, "Failed to strip anchor operands from tt.zeros_buffer calls");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder::frontend_passes
