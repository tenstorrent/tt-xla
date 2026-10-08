// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_flatten_return_tuple.h"

// llvm includes
#include "llvm/ADT/SmallVector.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

// stablehlo includes
#include "stablehlo/dialect/StablehloOps.h"

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt::module_builder::frontend_passes {

namespace internal {

// Recursively expands a possibly-tuple value into its leaf (non-tuple) values,
// inserting stablehlo.get_tuple_element ops via `builder` for every tuple
// level.
static void expandTupleValue(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value value,
                             llvm::SmallVectorImpl<mlir::Value> &leaves) {
  auto tuple_type = mlir::dyn_cast<mlir::TupleType>(value.getType());
  if (!tuple_type) {
    leaves.push_back(value);
    return;
  }

  llvm::ArrayRef<mlir::Type> element_types = tuple_type.getTypes();
  for (uint32_t index = 0; index < element_types.size(); ++index) {
    mlir::Value element = builder.create<mlir::stablehlo::GetTupleElementOp>(
        loc, element_types[index], value, index);
    expandTupleValue(builder, loc, element, leaves);
  }
}

static bool functionReturnsTuple(mlir::func::FuncOp func_op) {
  for (mlir::Type result_type : func_op.getFunctionType().getResults()) {
    if (mlir::isa<mlir::TupleType>(result_type)) {
      return true;
    }
  }
  return false;
}

} // namespace internal

tt_pjrt_status
flattenReturnTuple(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  llvm::SmallVector<mlir::func::FuncOp> funcs_to_flatten;
  mlir_module.get().walk([&](mlir::func::FuncOp func_op) {
    if (func_op.isPublic() && !func_op.isDeclaration() &&
        internal::functionReturnsTuple(func_op)) {
      funcs_to_flatten.push_back(func_op);
    }
  });

  for (mlir::func::FuncOp func_op : funcs_to_flatten) {
    // Entry functions have a single block terminated by func.return.
    auto return_op =
        mlir::dyn_cast<mlir::func::ReturnOp>(func_op.front().getTerminator());
    if (!return_op) {
      DLOG_F(ERROR,
             "Public function %s terminator is not func.return; cannot "
             "flatten tuple result",
             func_op.getName().str().c_str());
      return tt_pjrt_status::kInternal;
    }

    mlir::OpBuilder builder(return_op);
    llvm::SmallVector<mlir::Value> flat_operands;
    for (mlir::Value operand : return_op.getOperands()) {
      internal::expandTupleValue(builder, return_op.getLoc(), operand,
                                 flat_operands);
    }

    return_op->setOperands(flat_operands);

    llvm::SmallVector<mlir::Type> flat_result_types(llvm::map_range(
        flat_operands, [](mlir::Value v) { return v.getType(); }));
    func_op.setType(builder.getFunctionType(
        func_op.getFunctionType().getInputs(), flat_result_types));
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder::frontend_passes
