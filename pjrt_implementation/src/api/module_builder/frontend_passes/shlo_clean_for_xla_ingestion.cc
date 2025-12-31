// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_clean_for_xla_ingestion.h"

// llvm includes
#include "llvm/ADT/StringRef.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace tt::pjrt::module_builder::frontend_passes {

namespace internal {

// Strips all ttcore dialect attributes from a function's arguments and results.
// This helper function filters out any attribute whose dialect namespace is
// "ttcore", leaving only attributes that XLA can understand.
void stripTTCoreDialectAttributes(mlir::func::FuncOp funcOp) {
  auto *ctx = funcOp.getContext();

  // Helper to filter a dictionary of attributes, removing any ttcore dialect
  // attributes.
  auto filterTTCore = [&](mlir::DictionaryAttr dict) -> mlir::DictionaryAttr {
    if (!dict) {
      return nullptr;
    }

    mlir::NamedAttrList newList;
    for (auto attr : dict) {
      // Check if the attribute name belongs to ttcore dialect by checking
      // if it starts with "ttcore." prefix
      llvm::StringRef attrName = attr.getName().getValue();
      
      // Only keep the attribute if it DOES NOT belong to ttcore or ttir dialects
      if (!(attrName.starts_with("ttcore.") || attrName.starts_with("ttir."))) {
        newList.push_back(attr);
      }
    }

    // If the list is now empty, return null (strips the {} block)
    if (newList.empty()) {
      return nullptr;
    }
    return newList.getDictionary(ctx);
  };

  // Clean function arguments by removing ttcore attributes
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    funcOp.setArgAttrs(i, filterTTCore(funcOp.getArgAttrDict(i)));
  }

  // Clean function results by removing ttcore attributes
  for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
    funcOp.setResultAttrs(i, filterTTCore(funcOp.getResultAttrDict(i)));
  }
}

} // namespace internal

tt_pjrt_status cleanForXlaIngestion(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::ModuleOp module = mlir_module.get();
  module.walk([&](mlir::func::FuncOp funcOp) {
    internal::stripTTCoreDialectAttributes(funcOp);
  });

  // Strip all location information (loc attributes) from the module
  mlir::PassManager pm(mlir_module.get()->getName());
  pm.addPass(mlir::createStripDebugInfoPass());
  if (mlir::failed(pm.run(mlir_module.get()))) {
    // DLOG_F(ERROR, "Failed to strip debug info from module");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder::frontend_passes