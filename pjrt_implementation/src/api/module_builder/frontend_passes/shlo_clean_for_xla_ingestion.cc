// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_clean_for_xla_ingestion.h"

// c++ standard library includes
#include <optional>

// llvm includes
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "shardy/dialect/sdy/ir/dialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// stablehlo includes
#include "stablehlo/dialect/StablehloOps.h"

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt::module_builder::frontend_passes {

namespace internal {

// Strips all ttcore and ttir dialect attributes from a function's arguments and
// results. This helper function filters out any attribute whose dialect
// namespace is "ttcore" or "ttir", leaving only attributes that XLA can ingest.
void stripTTDialectAttributes(mlir::func::FuncOp funcOp) {
  auto *ctx = funcOp.getContext();

  auto filterTTDialect =
      [&](mlir::DictionaryAttr dict) -> mlir::DictionaryAttr {
    if (!dict) {
      return nullptr;
    }

    mlir::NamedAttrList newList;
    for (auto attr : dict) {
      // Check if the attribute name belongs to ttcore dialect by checking
      // if it starts with "ttcore." prefix
      llvm::StringRef attrName = attr.getName().getValue();

      // Only keep the attribute if it DOES NOT belong to ttcore or ttir
      // dialects
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

  // Clean function arguments by removing ttcore and ttir attributes
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    funcOp.setArgAttrs(i, filterTTDialect(funcOp.getArgAttrDict(i)));
  }

  // Clean function results by removing ttcore and ttir attributes
  for (unsigned i = 0; i < funcOp.getNumResults(); ++i) {
    funcOp.setResultAttrs(i, filterTTDialect(funcOp.getResultAttrDict(i)));
  }
}

// Given shardy out shardings from a manual computation op, return a list of
// GSPMDV2 out sharding strings in output order
// TODO: When HLOShardingV3 is uplifted, this transformation will need to change
// to support the new sharding format.
// Returns std::nullopt on error (e.g., invalid sharding configuration).
std::optional<llvm::SmallVector<std::string>>
extractSdyManualComputationOutSharding(
    mlir::sdy::ManualComputationOp manualComputationOp,
    llvm::SmallDenseMap<llvm::StringRef, int> &meshNameToSizeMap) {
  auto outShardingsArr = manualComputationOp.getOutShardings().getShardings();

  uint64_t totalNumDevices = std::accumulate(meshNameToSizeMap.values().begin(),
                                             meshNameToSizeMap.values().end(),
                                             1LL, std::multiplies<int64_t>());

  llvm::SmallVector<std::string> outShardingStrs;
  // Iterate over each output tensor sharding and construct the GSPMDV2 out
  // sharding string for each output tensor
  for (auto [outShardingIndex, outSharding] :
       llvm::enumerate(outShardingsArr)) {
    bool isOutputTensorReplicated = true;
    llvm::SmallVector<uint64_t> tensorAxisSizes;
    std::string outShardingStr = "{devices=[";
    auto dimShardings = outSharding.getDimShardings();

    // Iterate over each dimension sharding on each output tensor sharding
    // (replicated or simple axis sharding, without subaxes which are currently
    // not supported).
    for (auto [dimIndex, dimSharding] : llvm::enumerate(dimShardings)) {
      auto axisName = dimSharding.getAxes();
      // fully replicated case
      if (axisName.empty()) {
        tensorAxisSizes.push_back(1);
        continue;
      }

      // one-axis sharding case
      if (axisName.size() != 1) {
        DLOG_F(ERROR, "Expected single axis name, found %zu axes",
               axisName.size());
        return std::nullopt;
      }
      isOutputTensorReplicated = false;
      auto axisNameStr = axisName[0].getName();
      uint64_t axisSize = meshNameToSizeMap.at(axisNameStr);
      tensorAxisSizes.push_back(axisSize);
    }

    // The iota tile format requires replicated dimensions to be added as the
    // last dimension, along with the string " last_tile_dim_replicate" Spec:
    // https://github.com/jax-ml/jax/blob/84af8a8e74c05ce4196079e145d50f0c9504ff16/jax/_src/named_sharding.py#L415-L430
    uint64_t totalTensorShardingAxisSizes =
        std::accumulate(tensorAxisSizes.begin(), tensorAxisSizes.end(), 1LL,
                        std::multiplies<int64_t>());
    bool lastTileDimReplicate = false;
    if (totalTensorShardingAxisSizes != totalNumDevices) {
      lastTileDimReplicate = true;
      if (totalNumDevices % totalTensorShardingAxisSizes != 0) {
        DLOG_F(ERROR,
               "Total tensor sharding axis sizes (%llu) must be a divisor of "
               "total number of devices (%llu)",
               static_cast<unsigned long long>(totalTensorShardingAxisSizes),
               static_cast<unsigned long long>(totalNumDevices));
        return std::nullopt;
      }
      uint64_t replicatedDim = totalNumDevices / totalTensorShardingAxisSizes;
      tensorAxisSizes.push_back(replicatedDim);
    }

    std::string axisSizesString;
    llvm::raw_string_ostream os(axisSizesString);
    llvm::interleave(tensorAxisSizes, os, ",");
    outShardingStr += axisSizesString;
    outShardingStr += "]<=[";
    outShardingStr += std::to_string(totalNumDevices);
    outShardingStr += "]";

    if (lastTileDimReplicate) {
      outShardingStr += " last_tile_dim_replicate";
    }
    outShardingStr += "}";

    if (isOutputTensorReplicated) {
      outShardingStr = "{replicated}";
    }

    outShardingStrs.push_back(outShardingStr);
  }
  return outShardingStrs;
}

// Simplifies the main funcop by removing all body contents and
// returning dummy outputs, to strip the illegal sdy.manual_computation op.
void simplifyMainFuncOp(mlir::func::FuncOp funcOp) {
  if (!funcOp || funcOp.getBody().empty()) {
    return;
  }

  auto &bodyBlock = funcOp.getBody().front();
  auto *ctx = funcOp.getContext();
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&bodyBlock);

  // Get return types
  auto returnTypes = funcOp.getFunctionType().getResults();

  // Clear all existing operations in the body
  bodyBlock.clear();
  builder.setInsertionPointToStart(&bodyBlock);

  // Create zero constants for each return type
  llvm::SmallVector<mlir::Value> returnValues;
  for (auto returnType : returnTypes) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(returnType)) {
      auto elementType = tensorType.getElementType();

      // Create zero attribute - getZeroAttr should handle all types
      auto zeroAttr = builder.getZeroAttr(elementType);
      assert(zeroAttr && "Failed to create zero attribute for element type");

      // Create dense attribute for tensor using splat
      auto denseAttr = mlir::DenseElementsAttr::get(tensorType, zeroAttr);

      // Create constant op
      auto constantOp = builder.create<mlir::stablehlo::ConstantOp>(
          funcOp.getLoc(), denseAttr);
      returnValues.push_back(constantOp.getResult());
    } else {
      // Non-tensor types - skip for now (shouldn't happen in StableHLO)
      // If needed, handle appropriately
      continue;
    }
  }

  // Create return op
  builder.create<mlir::func::ReturnOp>(funcOp.getLoc(), returnValues);
}

} // namespace internal

tt_pjrt_status cleanForXlaIngestion(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module_with_sdy_annotations) {
  mlir::ModuleOp module = mlir_module_with_sdy_annotations.get();

  std::vector<mlir::sdy::ManualComputationOp> manualComputationOps;
  module.walk([&](mlir::sdy::ManualComputationOp op) {
    manualComputationOps.push_back(op);
  });

  module.walk([&](mlir::func::FuncOp funcOp) {
    internal::stripTTDialectAttributes(funcOp);
  });

  // Strip all location information (loc attributes) from the module
  mlir::PassManager pm(mlir_module_with_sdy_annotations.get()->getName());
  pm.addPass(mlir::createStripDebugInfoPass());
  if (mlir::failed(pm.run(mlir_module_with_sdy_annotations.get()))) {
    return tt_pjrt_status::kInternal;
  }

  // Extract mesh name to size map from sdy.mesh op, asserting that there is
  // exactly one sdy.mesh op in the module.
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps;
  module.walk([&](mlir::sdy::MeshOp op) { meshOps.push_back(op); });
  assert(meshOps.size() == 1 && "Expected exactly one sdy.mesh op in module");
  auto meshOp = meshOps.front();

  llvm::SmallDenseMap<llvm::StringRef, int> meshNameToSizeMap;
  for (auto axis : meshOp.getMesh().getAxes()) {
    meshNameToSizeMap[axis.getName()] = axis.getSize();
  }

  // Extract out sharding from manual computation ops
  assert(manualComputationOps.size() == 1 &&
         "Expected exactly one ManualComputationOp in module");
  auto manualComputationOp = manualComputationOps.front();
  auto outShardingResult = internal::extractSdyManualComputationOutSharding(
      manualComputationOp, meshNameToSizeMap);

  if (!outShardingResult.has_value()) {
    DLOG_F(ERROR, "Failed to extract sharding from manual computation op");
    return tt_pjrt_status::kInternal;
  }

  // Inject out sharding result into module as a moduleOp attr,
  // mhlo.spmd_output_shardings format list into tuple type opSharding
  std::string outShardingTupleString = "{";
  llvm::raw_string_ostream os(outShardingTupleString);
  llvm::interleave(*outShardingResult, os, ",");
  outShardingTupleString += "}";

  module->setAttr(
      "mhlo.spmd_output_sharding",
      mlir::StringAttr::get(module.getContext(), outShardingTupleString));

  // Remove sdy.mesh operations
  std::vector<mlir::sdy::MeshOp> meshOpsToErase;
  module.walk([&](mlir::sdy::MeshOp meshOpToErase) {
    meshOpsToErase.push_back(meshOpToErase);
  });
  for (auto meshOpToErase : meshOpsToErase) {
    meshOpToErase.erase();
  }

  // Remove the sdy.manual_computation op by simplifying the main funcop
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (funcOp.getSymName() == "main") {
      internal::simplifyMainFuncOp(funcOp);
    }
  });

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder::frontend_passes
