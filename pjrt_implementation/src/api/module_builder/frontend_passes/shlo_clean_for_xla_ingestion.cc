// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/frontend_passes/shlo_clean_for_xla_ingestion.h"

// llvm includes
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "shardy/dialect/sdy/ir/dialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// stablehlo includes
#include "stablehlo/dialect/StablehloOps.h"

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


llvm::SmallVector<std::string> extractSdyManualComputationOutSharding(mlir::sdy::ManualComputationOp manualComputationOp, llvm::SmallDenseMap<llvm::StringRef, int> &meshNameToSizeMap) {
  // returns a list of out shardings
  auto *ctx = manualComputationOp.getOperation()->getContext();
  auto outShardingsArr = manualComputationOp.getOutShardings().getShardings();
  for (auto outSharding : outShardingsArr) {
    // Print each outSharding as a string
    std::string sharding_str;
    llvm::raw_string_ostream os(sharding_str);
    mlir::Attribute attr = outSharding;
    attr.print(os);
    os.flush();
    llvm::errs() << "Out sharding: " << sharding_str << "\n";
  }

  uint64_t totalNumDevices = std::accumulate(meshNameToSizeMap.values().begin(), meshNameToSizeMap.values().end(), 1LL, std::multiplies<int64_t>());

  llvm::SmallVector<std::string> outShardingStrs;
  // iterating over each output tensor sharding
  for (auto [outShardingIndex, outSharding] : llvm::enumerate(outShardingsArr)) {
    bool isOutputTensorReplicated = true;
    llvm::SmallVector<uint64_t> tensorAxisSizes;
    std::string outShardingStr="{devices=[";
    auto dimShardings = outSharding.getDimShardings();

    // on each output tensor sharding, iterating over each dimension sharding (replicated or one-axis sharding)
    for(auto [dimIndex, dimSharding] : llvm::enumerate(dimShardings)) {
      auto axisName = dimSharding.getAxes();
      llvm::errs() << "  Out sharding #" << outShardingIndex << ", dim #" << dimIndex << ", axisName.size()=" << axisName.size() << "\n";
      // replicated case
      if(axisName.empty()){
        tensorAxisSizes.push_back(1);
        continue;
      }

      // one-axis sharding case
      assert(axisName.size() == 1 && "Expected single axis name");
      isOutputTensorReplicated = false;
      auto axisNameStr = axisName[0].getName();
      uint64_t axisSize = meshNameToSizeMap.at(axisNameStr);
      tensorAxisSizes.push_back(axisSize);
    }

    // construct result string
    std::string axisSizesString;
    llvm::raw_string_ostream os(axisSizesString);
    llvm::interleave(tensorAxisSizes, os, ","); 
    outShardingStr += axisSizesString;
    outShardingStr += "]<=[";
    outShardingStr += std::to_string(totalNumDevices);
    outShardingStr += "]}";



    if(isOutputTensorReplicated){
      outShardingStr = "{replicated}";
    }

    outShardingStrs.push_back(outShardingStr);
    llvm::errs()<< "Out sharding str for out sharding #" << outShardingIndex << " is: " << outShardingStr.c_str() << "\n";
  }
  return outShardingStrs;
}

// Drastically simplifies the main funcop by removing all body contents and returning dummy outputs
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

// Strips sdy.manual_computation operations by inlining their body into the parent region
// and replacing the manual_computation results with the return operands.
void stripSdyManualComputation(mlir::sdy::ManualComputationOp manualComputationOp) {
  auto *op = manualComputationOp.getOperation();
  auto &bodyRegion = manualComputationOp.getBody();
  
  if (bodyRegion.empty()) {
    op->erase();
    return;
  }
  
  auto &bodyBlock = bodyRegion.front();
  
  // Find the sdy.return operation
  mlir::sdy::ReturnOp returnOp = nullptr;
  for (auto &bodyOp : bodyBlock) {
    if (auto ret = mlir::dyn_cast<mlir::sdy::ReturnOp>(bodyOp)) {
      returnOp = ret;
      break;
    }
  }
  
  if (!returnOp) {
    // No return found, just erase the op
    op->erase();
    return;
  }
  
  // Replace block arguments with the corresponding operands
  // The manual_computation has inner operands that become block arguments
  auto bodyArgs = bodyBlock.getArguments();
  auto allOperands = op->getOperands();
  // The inner operands start after the outer operands
  // For now, we'll replace block args with the outer operands if they match in count
  // Otherwise, we'll need to handle this more carefully
  if (bodyArgs.size() <= allOperands.size()) {
    // Use the last N operands as inner operands (this is a heuristic)
    size_t innerStartIdx = allOperands.size() - bodyArgs.size();
    for (size_t i = 0; i < bodyArgs.size(); ++i) {
      bodyArgs[i].replaceAllUsesWith(allOperands[innerStartIdx + i]);
    }
  }
  
  // Replace manual_computation results with return operands
  auto returnOperands = returnOp.getOperands();
  auto manualComputationResults = manualComputationOp.getResults();
  
  if (returnOperands.size() != manualComputationResults.size()) {
    // Mismatch in result counts, can't safely replace
    op->erase();
    return;
  }
  
  for (auto [result, operand] : llvm::zip(manualComputationResults, returnOperands)) {
    result.replaceAllUsesWith(operand);
  }
  
  // Move all operations from the body (except the return) before the manual_computation
  auto *insertionPoint = op;
  for (auto &bodyOp : llvm::make_early_inc_range(bodyBlock)) {
    if (mlir::isa<mlir::sdy::ReturnOp>(bodyOp)) {
      continue; // Skip the return op
    }
    bodyOp.moveBefore(insertionPoint);
  }
  
  // Erase the return op and the manual_computation op
  returnOp.erase();
  op->erase();
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
  
  llvm::SmallDenseMap<llvm::StringRef, int> meshNameToSizeMap;
  module.walk([&](mlir::sdy::MeshOp meshOp) {
    for(auto axis: meshOp.getMesh().getAxes()) {
      meshNameToSizeMap[axis.getName()] = axis.getSize();
    }
  });

  // Extract out sharding from manual computation ops
  std::vector<mlir::sdy::ManualComputationOp> manualComputationOps;
  module.walk([&](mlir::sdy::ManualComputationOp op) { manualComputationOps.push_back(op); });
  assert(manualComputationOps.size() == 1 && "Expected exactly one ManualComputationOp in module");
  auto manualComputationOp = manualComputationOps.front();
  auto outShardingResult = internal::extractSdyManualComputationOutSharding(manualComputationOp, meshNameToSizeMap);

  // Inject out sharding result into module as a moduleOp attr, mhlo.spmd_output_shardings
  // format list into tuple type 
  std::string outShardingTupleString = "{";
  llvm::raw_string_ostream os(outShardingTupleString);  
  llvm::interleave(outShardingResult, os, ",");
  outShardingTupleString += "}";

  module->setAttr("mhlo.spmd_output_shardings", mlir::StringAttr::get(module.getContext(), outShardingTupleString));
  
  // Remove sdy.mesh operations
  std::vector<mlir::sdy::MeshOp> meshOpsToErase;
  module.walk([&](mlir::sdy::MeshOp meshOp) {
    meshOpsToErase.push_back(meshOp);
  });
  for (auto meshOp : meshOpsToErase) {
    meshOp.erase();
  }
  
  // Simplify the main function by removing all body contents and returning dummy outputs
  // This is an alternative to stripping the manual_computation operation
  module.walk([&](mlir::func::FuncOp funcOp) {
    if (funcOp.getSymName() == "main") {
      internal::simplifyMainFuncOp(funcOp);
    }
  });

  // Alternative: Strip the manual_computation operation by inlining its body
  // internal::stripSdyManualComputation(manualComputationOp);
  
  llvm::errs() << "Module after injecting out sharding result and simplifying main function:\n";
  module.print(llvm::errs());

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder::frontend_passes