// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/module_builder/frontend_passes/sdy_round_trip_import/sdy_round_trip_import.h"

// c++ standard library includes
#include <map>
#include <memory>

// llvm includes
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

// mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// stablehlo includes
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"

// shardy includes
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"

// local includes
#include "common/pjrt_implementation/module_builder/frontend_passes/sdy_round_trip_import/constants.h"
#include "common/pjrt_implementation/module_builder/frontend_passes/sdy_round_trip_import/utils.h"

// First define all the pass implementations

namespace tt::pjrt::module_builder::frontend_passes::sdy_round_trip_import {

namespace internal {

void addCommonPreImportPasses(mlir::OpPassManager& pm, bool enableConstantImport) {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true)
      .setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled)
      .enableFolding(false)
      .enableConstantCSE(false);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloAggressiveSimplificationPass({}, config));
}

std::unique_ptr<mlir::Pass> createImportUninlineableFuncCallsPass() {
  class ImportUninlineableFuncCallsPass : public mlir::PassWrapper<ImportUninlineableFuncCallsPass, mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportUninlineableFuncCallsPass)
    void runOnOperation() final { /* placeholder */ }
    llvm::StringRef getArgument() const override { return "xla-sdy-import-uninlineable-func-calls"; }
    llvm::StringRef getDescription() const override { return "Placeholder pass"; }
  };
  return std::make_unique<ImportUninlineableFuncCallsPass>();
}

void addCommonPostImportPasses(mlir::OpPassManager& pm, bool importFuncCalls) {
  pm.addPass(createImportUninlineableFuncCallsPass());
}

std::unique_ptr<mlir::Pass> createSdyRoundTripCloneManualComputationCallsPass() {
  class SdyRoundTripCloneManualComputationCallsPass : public mlir::OperationPass<mlir::ModuleOp> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripCloneManualComputationCallsPass)
    
    explicit SdyRoundTripCloneManualComputationCallsPass() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<SdyRoundTripCloneManualComputationCallsPass>()) {}
    
    llvm::StringRef getName() const override { return "SdyRoundTripCloneManualComputationCallsPass"; }
    llvm::StringRef getArgument() const override { return "sdy-round-trip-clone-manual-computation-calls"; }
    llvm::StringRef getDescription() const override { 
      return "Clone manual computation functions so each call gets its own unique function";
    }
    
    std::unique_ptr<mlir::Pass> clonePass() const override {
      return std::make_unique<SdyRoundTripCloneManualComputationCallsPass>(*this);
    }
    
    void runOnOperation() override {
      mlir::ModuleOp moduleOp = getOperation();
      mlir::SymbolTable symbolTable(moduleOp);
      llvm::DenseSet<llvm::StringRef> seenCalleeNames;
      
      // First pass: collect frontend attributes from custom calls and transfer to CallOps
      moduleOp->walk([&](mlir::func::CallOp callOp) {
        if (!callOp.getCallee().contains(kManualComputationBodyFuncName)) {
          return;
        }
        
        // Look for surrounding custom calls that have frontend attributes
        transferFrontendAttributesToCallOp(callOp);
      });
      
      // Second pass: clone manual computation calls 
      moduleOp->walk([&](mlir::func::CallOp callOp) {
        if (!callOp.getCallee().contains(kManualComputationBodyFuncName)) {
          return;
        }
        
        if (seenCalleeNames.insert(callOp.getCallee()).second) {
          return; // First time seeing this callee, no need to clone
        }
        
        // Clone the function and give it a unique name
        auto funcOp = symbolTable.lookup<mlir::func::FuncOp>(callOp.getCallee());
        if (!funcOp) return;
        
        auto clonedFuncOp = funcOp.clone();
        
        // Clone any tt.mark_* functions that this function references
        cloneReferencedMarkFunctions(clonedFuncOp, symbolTable, seenCalleeNames.size());
        
        callOp.setCallee(symbolTable.insert(clonedFuncOp));
      });
    }
    
  private:
    void transferFrontendAttributesToCallOp(mlir::func::CallOp callOp) {
      // Look for GlobalToLocalShape custom call that feeds into this CallOp
      mlir::DictionaryAttr globalToLocalAttrs = nullptr;
      mlir::DictionaryAttr localToGlobalAttrs = nullptr;
      
      // Find GlobalToLocalShape custom call
      for (auto operand : callOp.getOperands()) {
        if (auto customCallOp = operand.getDefiningOp<mlir::stablehlo::CustomCallOp>()) {
          if (customCallOp.getCallTargetName() == kGlobalToLocalShapeCallTargetName) {
            globalToLocalAttrs = customCallOp->getAttrOfType<mlir::DictionaryAttr>(kFrontendAttributesAttr);
            break;
          }
        }
      }
      
      // Find LocalToGlobalShape custom call
      for (auto user : callOp->getResult(0).getUsers()) {
        if (auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(user)) {
          if (customCallOp.getCallTargetName() == kLocalToGlobalShapeCallTargetName) {
            localToGlobalAttrs = customCallOp->getAttrOfType<mlir::DictionaryAttr>(kFrontendAttributesAttr);
            break;
          }
        }
      }
      
      // Combine attributes and set on CallOp
      if (globalToLocalAttrs && localToGlobalAttrs) {
        llvm::SmallVector<mlir::NamedAttribute> combinedAttrs;
        
        // Add in_shardings and manual_axes from GlobalToLocal
        if (auto inShardings = globalToLocalAttrs.get(kInShardings)) {
          combinedAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(callOp->getContext(), kInShardings), inShardings));
        }
        if (auto manualAxes = globalToLocalAttrs.get(kManualAxes)) {
          combinedAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(callOp->getContext(), kManualAxes), manualAxes));
        }
        
        // Add out_shardings from LocalToGlobal  
        if (auto outShardings = localToGlobalAttrs.get(kOutShardings)) {
          combinedAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(callOp->getContext(), kOutShardings), outShardings));
        }
        
        // Set the combined frontend attributes on the CallOp
        auto frontendAttrsDict = mlir::DictionaryAttr::get(callOp->getContext(), combinedAttrs);
        callOp->setAttr(kFrontendAttributesAttr, frontendAttrsDict);
      }
    }
    
    void cloneReferencedMarkFunctions(mlir::func::FuncOp funcOp, mlir::SymbolTable& symbolTable, size_t cloneIndex) {
      llvm::DenseMap<llvm::StringRef, std::string> oldToNewMarkFunctionNames;
      
      // First pass: find all tt.mark_* function calls and clone those functions
      funcOp.walk([&](mlir::func::CallOp callOp) {
        llvm::StringRef calleeName = callOp.getCallee();
        if (calleeName.contains("tt.mark_")) {
          // Check if we already cloned this mark function
          if (oldToNewMarkFunctionNames.find(calleeName) != oldToNewMarkFunctionNames.end()) {
            return;
          }
          
          // Clone the tt.mark_* function
          if (auto markFuncOp = symbolTable.lookup<mlir::func::FuncOp>(calleeName)) {
            auto clonedMarkFunc = markFuncOp.clone();
            std::string newMarkName = calleeName.str() + "_clone_" + std::to_string(cloneIndex);
            
            clonedMarkFunc.setSymName(newMarkName);
            symbolTable.insert(clonedMarkFunc);
            oldToNewMarkFunctionNames[calleeName] = newMarkName;
          }
        }
      });
      
      // Second pass: update all call references to use the cloned mark functions
      funcOp.walk([&](mlir::func::CallOp callOp) {
        llvm::StringRef calleeName = callOp.getCallee();
        if (auto it = oldToNewMarkFunctionNames.find(calleeName); it != oldToNewMarkFunctionNames.end()) {
          callOp.setCallee(it->second);
        }
      });
    }
  };
  
  return std::make_unique<SdyRoundTripCloneManualComputationCallsPass>();
}

std::unique_ptr<mlir::Pass> createSdyRoundTripDedupMeshesPass() {
  class SdyRoundTripDedupMeshesPass : public mlir::OperationPass<mlir::ModuleOp> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripDedupMeshesPass)
    
    explicit SdyRoundTripDedupMeshesPass() : mlir::OperationPass<mlir::ModuleOp>(mlir::TypeID::get<SdyRoundTripDedupMeshesPass>()) {}
    
    llvm::StringRef getName() const override { return "SdyRoundTripDedupMeshesPass"; }
    llvm::StringRef getArgument() const override { return "sdy-round-trip-dedup-meshes"; }
    llvm::StringRef getDescription() const override { 
      return "Deduplicate meshes with identical device configurations but different names";
    }
    
    std::unique_ptr<mlir::Pass> clonePass() const override {
      return std::make_unique<SdyRoundTripDedupMeshesPass>(*this);
    }
    
    void runOnOperation() override {
      mlir::ModuleOp moduleOp = getOperation();
      
      // Collect all mesh operations
      llvm::SmallVector<mlir::sdy::MeshOp> meshOps;
      for (auto meshOp : moduleOp.getOps<mlir::sdy::MeshOp>()) {
        meshOps.push_back(meshOp);
      }
      
      if (meshOps.size() <= 1) {
        return; // Nothing to deduplicate
      }
      
      // Build mapping from mesh identifier to main mesh and duplicates
      std::map<MeshDeviceIdentifier, MeshGroup> meshGroups;
      
      for (auto meshOp : meshOps) {
        auto meshAttr = meshOp.getMeshAttr();
        MeshDeviceIdentifier identifier = getMeshDeviceIdentifier(meshAttr);
        
        auto& group = meshGroups[identifier];
        if (!group.mainMesh) {
          group.mainMesh = meshOp;
        } else {
          group.duplicateMeshes.push_back(meshOp);
        }
      }
      
      // Process each group and remove duplicates
      for (auto& [identifier, group] : meshGroups) {
        if (group.duplicateMeshes.empty()) {
          continue; // No duplicates to process
        }
        
        // Build axis mapping from duplicates to main mesh
        auto mainMeshAttr = group.mainMesh.getMeshAttr();
        
        for (auto duplicateMesh : group.duplicateMeshes) {
          auto duplicateMeshAttr = duplicateMesh.getMeshAttr();
          
          // Map axes from duplicate to main mesh
          llvm::DenseMap<llvm::StringRef, llvm::StringRef> axisMapping = 
              buildAxisMapping(duplicateMeshAttr, mainMeshAttr);
          
          // Remove the duplicate mesh
          duplicateMesh.erase();
        }
      }
    }
    
  private:
    struct MeshDeviceIdentifier {
      int64_t totalDeviceCount;
      llvm::SmallVector<int64_t> deviceIds;
      
      bool operator==(const MeshDeviceIdentifier& other) const {
        return totalDeviceCount == other.totalDeviceCount && 
               deviceIds == other.deviceIds;
      }
      
      bool operator<(const MeshDeviceIdentifier& other) const {
        if (totalDeviceCount != other.totalDeviceCount) {
          return totalDeviceCount < other.totalDeviceCount;
        }
        return deviceIds < other.deviceIds;
      }
    };
    
    struct MeshGroup {
      mlir::sdy::MeshOp mainMesh;
      llvm::SmallVector<mlir::sdy::MeshOp> duplicateMeshes;
    };
    
    MeshDeviceIdentifier getMeshDeviceIdentifier(mlir::sdy::MeshAttr meshAttr) {
      MeshDeviceIdentifier identifier;
      identifier.totalDeviceCount = meshAttr.getTotalSize();
      
      auto deviceIds = meshAttr.getDeviceIds();
      if (!deviceIds.empty()) {
        identifier.deviceIds.assign(deviceIds.begin(), deviceIds.end());
      } else {
        // Generate implicit device IDs (iota)
        for (int64_t i = 0; i < identifier.totalDeviceCount; ++i) {
          identifier.deviceIds.push_back(i);
        }
      }
      
      return identifier;
    }
    
    llvm::DenseMap<llvm::StringRef, llvm::StringRef> buildAxisMapping(
        mlir::sdy::MeshAttr fromMesh, mlir::sdy::MeshAttr toMesh) {
      llvm::DenseMap<llvm::StringRef, llvm::StringRef> mapping;
      
      auto fromAxes = fromMesh.getAxes();
      auto toAxes = toMesh.getAxes();
      
      // Simple mapping: match by position if same size, otherwise by name
      if (fromAxes.size() == toAxes.size()) {
        for (size_t i = 0; i < fromAxes.size(); ++i) {
          mapping[fromAxes[i].getName()] = toAxes[i].getName();
        }
      } else {
        // Try to match by name first
        for (auto fromAxis : fromAxes) {
          for (auto toAxis : toAxes) {
            if (fromAxis.getName() == toAxis.getName() && 
                fromAxis.getSize() == toAxis.getSize()) {
              mapping[fromAxis.getName()] = toAxis.getName();
              break;
            }
          }
        }
      }
      
      return mapping;
    }
  };
  
  return std::make_unique<SdyRoundTripDedupMeshesPass>();
}

} // namespace internal

// Local implementations of deleted Shardy passes
namespace {

// SdyRoundTripImportShardyAttrsPass implementation

// Builds the shardy attributes coming from Shardy previously. This means
// the module was exported from Shardy and we are now round-tripping back.
// This should happen after the meshes were created from the `ModuleOp` attrs
// (see `SdyRoundTripImportShardyAttrsPass`).
void convertShardyAttrs(mlir::func::FuncOp funcOp, mlir::IRRewriter& rewriter) {
  // Copy over the argument shardings, but not the result shardings yet.
  // We need to wait until after we've converted all the Operations before
  // copying the result shardings.
  for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    funcOp.removeArgAttr(argNum, kXlaShardingAttr);
    // Attempt to extract the TensorShardingAttr from the frontend attributes of
    // the function argument/result.
    if (mlir::DictionaryAttr dictAttr = getFuncArgFrontendAttrs(funcOp, argNum)) {
      if (auto sharding = parseStringAttr<mlir::sdy::TensorShardingAttr>(
              dictAttr, kShardingRoundTripAttr)) {
        funcOp.setArgAttr(argNum, mlir::sdy::kShardingAttr, sharding);
        removeFrontendAttribute(funcOp, kShardingRoundTripAttr, argNum);
      }
    }
  }

  // Due to `SdyRoundTripExportShardingsPass` keeping `mhlo.sharding`s, remove
  // them purely for cleanliness of the module.
  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    funcOp.removeResultAttr(
        resNum, mlir::StringAttr::get(funcOp.getContext(), kXlaShardingAttr));
  }

  // Extract the round-tripped SDY shardy attributes from the operations.
  funcOp.front().walk([&](mlir::Operation* op) {
    op->removeAttr(kXlaShardingAttr);
    mlir::DictionaryAttr dictAttr = getFrontendAttrs(op);
    if (!dictAttr) {
      return;
    }
    // `SendOp` and `RecvOp` can have a sharding when doing TPU callbacks
    // through JAX.
    if (mlir::isa<mlir::stablehlo::SendOp, mlir::stablehlo::RecvOp>(op)) {
      auto sharding = parseStringAttr<mlir::sdy::TensorShardingPerValueAttr>(
          dictAttr, kShardingRoundTripAttr);
      // Expect sharding to exist for SendOp/RecvOp.
      assert(sharding != nullptr);
      op->setAttr(mlir::sdy::kShardingAttr, sharding);
    }
    // NOTE: we are only setting the sharding on known custom-calls. For any
    // other op that has a `kShardingRoundTripAttr` we discard it. XLA sometimes
    // creates new instructions, copying over the operand's frontend attrs,
    // which may mean the shapes are wrong when the new instruction is a reshape
    // for example. This does mean we can't fully round-trip b/w HLO and MLIR
    // after SDY propagation.
    if (auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      llvm::StringRef targetName = customCallOp.getCallTargetName();
      if (targetName == kFuncResultShardingTargetName) {
        // This is a temporary CustomCallOp that holds the sharding from a
        // func result. When importing we want to move that sharding to the
        // func result and delete the CustomCallOp.
        auto shardingPerValueAttr = parseStringAttr<mlir::sdy::TensorShardingPerValueAttr>(
            dictAttr, kShardingRoundTripAttr);
        for (mlir::OpOperand& use :
             llvm::make_early_inc_range(customCallOp->getUses())) {
          // We currently ignore users that are not the func return op.
          // This might happen due to inlined func ops that originally had
          // result shardings.
          // TODO(b/370984308): explore if we need to support this properly.
          if (mlir::isa<mlir::func::ReturnOp>(use.getOwner())) {
            funcOp.setResultAttr(use.getOperandNumber(), mlir::sdy::kShardingAttr,
                                 shardingPerValueAttr.getSharding(0));
            use.set(customCallOp.getOperand(0));
          }
        }
        rewriter.replaceOp(customCallOp, customCallOp.getOperand(0));
        return;
      }
      if (targetName == kShardingCustomCallTargetName ||
          isPythonCallbackCustomCall(customCallOp)) {
        customCallOp->setAttr(mlir::sdy::kShardingAttr,
                              parseStringAttr<mlir::sdy::TensorShardingPerValueAttr>(
                                  dictAttr, kShardingRoundTripAttr));
      }
    }
    removeFrontendAttribute(op, kShardingRoundTripAttr);

    // Import sharding rules.
    if (auto shardingRuleAttr = parseStringAttr<mlir::sdy::OpShardingRuleAttr>(
            dictAttr, kShardingRuleRoundTripAttr)) {
      op->setAttr(mlir::sdy::kShardingRuleAttr, shardingRuleAttr);
      removeFrontendAttribute(op, kShardingRuleRoundTripAttr);
    }
  });
}

class SdyRoundTripImportShardyAttrsPass
    : public mlir::PassWrapper<SdyRoundTripImportShardyAttrsPass,
                         mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripImportShardyAttrsPass)

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();

    // We can use the saved string attributes to restore the original mesh and
    // value shardings with the original mesh axis names and priorities on the
    // sharding. If there is no `kMeshesRoundTripAttr, there were no meshes in
    // the original Shardy model.
    std::optional<mlir::DictionaryAttr> meshesAttr =
        tryGetFrontendAttr<mlir::DictionaryAttr>(moduleOp.getOperation(), kMeshesRoundTripAttr);
    mlir::ArrayRef<mlir::NamedAttribute> sdyMeshes = meshesAttr.has_value()
                                             ? meshesAttr.value().getValue()
                                             : mlir::ArrayRef<mlir::NamedAttribute>();

    mlir::IRRewriter rewriter(moduleOp);
    // Insert the meshes before any functions.
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    mlir::SymbolTable symbolTable(moduleOp);
    for (mlir::NamedAttribute mesh : sdyMeshes) {
      auto meshAttr = mlir::cast<mlir::sdy::MeshAttr>(mesh.getValue());
      symbolTable.insert(
          rewriter.create<mlir::sdy::MeshOp>(moduleOp.getLoc(), mesh.getName(), meshAttr));
    }
    removeFrontendAttribute(moduleOp.getOperation(), kMeshesRoundTripAttr);

    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      convertShardyAttrs(funcOp, rewriter);
    }
  }

  llvm::StringRef getArgument() const override {
    return "sdy-round-trip-import-shardy-attrs";
  }

  llvm::StringRef getDescription() const override {
    return "Converts the shardy attributes from strings in MHLO frontend "
           "attributes to SDY meshes, shardings and sharding rules.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripImportShardyAttrsPass() {
  return std::make_unique<SdyRoundTripImportShardyAttrsPass>();
}

// Real implementations of deleted Shardy passes

std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapImportPass() {
  // Placeholder that does nothing for now
  class SdyRoundTripShardMapImportPass : public mlir::PassWrapper<SdyRoundTripShardMapImportPass, mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripShardMapImportPass)
    void runOnOperation() final { /* placeholder */ }
    llvm::StringRef getArgument() const override { return "sdy-round-trip-shard-map-import"; }
    llvm::StringRef getDescription() const override { return "Placeholder pass"; }
  };
  return std::make_unique<SdyRoundTripShardMapImportPass>();
}

std::unique_ptr<mlir::Pass> createImportSdyCustomCallsPass() {
  class ImportSdyCustomCallsPass : public mlir::PassWrapper<ImportSdyCustomCallsPass, mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportSdyCustomCallsPass)
    void runOnOperation() final { /* placeholder */ }
    llvm::StringRef getArgument() const override { return "sdy-import-sdy-custom-calls"; }
    llvm::StringRef getDescription() const override { return "Placeholder pass"; }
  };
  return std::make_unique<ImportSdyCustomCallsPass>();
}

// Functions moved to internal namespace above

// Now define the main pipeline function
void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport,
                                   bool importFuncCalls,
                                   bool liftAndDedupMeshes) {
  internal::addCommonPreImportPasses(pm, enableConstantImport);
  pm.addPass(createSdyRoundTripImportShardyAttrsPass());
  // TODO(b/430894772): Drop the pass and handle cloning inside shard map import
  // pass.
  pm.addPass(internal::createSdyRoundTripCloneManualComputationCallsPass());
  pm.addPass(createSdyRoundTripShardMapImportPass());
  pm.addPass(createImportSdyCustomCallsPass());
  internal::addCommonPostImportPasses(pm, importFuncCalls);
  if (liftAndDedupMeshes) {
    // Lift and dedup meshes required here because of sdy shardings added
    // directly to hlo in tf2xla.
    pm.addPass(mlir::sdy::createLiftInlinedMeshesPass());
    pm.addPass(internal::createSdyRoundTripDedupMeshesPass());
  }
}

} // namespace tt::pjrt::module_builder::frontend_passes::sdy_round_trip_import

