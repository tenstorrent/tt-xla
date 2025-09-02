// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/module_builder/module_builder.h"

// c++ standard library includes
#include <cassert>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>

// loguru includes
#include "loguru/loguru.hpp"

// llvm includes
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

// llvm mlir includes
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// stablehlo includes
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

// shardy includes  
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/pipelines.h"
#include "shardy/round_trip_import/utils.h"
#include "shardy/round_trip_import/import_sdy_custom_calls.h"
#include "shardy/round_trip_import/import_shardy_attrs.h"
#include "shardy/round_trip_import/import_uninlineable_func_calls.h"
#include "shardy/round_trip_import/shard_map_import.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

// tt-xla includes
#include "common/pjrt_implementation/data_type_utils.h"
#include "common/pjrt_implementation/module_builder/frontend_passes/shlo_input_role_propagation.h"

namespace tt::pjrt::module_builder {

const std::string c_mlir_format_name = "mlir";

// Helper functions for XLA-style round-trip import pipeline
namespace {

void addCommonPreImportPasses(mlir::OpPassManager& pm, bool enableConstantImport) {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true)
      .setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled)
      .enableFolding(false)
      .enableConstantCSE(false);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloAggressiveSimplificationPass({}, config));
}

void addCommonPostImportPasses(mlir::OpPassManager& pm, bool importFuncCalls) {
  pm.addPass(mlir::sdy::createImportUninlineableFuncCallsPass());
}

// Implementation of missing XLA passes
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
        if (!callOp.getCallee().contains(mlir::sdy::kManualComputationBodyFuncName)) {
          return;
        }
        
        // Look for surrounding custom calls that have frontend attributes
        transferFrontendAttributesToCallOp(callOp);
      });
      
      // Second pass: clone manual computation calls 
      moduleOp->walk([&](mlir::func::CallOp callOp) {
        if (!callOp.getCallee().contains(mlir::sdy::kManualComputationBodyFuncName)) {
          return;
        }
        
        if (seenCalleeNames.insert(callOp.getCallee()).second) {
          return; // First time seeing this callee, no need to clone
        }
        
        // Clone the function and give it a unique name
        auto funcOp = symbolTable.lookup<mlir::func::FuncOp>(callOp.getCallee());
        if (!funcOp) return;
        
        auto clonedFuncOp = funcOp.clone();
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
          if (customCallOp.getCallTargetName() == mlir::sdy::kGlobalToLocalShapeCallTargetName) {
            globalToLocalAttrs = customCallOp->getAttrOfType<mlir::DictionaryAttr>(mlir::sdy::kFrontendAttributesAttr);
            break;
          }
        }
      }
      
      // Find LocalToGlobalShape custom call
      for (auto user : callOp->getResult(0).getUsers()) {
        if (auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(user)) {
          if (customCallOp.getCallTargetName() == mlir::sdy::kLocalToGlobalShapeCallTargetName) {
            localToGlobalAttrs = customCallOp->getAttrOfType<mlir::DictionaryAttr>(mlir::sdy::kFrontendAttributesAttr);
            break;
          }
        }
      }
      
      // Combine attributes and set on CallOp
      if (globalToLocalAttrs && localToGlobalAttrs) {
        llvm::SmallVector<mlir::NamedAttribute> combinedAttrs;
        
        // Add in_shardings and manual_axes from GlobalToLocal
        if (auto inShardings = globalToLocalAttrs.get(mlir::sdy::kInShardings)) {
          combinedAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(callOp->getContext(), mlir::sdy::kInShardings), inShardings));
        }
        if (auto manualAxes = globalToLocalAttrs.get(mlir::sdy::kManualAxes)) {
          combinedAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(callOp->getContext(), mlir::sdy::kManualAxes), manualAxes));
        }
        
        // Add out_shardings from LocalToGlobal  
        if (auto outShardings = localToGlobalAttrs.get(mlir::sdy::kOutShardings)) {
          combinedAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(callOp->getContext(), mlir::sdy::kOutShardings), outShardings));
        }
        
        // Set the combined frontend attributes on the CallOp
        auto frontendAttrsDict = mlir::DictionaryAttr::get(callOp->getContext(), combinedAttrs);
        callOp->setAttr(mlir::sdy::kFrontendAttributesAttr, frontendAttrsDict);
      }
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
          
          // Replace all uses of duplicate mesh with main mesh
          replaceMeshReferences(moduleOp, duplicateMesh.getSymName(), 
                               group.mainMesh.getSymName(), axisMapping);
          
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
    
    void replaceMeshReferences(mlir::ModuleOp moduleOp,
                              llvm::StringRef oldMeshName,
                              llvm::StringRef newMeshName,
                              const llvm::DenseMap<llvm::StringRef, llvm::StringRef>& axisMapping) {
      
      // Walk through all operations and replace mesh references
      moduleOp->walk([&](mlir::Operation* op) {
        // Handle sharding attributes
        if (auto shardingAttr = op->getAttrOfType<mlir::sdy::TensorShardingAttr>("sdy.sharding")) {
          auto newSharding = updateShardingMeshReference(shardingAttr, oldMeshName, newMeshName, axisMapping);
          if (newSharding != shardingAttr) {
            op->setAttr("sdy.sharding", newSharding);
          }
        }
        
        // Handle manual computation operations  
        if (auto manualCompOp = mlir::dyn_cast<mlir::sdy::ManualComputationOp>(op)) {
          updateManualComputationMeshReference(manualCompOp, oldMeshName, newMeshName, axisMapping);
        }
      });
    }
    
    mlir::sdy::TensorShardingAttr updateShardingMeshReference(
        mlir::sdy::TensorShardingAttr shardingAttr,
        llvm::StringRef oldMeshName, 
        llvm::StringRef newMeshName,
        const llvm::DenseMap<llvm::StringRef, llvm::StringRef>& axisMapping) {
      
      // This is a simplified implementation
      // The full implementation would need to update mesh references and axis names
      // within the sharding attribute structure
      return shardingAttr;
    }
    
    void updateManualComputationMeshReference(
        mlir::sdy::ManualComputationOp manualCompOp,
        llvm::StringRef oldMeshName,
        llvm::StringRef newMeshName, 
        const llvm::DenseMap<llvm::StringRef, llvm::StringRef>& axisMapping) {
      
      // This is a simplified implementation
      // The full implementation would update manual axes references
      // to use the new mesh and mapped axis names
    }
  };
  
  return std::make_unique<SdyRoundTripDedupMeshesPass>();
}

} // namespace

void addSdyRoundTripImportPipeline(mlir::OpPassManager& pm,
                                   bool enableConstantImport,
                                   bool importFuncCalls,
                                   bool liftAndDedupMeshes) {
  addCommonPreImportPasses(pm, enableConstantImport);
  pm.addPass(mlir::sdy::createSdyRoundTripImportShardyAttrsPass());
  // TODO(b/430894772): Drop the pass and handle cloning inside shard map import
  // pass.
  pm.addPass(createSdyRoundTripCloneManualComputationCallsPass());
  pm.addPass(mlir::sdy::createSdyRoundTripShardMapImportPass());
  pm.addPass(mlir::sdy::createImportSdyCustomCallsPass());
  addCommonPostImportPasses(pm, importFuncCalls);
  if (liftAndDedupMeshes) {
    // Lift and dedup meshes required here because of sdy shardings added
    // directly to hlo in tf2xla.
    pm.addPass(mlir::sdy::createLiftInlinedMeshesPass());
    pm.addPass(createSdyRoundTripDedupMeshesPass());
  }
}

ModuleBuilder::ModuleBuilder()
    : m_context(std::make_unique<mlir::MLIRContext>()),
      m_flatbuffer_binary(nullptr), m_status(tt_pjrt_status::kSuccess) {
  // Register all the required dialects and passes.
  mlir::DialectRegistry registry;

  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();

  mlir::tt::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::sdy::registerAllDialects(registry);

  mlir::func::registerAllExtensions(registry);
  mlir::tt::registerAllExtensions(registry);

  mlir::tt::ttir::registerPasses();
  mlir::tt::ttnn::registerPasses();

  // We need to allow unregistered dialects since shardy uses specific mhlo
  // dialect attributes, which are not registered in the context and live in the
  // openXLA repository. See issue:
  // https://github.com/tenstorrent/tt-xla/issues/355
  m_context->allowUnregisteredDialects();
  m_context->appendDialectRegistry(registry);
}

tt_pjrt_status ModuleBuilder::buildModule(
    const std::string_view &mlir_code,
    const std::string &system_descriptor_path,
    const std::unordered_map<std::string, std::string> &compile_options_map) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::buildModule");

  m_status = tt_pjrt_status::kSuccess;

  auto compile_options = CompileOptions::parse(compile_options_map);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module = createVHLOModule(mlir_code);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  convertFromVHLOToSHLO(mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  runFrontendSHLOPipeline(mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  collectInputShardings(mlir_module);
  collectOutputShardings(mlir_module);
  collectInputArgumentRoles(mlir_module);
  collectOutputTypes(mlir_module);

  runCompilerStableHLOPipeline(mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  convertFromSHLOToTTIR(mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  collectMeshShape(mlir_module);
  collectNumDevicesToUtilize(mlir_module);

  convertFromTTIRToTTNN(system_descriptor_path, mlir_module, compile_options);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  createFlatbufferBinary(mlir_module);

  return m_status;
}

mlir::OwningOpRef<mlir::ModuleOp>
ModuleBuilder::createVHLOModule(const std::string_view &mlir_code) {
  mlir::OwningOpRef<mlir::ModuleOp> vhlo_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_code.data(), mlir_code.size()),
          mlir::ParserConfig{m_context.get(), /*verifyAfterParse=*/true});

  if (!vhlo_module) {
    DLOG_F(ERROR, "Failed to create VHLO module from the input program code");
    m_status = tt_pjrt_status::kInternal;
    return nullptr;
  }

  DLOG_F(LOG_DEBUG, "VHLO Module:");
  printModule(vhlo_module);

  return vhlo_module;
}

void ModuleBuilder::convertFromVHLOToSHLO(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::PassManager vhlo_to_shlo_pm(mlir_module.get()->getName());

  mlir::stablehlo::createStablehloDeserializePipeline(vhlo_to_shlo_pm);

  if (mlir::failed(vhlo_to_shlo_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from VHLO to SHLO module");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  // Run Shardy round-trip import pipeline if using Shardy
  if (isUsingShardy(mlir_module)) {
    mlir::PassManager shardy_pm(mlir_module.get()->getName());
    addSdyRoundTripImportPipeline(shardy_pm, 
                                  /*enableConstantImport=*/true,
                                  /*importFuncCalls=*/false, 
                                  /*liftAndDedupMeshes=*/false);
    if (mlir::failed(shardy_pm.run(mlir_module.get()))) {
      DLOG_F(ERROR,
             "Failed to convert from Shardy roundtrip import pass module");
      m_status = tt_pjrt_status::kInternal;
      return;
    }
  }

  DLOG_F(LOG_DEBUG, "SHLO Module:");
  printModule(mlir_module);
}

void ModuleBuilder::runFrontendSHLOPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {

  m_status = frontend_passes::annotateArgumentAttributes(mlir_module);

  DLOG_F(LOG_DEBUG, "SHLO Module after frontend StableHLO pipeline:");
  printModule(mlir_module);
}

void ModuleBuilder::collectInputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  m_input_shardings.clear();
  isUsingShardy(module) ? collectInputShardingsShardy(module)
                        : collectInputShardingsGSPMD(module);
}

void ModuleBuilder::collectInputShardingsGSPMD(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumArguments(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getArgAttr(i, mlir::sdy::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, m_input_shardings);
  if (result.failed()) {
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::collectInputShardingsShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  std::optional<mlir::sdy::MeshOp> mesh_op = getFirstShardyMeshOp(module);
  // Since this function is called only when we are using the shardy dialect,
  // this op should always be present.
  if (!mesh_op.has_value()) {
    DLOG_F(ERROR, "Failed to find mesh op in the module");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  mlir::sdy::MeshAttr shardy_mesh = mesh_op->getMesh();
  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::sdy::TensorShardingAttr> shardy_attributes;

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int arg_index = 0; arg_index < func_op.getNumArguments();
         ++arg_index) {
      shardy_attributes.push_back(
          func_op.getArgAttrOfType<mlir::sdy::TensorShardingAttr>(
              arg_index, mlir::sdy::kShardingAttr));
    }
  }

  mlir::LogicalResult result = createShardingsFromShardy(
      shardy_attributes, shardy_mesh, m_input_shardings);
  if (result.failed()) {
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::collectOutputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  m_output_shardings.clear();
  isUsingShardy(module) ? collectOutputShardingsShardy(module)
                        : collectOutputShardingsGSPMD(module);
}

void ModuleBuilder::collectOutputShardingsGSPMD(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumResults(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getResultAttr(i, mlir::sdy::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, m_output_shardings);
  if (result.failed()) {
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::collectOutputShardingsShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  std::optional<mlir::sdy::MeshOp> mesh_op = getFirstShardyMeshOp(module);
  // Since this function is called only when we are using the shardy dialect,
  // this op should always be present.
  if (!mesh_op.has_value()) {
    DLOG_F(ERROR, "Failed to find mesh op in the module");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  mlir::sdy::MeshAttr shardy_mesh = mesh_op->getMesh();
  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::sdy::TensorShardingAttr> shardy_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int result_index = 0; result_index < func_op.getNumResults();
         ++result_index) {
      shardy_attributes.push_back(
          func_op.getResultAttrOfType<mlir::sdy::TensorShardingAttr>(
              result_index, mlir::sdy::kShardingAttr));
    }
  }
  mlir::LogicalResult result = createShardingsFromShardy(
      shardy_attributes, shardy_mesh, m_output_shardings);
  if (result.failed()) {
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::collectInputArgumentRoles(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  m_input_argument_roles.clear();

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int arg_index = 0; arg_index < func_op.getNumArguments();
         ++arg_index) {
      // Check for ttcore.argument_type attribute
      mlir::StringAttr role_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
          arg_index, mlir::tt::ttcore::ArgumentTypeAttr::name);

      if (role_attr && role_attr.getValue() == "weight") {
        m_input_argument_roles.push_back(InputArgumentRole::kWeight);
      } else {
        // Default to input if attribute is not present or not "weight"
        m_input_argument_roles.push_back(InputArgumentRole::kInput);
      }

      // Remove the ttcore.argument_type attribute after collecting it
      if (role_attr) {
        func_op.removeArgAttr(arg_index,
                              mlir::tt::ttcore::ArgumentTypeAttr::name);
      }
    }
  }
}

void ModuleBuilder::collectOutputTypes(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  m_is_output_scalar.clear();
  m_output_data_types.clear();

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (const mlir::Type &returnType :
         func_op.getFunctionType().getResults()) {
      m_is_output_scalar.push_back(isScalarType(returnType));
      m_output_data_types.push_back(
          tt::pjrt::data_type_utils::convertMLIRToPJRTDataType(returnType));
    }
  }
}

std::vector<mlir::func::FuncOp> ModuleBuilder::getPublicFuncOps(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  std::vector<mlir::func::FuncOp> public_func_ops;
  module.get().walk([&](mlir::Operation *op) {
    mlir::func::FuncOp funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op);
    if (funcOp && funcOp.isPublic()) {
      public_func_ops.push_back(funcOp);
    }
  });
  return public_func_ops;
}

bool ModuleBuilder::isScalarType(mlir::Type type) {
  if (mlir::isa<mlir::FloatType>(type) || mlir::isa<mlir::IntegerType>(type)) {
    return true;
  }
  if (mlir::RankedTensorType tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return tensorType.getRank() == 0;
  }
  return false;
}

mlir::LogicalResult ModuleBuilder::createShardingsFromGSPMD(
    const std::vector<mlir::StringAttr> &gspmd_attributes,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings) {

  for (const mlir::StringAttr &gspmd_attr : gspmd_attributes) {

    // If there is no sharding attribute, we put the default sharding,
    // which means there is no sharding.
    if (!gspmd_attr) {
      llvm::Expected<mlir::tt::gspmd_utils::GSPMDMeshSharding>
          default_mesh_sharding_result =
              mlir::tt::gspmd_utils::GSPMDMeshSharding::generateDefault();
      if (default_mesh_sharding_result.takeError()) {
        DLOG_F(ERROR, "Failed to generate default mesh sharding");
        return llvm::LogicalResult::failure();
      }
      shardings.push_back(*default_mesh_sharding_result);
      continue;
    }
    llvm::Expected<mlir::tt::gspmd_utils::GSPMDMeshSharding>
        mesh_sharding_result =
            mlir::tt::gspmd_utils::GSPMDMeshSharding::generate(
                gspmd_attr.getValue(),
                /*operandShardingStr=*/gspmd_attr.getValue(),
                mlir::tt::ttcore::ShardStatus::Unsharded,
                mlir::tt::ttcore::MeshShardDirection::FullToShard);
    if (mesh_sharding_result.takeError()) {
      DLOG_F(ERROR, "Failed to convert sharding attribute to mesh sharding");
      return llvm::LogicalResult::failure();
    }

    shardings.push_back(*mesh_sharding_result);
  }

  return llvm::LogicalResult::success();
}

mlir::LogicalResult ModuleBuilder::createShardingsFromShardy(
    std::vector<mlir::sdy::TensorShardingAttr> &shardy_attributes,
    const mlir::sdy::MeshAttr &shardy_mesh,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings) {
  for (const mlir::sdy::TensorShardingAttr &shardy_attr : shardy_attributes) {

    // If there is no sharding attribute, we put the default sharding,
    // which means there is no sharding.
    if (!shardy_attr) {
      llvm::Expected<mlir::tt::shardy_utils::ShardyMeshSharding>
          default_mesh_sharding_result =
              mlir::tt::shardy_utils::ShardyMeshSharding::generateDefault();
      if (llvm::Error e = default_mesh_sharding_result.takeError()) {
        DLOG_F(ERROR, "Failed to generate default mesh sharding");
        return llvm::LogicalResult::failure();
      }
      shardings.push_back(*default_mesh_sharding_result);
      continue;
    }

    llvm::Expected<mlir::tt::shardy_utils::ShardyMeshSharding>
        mesh_sharding_result =
            mlir::tt::shardy_utils::ShardyMeshSharding::generate(
                shardy_mesh, shardy_attr,
                mlir::tt::ttcore::ShardStatus::Unsharded,
                mlir::tt::ttcore::MeshShardDirection::FullToShard);
    if (llvm::Error e = mesh_sharding_result.takeError()) {
      DLOG_F(ERROR, "Failed to convert sharding attribute to mesh sharding");
      return llvm::LogicalResult::failure();
    }

    shardings.push_back(*mesh_sharding_result);
  }

  return llvm::LogicalResult::success();
}

void ModuleBuilder::runCompilerStableHLOPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::PassManager stablehlo_pipeline_pm(mlir_module.get()->getName(),
                                          mlir::PassManager::Nesting::Implicit);
  mlir::tt::stablehlo::StableHLOPipelineOptions stablehlo_pipeline_options;
  mlir::tt::stablehlo::createStableHLOPipeline(stablehlo_pipeline_pm,
                                               stablehlo_pipeline_options);
  if (mlir::failed(stablehlo_pipeline_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to run stablehlo pipeline");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  DLOG_F(LOG_DEBUG, "SHLO Module after compiler StableHLO pipeline:");
  printModule(mlir_module);
}

void ModuleBuilder::convertFromSHLOToTTIR(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  // Implicit nesting required to call the stablehlo.composite --> func.call
  // conversion.
  mlir::PassManager shlo_to_ttir_pm(mlir_module.get()->getName(),
                                    mlir::PassManager::Nesting::Implicit);

  mlir::tt::ttir::StableHLOToTTIRPipelineOptions shlo_options;
  shlo_options.arithDialectConversionsEnabled = true;
  shlo_options.legalizeCompositeToCallEnabled = true;
  mlir::tt::ttir::createStableHLOToTTIRPipeline(shlo_to_ttir_pm, shlo_options);

  if (mlir::failed(shlo_to_ttir_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from SHLO to TTIR module");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  DLOG_F(LOG_DEBUG, "TTIR Module:");
  printModule(mlir_module);
}

void ModuleBuilder::collectMeshShape(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::tt::ttcore::MeshesAttr meshes_attr =
      module.get()->getAttrOfType<mlir::tt::ttcore::MeshesAttr>(
          mlir::tt::ttcore::MeshesAttr::name);
  if (!meshes_attr || meshes_attr.getMeshes().empty()) {
    // If mesh attribute is not set we can still estimate the mesh based on the
    // input shardings.
    estimateMeshShape();
    return;
  }

  llvm::ArrayRef<mlir::tt::ttcore::MeshAttr> meshes = meshes_attr.getMeshes();

  // For now, use the first mesh shape (same as what is used in tt-mlir).
  llvm::ArrayRef<int64_t> mesh_shape = meshes[0].getShape();

  m_devices_mesh_shape =
      std::vector<std::uint32_t>(mesh_shape.begin(), mesh_shape.end());
}

void ModuleBuilder::estimateMeshShape() {
  for (const mlir::tt::sharding_utils::MeshSharding &input_sharding :
       m_input_shardings) {
    if (input_sharding.getShardType() ==
        mlir::tt::ttcore::MeshShardType::Devices) {
      m_devices_mesh_shape =
          std::vector<std::uint32_t>(input_sharding.getMeshShape().begin(),
                                     input_sharding.getMeshShape().end());
      return;
    }
  }

  // Assuming single device if there are no inputs sharded on device.
  m_devices_mesh_shape = {1, 1};
}

void ModuleBuilder::collectNumDevicesToUtilize(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  auto num_partitions_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_partitions");
  // Assuming one partition by default.
  m_num_partitions = 1;
  if (num_partitions_attr) {
    m_num_partitions = static_cast<size_t>(num_partitions_attr.getInt());
  } else {
    DLOG_F(WARNING,
           "`mhlo.num_partitions` attribute not found, assuming default number "
           "of partitions: %zu",
           m_num_partitions);
  }

  auto num_replicas_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_replicas");
  // Assuming one replica by default.
  m_num_replicas = 1;
  if (num_replicas_attr) {
    m_num_replicas = static_cast<size_t>(num_replicas_attr.getInt());
  } else {
    DLOG_F(WARNING,
           "`mhlo.num_replicas` attribute not found, assuming default number "
           "of replicas: %zu",
           m_num_replicas);
  }

  if (!num_partitions_attr && !num_replicas_attr) {
    // When both mhlo.num_partitions and mhlo.num_replicas are not populated
    // (torch_xla doesn't populate them), we estimate the number of devices from
    // the mesh shape.
    DLOG_F(WARNING, "Num replicas and num partitions are not set, inferring "
                    "the number of devices from mesh shape");
    m_num_devices_to_utilize =
        std::accumulate(m_devices_mesh_shape.begin(),
                        m_devices_mesh_shape.end(), 1, std::multiplies<>());
  } else {
    // If at least one mhlo parameter is populated we assume the default value
    // of the other one.
    m_num_devices_to_utilize = m_num_partitions * m_num_replicas;
  }
}

void ModuleBuilder::convertFromTTIRToTTNN(
    const std::string &system_descriptor_path,
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const CompileOptions &compile_options) {
  mlir::PassManager ttir_to_ttnn_pm(mlir_module.get()->getName());

  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;

  options.optimizerPassEnabled = compile_options.enable_optimizer;
  options.memoryLayoutAnalysisEnabled = compile_options.enable_optimizer;
  options.enableBfp8Conversion = compile_options.enable_bfp8_conversion;
  options.systemDescPath = system_descriptor_path.data();

  if (m_devices_mesh_shape.size() != 2) {
    DLOG_F(ERROR,
           "Invalid mesh shape size: %zu. Shape must have two dimensions!",
           m_devices_mesh_shape.size());
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  options.meshShape = {m_devices_mesh_shape[0], m_devices_mesh_shape[1]};
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(ttir_to_ttnn_pm, options);

  // Run the pass manager.
  if (mlir::failed(ttir_to_ttnn_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from TTIR to TTNN module");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  DLOG_F(LOG_DEBUG, "TTNN Module:");
  printModule(mlir_module);
}

void ModuleBuilder::createFlatbufferBinary(
    const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  m_flatbuffer_binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());

  verifyCreatedFlatbufferBinary();
}

void ModuleBuilder::verifyCreatedFlatbufferBinary() {
  if (m_flatbuffer_binary.handle == nullptr) {
    DLOG_F(ERROR, "Failed to generate flatbuffer binary");
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;
  size_t num_inputs =
      m_flatbuffer_binary.getProgramInputs(program_index).size();
  std::vector<tt::runtime::TensorDesc> output_specs =
      m_flatbuffer_binary.getProgramOutputs(program_index);
  size_t num_outputs = output_specs.size();

  if (num_inputs != m_input_shardings.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of inputs %zu "
           "than expected from the m_input_shardings %zu",
           num_inputs, m_input_shardings.size());
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  if (num_outputs != m_is_output_scalar.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of outputs %zu "
           "than expected from the m_is_output_scalar %zu",
           num_outputs, m_is_output_scalar.size());
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  if (num_outputs != m_output_shardings.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of outputs %zu "
           "than expected from the m_output_shardings %zu",
           num_outputs, m_output_shardings.size());
    m_status = tt_pjrt_status::kInternal;
    return;
  }

  checkOutputShardingShapes(output_specs);
}

void ModuleBuilder::checkOutputShardingShapes(
    const std::vector<tt::runtime::TensorDesc> &output_specs) {
  for (size_t output_index = 0; output_index < output_specs.size();
       ++output_index) {
    const mlir::tt::sharding_utils::MeshSharding &output_sharding =
        m_output_shardings[output_index];
    if (output_sharding.getShardType() ==
            mlir::tt::ttcore::MeshShardType::Identity ||
        output_sharding.getShardType() ==
            mlir::tt::ttcore::MeshShardType::Replicate) {
      continue;
    }

    const llvm::ArrayRef<int64_t> &shard_shape =
        output_sharding.getShardShape();
    const std::vector<std::uint32_t> &output_shape =
        output_specs[output_index].shape;

    if (shard_shape.size() != output_shape.size()) {
      DLOG_F(ERROR,
             "Output sharding shape (%zu) doesn't match the output shape (%zu)",
             shard_shape.size(), output_shape.size());

      m_status = tt_pjrt_status::kInternal;
      return;
    }

    for (size_t shard_dim = 0; shard_dim < shard_shape.size(); ++shard_dim) {
      if (output_shape[shard_dim] % shard_shape[shard_dim] != 0) {
        DLOG_F(ERROR,
               "Output shape (%u) is not divisible by the sharding shape (%zu)",
               output_shape[shard_dim], shard_shape[shard_dim]);

        m_status = tt_pjrt_status::kInternal;
        return;
      }
    }
  }
}

void ModuleBuilder::printModule(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  if (loguru::g_stderr_verbosity < LOG_DEBUG) {
    return;
  }

  mlir_module->dump();
}

bool ModuleBuilder::isUsingShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // If the module is using the Shardy dielect, it should have the
  // xla.sdy.meshes attribute denoting the shape of its meshes as a module
  // attribute. Note: this is only true for the Shardy dialect gotten directly
  // from xla, after passing trough SdyRoundTripImportPipeline, it will no
  // longer have this attribute.
  if (mlir::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
          module.get(), mlir::sdy::kMeshesRoundTripAttr)
          .has_value()) {
    return true;
  }

  // After running through the SdyRoundTripImportPipeline, the module which uses
  // shardy dialect will have the sdy.mesh op.
  std::optional<mlir::sdy::MeshOp> mesh_op = getFirstShardyMeshOp(module);

  return mesh_op.has_value();
}

bool ModuleBuilder::isUsingShardyManualComputation(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (!isUsingShardy(module)) {
    return false;
  }

  bool is_using_shardy_manual_computation = false;
  module.get().walk([&](mlir::sdy::ManualComputationOp op) {
    is_using_shardy_manual_computation = true;

    return mlir::WalkResult::interrupt();
  });

  return is_using_shardy_manual_computation;
}

std::optional<mlir::sdy::MeshOp> ModuleBuilder::getFirstShardyMeshOp(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  std::optional<mlir::sdy::MeshOp> mesh_op;
  module.get().walk([&](mlir::sdy::MeshOp op) {
    mesh_op = op;
    return mlir::WalkResult::interrupt();
  });
  return mesh_op;
}

llvm::StringMap<llvm::SmallVector<mlir::tt::ttcore::ArgumentType>>
ModuleBuilder::createArgumentTypeMap(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::StringMap<llvm::SmallVector<mlir::tt::ttcore::ArgumentType>>
      argTypesMap;

  if (m_input_argument_roles.empty()) {
    return argTypesMap;
  }

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  size_t arg_offset = 0;

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    llvm::SmallVector<mlir::tt::ttcore::ArgumentType> argTypes;
    for (unsigned int i = 0; i < func_op.getNumArguments(); ++i) {
      assert(arg_offset + i < m_input_argument_roles.size() &&
             "TTIR module should have the same number of input arguments as "
             "the SHLO module");
      if (m_input_argument_roles[arg_offset + i] ==
          InputArgumentRole::kWeight) {
        argTypes.push_back(mlir::tt::ttcore::ArgumentType::Constant);
      } else {
        argTypes.push_back(mlir::tt::ttcore::ArgumentType::Input);
      }
    }
    argTypesMap[func_op.getName().str()] = argTypes;
    arg_offset += func_op.getNumArguments();
  }

  return argTypesMap;
}

} // namespace tt::pjrt::module_builder
