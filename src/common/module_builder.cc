// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/module_builder.h"

// c++ standard library includes
#include <cstdlib>

// loguru includes
#include "loguru/loguru.hpp"

// llvm includes
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

// llvm mlir includes
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// stablehlo includes
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

// shardy includes
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/round_trip_import/constants.h"
#include "shardy/round_trip_import/pipelines.h"
#include "shardy/round_trip_import/utils.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

namespace tt::pjrt {

ModuleBuilder::ModuleBuilder()
    : m_status(tt_pjrt_status::kSuccess), m_flatbuffer_binary(nullptr) {
  m_context = std::make_unique<mlir::MLIRContext>();

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

  m_context->appendDialectRegistry(registry);
}

tt_pjrt_status
ModuleBuilder::buildModule(const std::string_view &code,
                           const std::string_view &format,
                           const std::string &system_descriptor_path) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::buildModule");

  m_status = tt_pjrt_status::kSuccess;

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module = createVHLOModule(code);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  collectNumDevicesToUtilize(mlir_module);

  convertFromVHLOToSHLO(mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  collectInputShardings(mlir_module);
  collectOutputShardings(mlir_module);
  collectOutputTypes(mlir_module);

  convertFromSHLOToTTIR(mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  convertFromTTIRToTTNN(system_descriptor_path, mlir_module);
  if (!tt_pjrt_status_is_ok(m_status)) {
    return m_status;
  }

  createFlatbufferBinary(mlir_module);

  return m_status;
}

mlir::OwningOpRef<mlir::ModuleOp>
ModuleBuilder::createVHLOModule(const std::string_view &code) {
  mlir::OwningOpRef<mlir::ModuleOp> vhlo_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(code.data(), code.size()),
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

void ModuleBuilder::collectNumDevicesToUtilize(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  auto num_partitions_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_partitions");
  auto num_replicas_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_replicas");
  if (num_partitions_attr && num_replicas_attr) {
    m_num_devices_to_utilize =
        num_partitions_attr.getInt() * num_replicas_attr.getInt();
  } else {
    m_num_devices_to_utilize = 1;
    DLOG_F(WARNING, "mhlo.num_partitions, mhlo.num_replicas not found, using "
                    "default number of devices: 1");
  }
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

  // TODO(wooseoklee) : This is a temporary solution for the "roundtrip" mlir
  // from openXLA. Once openXLA natively supports Shardy, we can remove
  // following import passes. https://github.com/tenstorrent/tt-xla/issues/284
  // Detect Shardy by looking at the meshes attribute in module.
  bool is_shardy_dialect =
      mlir::sdy::tryGetFrontendAttr<mlir::DictionaryAttr>(
          mlir_module.get(), mlir::sdy::kMeshesRoundTripAttr)
          .has_value();
  if (is_shardy_dialect) {
    mlir::PassManager shardy_pm(mlir_module.get()->getName());
    mlir::sdy::addSdyRoundTripImportPipeline(shardy_pm);
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

void ModuleBuilder::collectInputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::collectInputShardings");
  m_input_shardings.clear();

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumArguments(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getArgAttr(i, mlir::tt::sharding_utils::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, m_input_shardings);
  if (result.failed()) {
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::collectOutputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::collectOutputShardings");
  m_output_shardings.clear();

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumResults(); ++i) {
      gspmd_attributes.push_back(
          llvm::dyn_cast_if_present<mlir::StringAttr>(func_op.getResultAttr(
              i, mlir::tt::sharding_utils::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, m_output_shardings);
  if (result.failed()) {
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::collectOutputTypes(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::collectOutputTypes");

  m_is_output_scalar.clear();

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);

  for (mlir::func::FuncOp &func_op : publicFuncOps) {

    for (const mlir::Type &returnType :
         func_op.getFunctionType().getResults()) {
      m_is_output_scalar.push_back(isScalarType(returnType));
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

    mlir::tt::sharding_utils::MeshSharding mesh_sharding;

    // If there is no sharding attribute, we put the default sharding, marked
    // as "manual", which means there is no sharding.
    if (!gspmd_attr) {
      shardings.push_back(mesh_sharding);
      continue;
    }

    llvm::Expected<bool> error =
        mesh_sharding.convertGSPMDShardingToMeshSharding(gspmd_attr.getValue());
    if (llvm::Error e = error.takeError()) {
      DLOG_F(ERROR, "Failed to convert sharding attribute to mesh sharding");

      return llvm::LogicalResult::failure();
    }

    shardings.push_back(mesh_sharding);
  }

  return llvm::LogicalResult::success();
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

void ModuleBuilder::convertFromTTIRToTTNN(
    const std::string &system_descriptor_path,
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::PassManager ttir_to_ttnn_pm(mlir_module.get()->getName());

  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;
  options.systemDescPath = system_descriptor_path.data();
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

  if (m_flatbuffer_binary.handle == nullptr) {
    DLOG_F(ERROR, "Failed to generate flatbuffer binary");
    m_status = tt_pjrt_status::kInternal;
  }
}

void ModuleBuilder::printModule(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  if (loguru::g_stderr_verbosity < LOG_DEBUG) {
    return;
  }

  mlir_module->dump();
}

} // namespace tt::pjrt
