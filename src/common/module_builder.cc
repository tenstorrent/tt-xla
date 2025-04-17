// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/module_builder.h"

// c++ standard library includes
#include <cstdlib>
#include <optional>

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

const std::string ModuleBuilder::c_mlir_format_name = "mlir";

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

tt_pjrt_status
ModuleBuilder::buildModule(const std::string_view &mlir_code,
                           const std::string &system_descriptor_path) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::buildModule");

  m_status = tt_pjrt_status::kSuccess;

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module = createVHLOModule(mlir_code);
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

  collectMeshShape(mlir_module);

  convertFromTTIRToTTNN(system_descriptor_path, mlir_module);
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

  m_num_devices_to_utilize = m_num_partitions * m_num_replicas;
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
  if (isUsingShardy(mlir_module)) {
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
          func_op.getArgAttr(i, mlir::tt::sharding_utils::kXlaShardingAttr)));
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

void ModuleBuilder::collectOutputTypes(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
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
    // as "identity", which means there is no sharding.
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

mlir::LogicalResult ModuleBuilder::createShardingsFromShardy(
    std::vector<mlir::sdy::TensorShardingAttr> &shardy_attributes,
    const mlir::sdy::MeshAttr &shardy_mesh,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings) {
  for (const mlir::sdy::TensorShardingAttr &shardy_attr : shardy_attributes) {

    mlir::tt::sharding_utils::MeshSharding mesh_sharding;

    // If there is no sharding attribute, we put the default sharding, marked
    // as "identity", which means there is no sharding.
    if (!shardy_attr) {
      shardings.push_back(mesh_sharding);
      continue;
    }

    llvm::Expected<bool> error = mesh_sharding.convertSdyShardingToMeshSharding(
        shardy_attr, shardy_mesh, mlir::tt::MeshShardDirection::FullToShard);
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

void ModuleBuilder::collectMeshShape(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::tt::MeshesAttr meshes_attr =
      module.get()->getAttrOfType<mlir::tt::MeshesAttr>(
          mlir::tt::MeshesAttr::name);
  if (!meshes_attr || meshes_attr.getMeshes().empty()) {
    // If mesh attribute is not set we can still estimate the mesh based on the
    // input shardings.
    estimateMeshShape();
    return;
  }

  llvm::ArrayRef<mlir::tt::MeshAttr> meshes = meshes_attr.getMeshes();

  // For now, use the first mesh shape (same as what is used in tt-mlir).
  llvm::ArrayRef<int64_t> mesh_shape = meshes[0].getShape();

  m_devices_mesh_shape =
      std::vector<std::uint32_t>(mesh_shape.begin(), mesh_shape.end());
}

void ModuleBuilder::estimateMeshShape() {
  for (const mlir::tt::sharding_utils::MeshSharding &input_sharding :
       m_input_shardings) {
    if (input_sharding.getShardType() == mlir::tt::MeshShardType::Devices) {
      m_devices_mesh_shape =
          std::vector<std::uint32_t>(input_sharding.getMeshShape().begin(),
                                     input_sharding.getMeshShape().end());
      return;
    }
  }

  // Assuming single device if there are no inputs sharded on device.
  m_devices_mesh_shape = {1, 1};
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

std::optional<mlir::sdy::MeshOp> ModuleBuilder::getFirstShardyMeshOp(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  std::optional<mlir::sdy::MeshOp> mesh_op;
  module.get().walk([&](mlir::sdy::MeshOp op) {
    mesh_op = op;
    return mlir::WalkResult::interrupt();
  });
  return mesh_op;
}

} // namespace tt::pjrt
