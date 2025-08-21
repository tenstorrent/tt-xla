// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/module_builder/module_builder.h"

// c++ standard library includes
#include <cassert>
#include <cstdlib>
#include <numeric>

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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// stablehlo includes
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

// shardy includes
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
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
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/data_type_utils.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/pjrt_implementation/memory_instance.h"
#include "common/pjrt_implementation/module_builder/frontend_passes/shlo_input_role_propagation.h"
#include "common/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace tt::pjrt::module_builder {

const std::string c_mlir_format_name = "mlir";

ModuleBuilder::ModuleBuilder()
    : m_context(std::make_unique<mlir::MLIRContext>()) {
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

std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
ModuleBuilder::buildModule(
    const std::string_view &mlir_code,
    const std::string &system_descriptor_path,
    const std::unordered_map<std::string, std::string> &compile_options_map,
    ClientInstance *client_instance) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::buildModule");

  auto compile_options = CompileOptions::parse(compile_options_map);

  tt_pjrt_status status;
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  status = createVHLOModule(mlir_code, mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  std::string original_mlir_code(mlir_code);

  status = convertFromVHLOToSHLO(mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  status = runFrontendSHLOPipeline(mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings;
  status = collectInputShardings(mlir_module, input_shardings);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  std::vector<mlir::tt::sharding_utils::MeshSharding> output_shardings;
  status = collectOutputShardings(mlir_module, output_shardings);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  NumArgumentsResult num_arguments;
  status = collectNumArguments(mlir_module, num_arguments);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  std::vector<PJRT_Buffer_Type> output_types = collectOutputTypes(mlir_module);

  status = runCompilerStableHLOPipeline(mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  std::string ttir_mlir;
  status = convertFromSHLOToTTIR(mlir_module, ttir_mlir);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  std::vector<std::uint32_t> mesh_shape =
      collectMeshShape(mlir_module, input_shardings);

  NumDevicesResult num_devices_result =
      collectNumDevicesToUtilize(mlir_module, mesh_shape);
  size_t num_partitions = num_devices_result.num_partitions;
  size_t num_replicas = num_devices_result.num_replicas;
  size_t num_devices_to_utilize = num_devices_result.num_devices_to_utilize;

  std::string ttnn_mlir;
  status = convertFromTTIRToTTNN(system_descriptor_path, mlir_module,
                                 compile_options, client_instance, mesh_shape, ttnn_mlir);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  tt::runtime::Binary flatbuffer(nullptr);
  status = createFlatbufferBinary(mlir_module, input_shardings,
                                  output_shardings, flatbuffer);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  // Collect memory kinds for output buffers
  std::vector<const char *> output_memory_kinds;
  std::vector<size_t> output_memory_kinds_sizes;
  collectMemoryKinds(num_arguments.num_outputs, output_memory_kinds,
                     output_memory_kinds_sizes);

  // TODO(mrakita): Use the VHLO module name from the module builder, if it has
  // a name, otherwise some default string like the current one.
  std::string executable_name = "tt_executable";

  return {tt_pjrt_status::kSuccess,
          ExecutableImage::createInstance(
              flatbuffer, std::move(original_mlir_code), std::move(ttir_mlir),
              std::move(ttnn_mlir), std::move(executable_name),
              num_arguments.num_inputs, num_arguments.num_outputs,
              std::move(num_arguments.output_dimensions),
              std::move(num_arguments.output_ranks),
              std::move(num_arguments.output_dimensions_flat), num_partitions,
              num_replicas, num_devices_to_utilize, mesh_shape, input_shardings,
              output_shardings, output_types, std::move(output_memory_kinds),
              std::move(output_memory_kinds_sizes),
              std::move(compile_options))};
}

tt_pjrt_status ModuleBuilder::createVHLOModule(
    const std::string_view &mlir_code,
    mlir::OwningOpRef<mlir::ModuleOp> &vhlo_module) {
  vhlo_module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlir_code.data(), mlir_code.size()),
      mlir::ParserConfig{m_context.get(), /*verifyAfterParse=*/true});

  if (!vhlo_module) {
    DLOG_F(ERROR, "Failed to create VHLO module from the input program code");
    return tt_pjrt_status::kInternal;
  }

  DLOG_F(LOG_DEBUG, "VHLO Module:");
  printModule(vhlo_module);

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::convertFromVHLOToSHLO(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::PassManager vhlo_to_shlo_pm(mlir_module.get()->getName());

  mlir::stablehlo::createStablehloDeserializePipeline(vhlo_to_shlo_pm);

  enableVerboseIRPrinting(vhlo_to_shlo_pm);

  if (mlir::failed(vhlo_to_shlo_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from VHLO to SHLO module");
    return tt_pjrt_status::kInternal;
  }

  DLOG_F(LOG_DEBUG, "SHLO Module:");
  printModule(mlir_module);

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::runFrontendSHLOPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {

  tt_pjrt_status status =
      frontend_passes::annotateArgumentAttributes(mlir_module);

  DLOG_F(LOG_DEBUG, "SHLO Module after frontend StableHLO pipeline:");
  printModule(mlir_module);

  return status;
}

std::string
ModuleBuilder::getMlirCode(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  std::string mlir_code;
  llvm::raw_string_ostream os(mlir_code);
  mlir_module->print(os, mlir::OpPrintingFlags().enableDebugInfo());
  os.flush();
  return mlir_code;
}

tt_pjrt_status ModuleBuilder::collectInputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings) {
  if (auto shardy_shardings_opt = collectInputShardingsShardy(module)) {
    input_shardings = *shardy_shardings_opt;
    return tt_pjrt_status::kSuccess;
  }
  return collectInputShardingsGSPMD(module, input_shardings);
}

tt_pjrt_status ModuleBuilder::collectInputShardingsGSPMD(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumArguments(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getArgAttr(i, mlir::tt::gspmd_utils::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, input_shardings);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create input shardings from GSPMD attributes");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
ModuleBuilder::collectInputShardingsShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  auto mesh_op_opt = getFirstShardyMeshOp(module);
  if (!mesh_op_opt.has_value()) {
    return std::nullopt;
  }

  mlir::sdy::MeshOp mesh_op = *mesh_op_opt;
  mlir::sdy::MeshAttr shardy_mesh = mesh_op.getMesh();
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

  std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings;
  mlir::LogicalResult result = createShardingsFromShardy(
      shardy_attributes, shardy_mesh, input_shardings);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create input shardings from Shardy attributes");
    return std::nullopt;
  }
  return input_shardings;
}

tt_pjrt_status ModuleBuilder::collectOutputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings) {
  if (auto shardy_shardings_opt = collectOutputShardingsShardy(module)) {
    output_shardings = *shardy_shardings_opt;
    return tt_pjrt_status::kSuccess;
  }
  return collectOutputShardingsGSPMD(module, output_shardings);
}

tt_pjrt_status ModuleBuilder::collectOutputShardingsGSPMD(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings) {
  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumResults(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getResultAttr(i, mlir::tt::gspmd_utils::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, output_shardings);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create output shardings from GSPMD attributes");
    return tt_pjrt_status::kInternal;
  }
  return tt_pjrt_status::kSuccess;
}

std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
ModuleBuilder::collectOutputShardingsShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  auto mesh_op_opt = getFirstShardyMeshOp(module);
  if (!mesh_op_opt.has_value()) {
    return std::nullopt;
  }

  mlir::sdy::MeshOp mesh_op = *mesh_op_opt;
  mlir::sdy::MeshAttr shardy_mesh = mesh_op.getMesh();
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

  std::vector<mlir::tt::sharding_utils::MeshSharding> output_shardings;
  mlir::LogicalResult result = createShardingsFromShardy(
      shardy_attributes, shardy_mesh, output_shardings);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create output shardings from Shardy attributes");
    return std::nullopt;
  }
  return output_shardings;
}

std::vector<PJRT_Buffer_Type> ModuleBuilder::collectOutputTypes(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);

  std::vector<PJRT_Buffer_Type> output_types;

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (const mlir::Type &returnType :
         func_op.getFunctionType().getResults()) {
      output_types.push_back(
          tt::pjrt::data_type_utils::convertMLIRToPJRTDataType(returnType));
    }
  }

  return output_types;
}

tt_pjrt_status ModuleBuilder::collectNumArguments(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    NumArgumentsResult &result) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  if (publicFuncOps.size() != 1) {
    DLOG_F(ERROR, "Expected exactly one public function, found: %zu",
           publicFuncOps.size());
    return tt_pjrt_status::kInternal;
  }

  mlir::func::FuncOp &func_op = publicFuncOps[0];

  result.num_inputs = func_op.getNumArguments();
  result.num_outputs = func_op.getNumResults();

  for (size_t result_index = 0; result_index < func_op.getNumResults();
       ++result_index) {
    mlir::Type result_type = func_op.getFunctionType().getResult(result_index);

    if (auto ranked_tensor_type =
            mlir::dyn_cast<mlir::RankedTensorType>(result_type)) {

      llvm::ArrayRef<int64_t> shape = ranked_tensor_type.getShape();
      std::vector<std::uint32_t> output_shape;

      for (int64_t dim : shape) {
        assert(dim != mlir::ShapedType::kDynamic &&
               "Dynamic dimensions not supported");
        output_shape.push_back(static_cast<std::uint32_t>(dim));
      }

      result.output_dimensions.emplace_back(output_shape);
      result.output_ranks.push_back(output_shape.size());

      for (std::uint32_t dim : output_shape) {
        result.output_dimensions_flat.push_back(static_cast<std::int64_t>(dim));
      }
    } else {
      assert(false && "Expected ranked tensor type for function result");
    }
  }
  return tt_pjrt_status::kSuccess;
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

tt_pjrt_status ModuleBuilder::runCompilerStableHLOPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  mlir::PassManager stablehlo_pipeline_pm(mlir_module.get()->getName(),
                                          mlir::PassManager::Nesting::Implicit);
  mlir::tt::stablehlo::StableHLOPipelineOptions stablehlo_pipeline_options;
  mlir::tt::stablehlo::createStableHLOPipeline(stablehlo_pipeline_pm,
                                               stablehlo_pipeline_options);

  enableVerboseIRPrinting(stablehlo_pipeline_pm);

  if (mlir::failed(stablehlo_pipeline_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to run stablehlo pipeline");
    return tt_pjrt_status::kInternal;
  }

  DLOG_F(LOG_DEBUG, "SHLO Module after compiler StableHLO pipeline:");
  printModule(mlir_module);

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::convertFromSHLOToTTIR(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, std::string &ttir_mlir) {
  // Implicit nesting required to call the stablehlo.composite --> func.call
  // conversion.
  mlir::PassManager shlo_to_ttir_pm(mlir_module.get()->getName(),
                                    mlir::PassManager::Nesting::Implicit);

  mlir::tt::ttir::StableHLOToTTIRPipelineOptions shlo_options;
  shlo_options.arithDialectConversionsEnabled = true;
  shlo_options.legalizeCompositeToCallEnabled = true;
  mlir::tt::ttir::createStableHLOToTTIRPipeline(shlo_to_ttir_pm, shlo_options);

  enableVerboseIRPrinting(shlo_to_ttir_pm);

  if (mlir::failed(shlo_to_ttir_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from SHLO to TTIR module");
    return tt_pjrt_status::kInternal;
  }

  ttir_mlir = getMlirCode(mlir_module);

  DLOG_F(LOG_DEBUG, "TTIR Module:");
  printModule(mlir_module);

  return tt_pjrt_status::kSuccess;
}

std::vector<std::uint32_t> ModuleBuilder::collectMeshShape(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings) {
  mlir::tt::ttcore::MeshesAttr meshes_attr =
      module.get()->getAttrOfType<mlir::tt::ttcore::MeshesAttr>(
          mlir::tt::ttcore::MeshesAttr::name);
  if (!meshes_attr || meshes_attr.getMeshes().empty()) {
    // If mesh attribute is not set we can still estimate the mesh based on the
    // input shardings.
    return estimateMeshShape(input_shardings);
  }

  llvm::ArrayRef<mlir::tt::ttcore::MeshAttr> meshes = meshes_attr.getMeshes();

  // For now, use the first mesh shape (same as what is used in tt-mlir).
  llvm::ArrayRef<int64_t> mesh_shape = meshes[0].getShape();

  return std::vector<std::uint32_t>(mesh_shape.begin(), mesh_shape.end());
}

std::vector<std::uint32_t> ModuleBuilder::estimateMeshShape(
    std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings) {
  for (const mlir::tt::sharding_utils::MeshSharding &input_sharding :
       input_shardings) {
    if (input_sharding.getShardType() ==
        mlir::tt::ttcore::MeshShardType::Devices) {
      return std::vector<std::uint32_t>(input_sharding.getMeshShape().begin(),
                                        input_sharding.getMeshShape().end());
    }
  }

  // Assuming single device if there are no inputs sharded on device.
  return {1, 1};
}

NumDevicesResult ModuleBuilder::collectNumDevicesToUtilize(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    std::vector<std::uint32_t> devices_mesh_shape) {
  auto num_partitions_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_partitions");

  // Assuming one partition by default.
  size_t num_partitions = 1;

  if (num_partitions_attr) {
    num_partitions = static_cast<size_t>(num_partitions_attr.getInt());
  } else {
    DLOG_F(WARNING,
           "`mhlo.num_partitions` attribute not found, assuming default number "
           "of partitions: %zu",
           num_partitions);
  }

  auto num_replicas_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_replicas");
  // Assuming one replica by default.
  size_t num_replicas = 1;
  if (num_replicas_attr) {
    num_replicas = static_cast<size_t>(num_replicas_attr.getInt());
  } else {
    DLOG_F(WARNING,
           "`mhlo.num_replicas` attribute not found, assuming default number "
           "of replicas: %zu",
           num_replicas);
  }

  size_t num_devices_to_utilize = num_partitions * num_replicas;

  if (!num_partitions_attr && !num_replicas_attr) {
    // When both mhlo.num_partitions and mhlo.num_replicas are not populated
    // (torch_xla doesn't populate them), we estimate the number of devices from
    // the mesh shape.
    DLOG_F(WARNING, "Num replicas and num partitions are not set, inferring "
                    "the number of devices from mesh shape");
    num_devices_to_utilize =
        std::accumulate(devices_mesh_shape.begin(), devices_mesh_shape.end(), 1,
                        std::multiplies<>());
  } else {
    // If at least one mhlo parameter is populated we assume the default value
    // of the other one.
    num_devices_to_utilize = num_partitions * num_replicas;
  }

  return {.num_partitions = num_partitions,
          .num_replicas = num_replicas,
          .num_devices_to_utilize = num_devices_to_utilize};
}

tt_pjrt_status ModuleBuilder::convertFromTTIRToTTNN(
    const std::string &system_descriptor_path,
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const CompileOptions &compile_options, ClientInstance *client_instance,
    std::vector<std::uint32_t> devices_mesh_shape,
    std::string &ttnn_mlir) {
  mlir::PassManager ttir_to_ttnn_pm(mlir_module.get()->getName());

  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;

  options.optimizerPassEnabled = compile_options.enable_optimizer;
  options.memoryLayoutAnalysisEnabled = compile_options.enable_optimizer;
  options.enableBfp8Conversion = compile_options.enable_bfp8_conversion;
  options.systemDescPath = system_descriptor_path.data();

  if (devices_mesh_shape.size() != 2) {
    DLOG_F(ERROR,
           "Invalid mesh shape size: %zu. Shape must have two dimensions!",
           devices_mesh_shape.size());
    return tt_pjrt_status::kInternal;
  }

  options.meshShape = {devices_mesh_shape[0], devices_mesh_shape[1]};

  // Use the `options.devicePtr` to pass the device pointer to the optimizer in
  // order to avoid closing and reopening the device afterwards.
  tt::runtime::Device runtime_device =
      client_instance->getOrCreateMeshDevice(devices_mesh_shape);
  options.devicePtr =
      std::static_pointer_cast<tt::tt_metal::distributed::MeshDevice>(
          runtime_device.handle);
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(ttir_to_ttnn_pm, options);

  enableVerboseIRPrinting(ttir_to_ttnn_pm);

  // Run the pass manager.
  if (mlir::failed(ttir_to_ttnn_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from TTIR to TTNN module");
    return tt_pjrt_status::kInternal;
  }

  ttnn_mlir = getMlirCode(mlir_module);

  DLOG_F(LOG_DEBUG, "TTNN Module:");
  printModule(mlir_module);

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::createFlatbufferBinary(
    const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings,
    tt::runtime::Binary &flatbuffer_binary) {
  flatbuffer_binary = mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());

  tt_pjrt_status status = verifyCreatedFlatbufferBinary(
      flatbuffer_binary, input_shardings, output_shardings);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::verifyCreatedFlatbufferBinary(
    const tt::runtime::Binary &flatbuffer_binary,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings,
    const std::vector<mlir::tt::sharding_utils::MeshSharding>
        &output_shardings) {
  if (flatbuffer_binary.handle == nullptr) {
    DLOG_F(ERROR, "Failed to generate flatbuffer binary");
    return tt_pjrt_status::kInternal;
  }

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;
  size_t num_inputs = flatbuffer_binary.getProgramInputs(program_index).size();
  std::vector<tt::runtime::TensorDesc> output_specs =
      flatbuffer_binary.getProgramOutputs(program_index);
  size_t num_outputs = output_specs.size();

  if (num_inputs != input_shardings.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of inputs %zu"
           "than expected from the m_input_shardings %zu",
           num_inputs, input_shardings.size());
    return tt_pjrt_status::kInternal;
  }

  if (num_outputs != output_shardings.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of outputs %zu "
           "than expected from the m_output_shardings %zu",
           num_outputs, output_shardings.size());
    return tt_pjrt_status::kInternal;
  }

  return checkOutputShardingShapes(output_specs, output_shardings);
}

tt_pjrt_status ModuleBuilder::checkOutputShardingShapes(
    const std::vector<tt::runtime::TensorDesc> &output_specs,
    const std::vector<mlir::tt::sharding_utils::MeshSharding>
        &output_shardings) {
  for (size_t output_index = 0; output_index < output_specs.size();
       ++output_index) {
    const mlir::tt::sharding_utils::MeshSharding &output_sharding =
        output_shardings[output_index];
    if (output_sharding.getShardType() ==
            mlir::tt::ttcore::MeshShardType::Identity ||
        output_sharding.getShardType() ==
            mlir::tt::ttcore::MeshShardType::Replicate) {
      continue;
    }

    const llvm::SmallVector<int64_t> &shard_shape =
        output_sharding.getShardShape();
    const std::vector<std::uint32_t> &output_shape =
        output_specs[output_index].shape;

    if (shard_shape.size() != output_shape.size()) {
      DLOG_F(ERROR,
             "Output sharding shape (%zu) doesn't match the output shape (%zu)",
             shard_shape.size(), output_shape.size());

      return tt_pjrt_status::kInternal;
    }

    for (size_t shard_dim = 0; shard_dim < shard_shape.size(); ++shard_dim) {
      if (output_shape[shard_dim] % shard_shape[shard_dim] != 0) {
        DLOG_F(ERROR,
               "Output shape (%u) is not divisible by the sharding shape (%zu)",
               output_shape[shard_dim], shard_shape[shard_dim]);

        return tt_pjrt_status::kInternal;
      }
    }
  }
  return tt_pjrt_status::kSuccess;
}

void ModuleBuilder::printModule(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  if (loguru::g_stderr_verbosity < LOG_DEBUG) {
    return;
  }

  mlir_module->print(llvm::errs(), mlir::OpPrintingFlags().enableDebugInfo());
}

void ModuleBuilder::enableVerboseIRPrinting(mlir::PassManager &pm) {
  if (loguru::g_stderr_verbosity < LOG_VERBOSE) {
    return;
  }

  // Multithreading must be disabled when printing at module scope
  // to avoid interleaved output.
  pm.getContext()->disableMultithreading();
  pm.enableIRPrinting();
}

bool ModuleBuilder::isUsingShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // After running through the SdyRoundTripImportPipeline, the module which uses
  // shardy dialect will have the sdy.mesh op.
  return getFirstShardyMeshOp(module).has_value();
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

void ModuleBuilder::collectMemoryKinds(
    size_t num_outputs, std::vector<const char *> &output_memory_kinds,
    std::vector<size_t> &output_memory_kinds_sizes) {
  output_memory_kinds.reserve(num_outputs);
  output_memory_kinds_sizes.reserve(num_outputs);

  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    output_memory_kinds.emplace_back(
        MemoryInstance::c_device_memory_kind_name.c_str());
    output_memory_kinds_sizes.emplace_back(
        MemoryInstance::c_device_memory_kind_name.size());
  }
}

std::optional<mlir::sdy::MeshOp> ModuleBuilder::getFirstShardyMeshOp(
    const mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::sdy::MeshOp found_mesh_op = nullptr;
  module.get().walk([&](mlir::sdy::MeshOp op) {
    found_mesh_op = op;
    return mlir::WalkResult::interrupt();
  });

  if (!found_mesh_op) {
    return std::nullopt;
  }

  return found_mesh_op;
}

} // namespace tt::pjrt::module_builder
