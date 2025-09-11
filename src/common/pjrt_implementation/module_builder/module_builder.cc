// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/module_builder/module_builder.h"

// c++ standard library includes
#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>

// loguru includes
#include "common/status.h"
#include "loguru/loguru.hpp"

// llvm includes
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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
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
#include "ttmlir/Target/Python/PythonEmitter.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

// tt-xla includes
#include "common/pjrt_implementation/data_type_utils.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/pjrt_implementation/module_builder/frontend_passes/shlo_input_role_propagation.h"

namespace tt::pjrt::module_builder {

const std::string c_mlir_format_name = "mlir";

ModuleBuilder::ModuleBuilder()
    : m_context(std::make_unique<mlir::MLIRContext>()),
      m_tt_alchemist_handle(nullptr), m_alchemist_available(false),
      m_tt_alchemist_get_instance(nullptr),
      m_tt_alchemist_generate_python(nullptr),
      m_tt_alchemist_generate_cpp(nullptr) {
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

  // Try to load tt-alchemist library and function pointers
  loadTTAlchemistFunctions();
}

ModuleBuilder::~ModuleBuilder() {
  if (m_tt_alchemist_handle) {
    dlclose(m_tt_alchemist_handle);
    m_tt_alchemist_handle = nullptr;
  }
}

std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
ModuleBuilder::buildModule(
    const std::string_view &mlir_code,
    const std::string &system_descriptor_path,
    const std::unordered_map<std::string, std::string> &compile_options_map) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::buildModule");

  auto compile_options = CompileOptions::parse(compile_options_map);

  if (compile_options.backend == Backend::Default) {
    auto fbexecutable = FlatbufferExecutableImage::createInstance();
    fbexecutable->m_compile_options = compile_options;

    auto [status, mlir_module] = buildCommon(mlir_code, fbexecutable.get());
    if (!tt_pjrt_status_is_ok(status)) {
      return {status, nullptr};
    }

    status = buildFlatbuffer(mlir_module, system_descriptor_path,
                             fbexecutable.get());
    if (!tt_pjrt_status_is_ok(status)) {
      return {status, nullptr};
    }

    fbexecutable->validate();
    return {status, fbexecutable};
  } else if (compile_options.backend == Backend::CodegenCpp) {
    auto soexecutable = SOExecutableImage::createInstance();
    soexecutable->m_compile_options = compile_options;
    auto [status, mlir_module] = buildCommon(mlir_code, soexecutable.get());
    if (!tt_pjrt_status_is_ok(status)) {
      return {status, nullptr};
    }
    status = buildForCodegenCpp(mlir_module, soexecutable.get());
    if (!tt_pjrt_status_is_ok(status)) {
      return {status, nullptr};
    }

    soexecutable->validate();
    return {status, soexecutable};
  } else if (compile_options.backend == Backend::CodegenPy) {
    auto soexecutable = SOExecutableImage::createInstance();
    soexecutable->m_compile_options = compile_options;
    auto [status, mlir_module] = buildCommon(mlir_code, soexecutable.get());
    if (!tt_pjrt_status_is_ok(status)) {
      return {status, nullptr};
    }
    status = buildForCodegenPy(mlir_module, soexecutable.get());
    if (!tt_pjrt_status_is_ok(status)) {
      return {status, nullptr};
    }

    soexecutable->validate();
    return {status, soexecutable};
  } else {
    DLOG_F(ERROR, "Unknown backend type");
    return {tt_pjrt_status::kInternal, nullptr};
  }
}

std::tuple<tt_pjrt_status, mlir::OwningOpRef<mlir::ModuleOp>>
ModuleBuilder::buildCommon(const std::string_view &mlir_code,
                           ExecutableImage *executable) {
  auto [status, mlir_module] = createVHLOModule(mlir_code);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  executable->m_original_mlir_code = std::string(mlir_code);

  status = convertFromVHLOToSHLO(mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  status = runFrontendSHLOPipeline(mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  status = collectInputShardings(mlir_module, executable);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  status = collectOutputShardings(mlir_module, executable);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  collectInputArgumentRoles(mlir_module, executable);
  collectNumArguments(mlir_module, executable);
  // Arg names are not always needed, but it is much easier to
  // always collect them, then to reparse TTIR text in LoadedExecutableInstance
  collectArgumentNames(mlir_module, executable);
  collectOutputTypes(mlir_module, executable);

  status = runCompilerStableHLOPipeline(mlir_module);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  status = convertFromSHLOToTTIR(mlir_module, executable);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  collectMeshShape(mlir_module, executable);
  collectNumDevicesToUtilize(mlir_module, executable);

  return {tt_pjrt_status::kSuccess, std::move(mlir_module)};
}

tt_pjrt_status
ModuleBuilder::buildFlatbuffer(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                               const std::string &system_descriptor_path,
                               FlatbufferExecutableImage *executable) {

  tt_pjrt_status status =
      convertFromTTIRToTTNN(system_descriptor_path, mlir_module,
                            executable->m_compile_options, executable);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }
  executable->m_ttnn_mlir = getMlirCode(mlir_module);

  return createFlatbufferBinary(mlir_module, executable);
}

tt_pjrt_status ModuleBuilder::buildForCodegenCpp(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    SOExecutableImage *executable) {
  std::string folder = executable->m_compile_options.export_path;
  std::filesystem::create_directories(folder);

  auto ttir = executable->m_ttir_mlir;
  std::ofstream ttir_file(folder + "/ttir.mlir");
  ttir_file << ttir;
  ttir_file.close();

  if (!m_alchemist_available) {
    DLOG_F(ERROR, "tt-alchemist library or functions not available");
    return tt_pjrt_status::kInternal;
  }

  void *instance = m_tt_alchemist_get_instance();
  if (!instance) {
    DLOG_F(ERROR, "Failed to get tt-alchemist instance");
    return tt_pjrt_status::kInternal;
  }

  auto input_file = folder + "/ttir.mlir";
  auto output_dir = folder;
  bool is_local = false;
  bool cpp_result = m_tt_alchemist_generate_cpp(
      instance, input_file.c_str(), output_dir.c_str(), is_local, "");
  if (!cpp_result) {
    DLOG_F(ERROR, "tt-alchemist generateCpp failed");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status
ModuleBuilder::buildForCodegenPy(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                                 SOExecutableImage *executable) {
  std::string folder = executable->m_compile_options.export_path;
  std::filesystem::create_directories(folder);

  auto ttir = executable->m_ttir_mlir;
  std::ofstream ttir_file(folder + "/ttir.mlir");
  ttir_file << ttir;
  ttir_file.close();

  if (!m_alchemist_available) {
    DLOG_F(ERROR, "tt-alchemist library or functions not available");
    return tt_pjrt_status::kInternal;
  }

  void *instance = m_tt_alchemist_get_instance();
  if (!instance) {
    DLOG_F(ERROR, "Failed to get tt-alchemist instance");
    return tt_pjrt_status::kInternal;
  }

  auto input_file = folder + "/ttir.mlir";
  auto output_dir = folder;
  bool is_local = false;
  bool python_result = m_tt_alchemist_generate_python(
      instance, input_file.c_str(), output_dir.c_str(), is_local, "");
  if (!python_result) {
    DLOG_F(ERROR, "tt-alchemist generatePython failed");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

std::tuple<tt_pjrt_status, mlir::OwningOpRef<mlir::ModuleOp>>
ModuleBuilder::createVHLOModule(const std::string_view &mlir_code) {
  mlir::OwningOpRef<mlir::ModuleOp> vhlo_module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(mlir_code.data(), mlir_code.size()),
          mlir::ParserConfig{m_context.get(), /*verifyAfterParse=*/true});

  if (!vhlo_module) {
    DLOG_F(ERROR, "Failed to create VHLO module from the input program code");
    return {tt_pjrt_status::kInternal, nullptr};
  }

  DLOG_F(LOG_DEBUG, "VHLO Module:");
  printModule(vhlo_module);

  return {tt_pjrt_status::kSuccess, std::move(vhlo_module)};
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
    ExecutableImage *executable) {
  return isUsingShardy(module) ? collectInputShardingsShardy(module, executable)
                               : collectInputShardingsGSPMD(module, executable);
}

tt_pjrt_status ModuleBuilder::collectInputShardingsGSPMD(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumArguments(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getArgAttr(i, mlir::tt::gspmd_utils::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, executable->m_input_sharding);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create input shardings from GSPMD attributes");
    return tt_pjrt_status::kInternal;
  }
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::collectInputShardingsShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  std::optional<mlir::sdy::MeshOp> mesh_op = getFirstShardyMeshOp(module);
  // Since this function is called only when we are using the shardy dialect,
  // this op should always be present.
  if (!mesh_op.has_value()) {
    DLOG_F(ERROR, "Failed to find mesh op in the module");
    return tt_pjrt_status::kInternal;
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
      shardy_attributes, shardy_mesh, executable->m_input_sharding);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create input shardings from Shardy attributes");
    return tt_pjrt_status::kInternal;
  }
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::collectOutputShardings(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  return isUsingShardy(module)
             ? collectOutputShardingsShardy(module, executable)
             : collectOutputShardingsGSPMD(module, executable);
}

tt_pjrt_status ModuleBuilder::collectOutputShardingsGSPMD(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  std::vector<mlir::StringAttr> gspmd_attributes;
  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int i = 0; i < func_op.getNumResults(); ++i) {
      gspmd_attributes.push_back(llvm::dyn_cast_if_present<mlir::StringAttr>(
          func_op.getResultAttr(i, mlir::tt::gspmd_utils::kXlaShardingAttr)));
    }
  }

  mlir::LogicalResult result =
      createShardingsFromGSPMD(gspmd_attributes, executable->m_output_sharding);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create output shardings from GSPMD attributes");
    return tt_pjrt_status::kInternal;
  }
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::collectOutputShardingsShardy(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  std::optional<mlir::sdy::MeshOp> mesh_op = getFirstShardyMeshOp(module);
  // Since this function is called only when we are using the shardy dialect,
  // this op should always be present.
  if (!mesh_op.has_value()) {
    DLOG_F(ERROR, "Failed to find mesh op in the module");
    return tt_pjrt_status::kInternal;
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
      shardy_attributes, shardy_mesh, executable->m_output_sharding);
  if (result.failed()) {
    DLOG_F(ERROR, "Failed to create output shardings from Shardy attributes");
    return tt_pjrt_status::kInternal;
  }
  return tt_pjrt_status::kSuccess;
}

void ModuleBuilder::collectInputArgumentRoles(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  executable->m_input_argument_roles.clear();

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (unsigned int arg_index = 0; arg_index < func_op.getNumArguments();
         ++arg_index) {
      // Check for ttcore.argument_type attribute
      mlir::StringAttr role_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
          arg_index, mlir::tt::ttcore::ArgumentTypeAttr::name);

      if (role_attr && role_attr.getValue() == "weight") {
        executable->m_input_argument_roles.push_back(
            tt::pjrt::InputArgumentRole::kWeight);
      } else {
        // Default to input if attribute is not present or not "weight"
        executable->m_input_argument_roles.push_back(
            tt::pjrt::InputArgumentRole::kInput);
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
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    for (const mlir::Type &returnType :
         func_op.getFunctionType().getResults()) {
      executable->m_output_types.push_back(
          tt::pjrt::data_type_utils::convertMLIRToPJRTDataType(returnType));
    }
  }
}

void ModuleBuilder::collectNumArguments(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  DLOG_F(LOG_DEBUG, "ModuleBuilder::collectNumArguments");

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  assert(publicFuncOps.size() == 1 && "Expected exactly one public function");

  mlir::func::FuncOp &func_op = publicFuncOps[0];

  executable->m_num_inputs = func_op.getNumArguments();
  executable->m_num_outputs = func_op.getNumResults();

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

      executable->m_output_dimensions.emplace_back(output_shape);
      executable->m_output_ranks.push_back(output_shape.size());

      for (std::uint32_t dim : output_shape) {
        executable->m_output_dimensions_flat.push_back(
            static_cast<std::int64_t>(dim));
      }
    } else {
      assert(false && "Expected ranked tensor type for function result");
    }
  }
}

void ModuleBuilder::collectArgumentNames(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  assert(publicFuncOps.size() == 1 && "Expected exactly one public function");
  mlir::func::FuncOp &func_op = publicFuncOps[0];

  std::vector<std::string> arg_locs;
  for (unsigned i = 0; i < func_op.getNumArguments(); ++i) {
    mlir::Location loc = func_op.getArgument(i).getLoc();
    std::string loc_str;
    llvm::raw_string_ostream stream(loc_str);
    loc.print(stream);
    arg_locs.push_back(std::move(loc_str));
  }

  executable->m_input_argument_names = arg_locs;
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
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    ExecutableImage *executable) {
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

  executable->m_ttir_mlir = getMlirCode(mlir_module);

  DLOG_F(LOG_DEBUG, "TTIR Module:");
  printModule(mlir_module);

  return tt_pjrt_status::kSuccess;
}

void ModuleBuilder::collectMeshShape(
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  mlir::tt::ttcore::MeshesAttr meshes_attr =
      module.get()->getAttrOfType<mlir::tt::ttcore::MeshesAttr>(
          mlir::tt::ttcore::MeshesAttr::name);
  if (!meshes_attr || meshes_attr.getMeshes().empty()) {
    // If mesh attribute is not set we can still estimate the mesh based on the
    // input shardings.
    estimateMeshShape(executable);
    return;
  }

  llvm::ArrayRef<mlir::tt::ttcore::MeshAttr> meshes = meshes_attr.getMeshes();

  // For now, use the first mesh shape (same as what is used in tt-mlir).
  llvm::ArrayRef<int64_t> mesh_shape = meshes[0].getShape();

  executable->m_devices_mesh_shape =
      std::vector<std::uint32_t>(mesh_shape.begin(), mesh_shape.end());
}

void ModuleBuilder::estimateMeshShape(ExecutableImage *executable) {
  for (const mlir::tt::sharding_utils::MeshSharding &input_sharding :
       executable->m_input_sharding) {
    if (input_sharding.getShardType() ==
        mlir::tt::ttcore::MeshShardType::Devices) {
      executable->m_devices_mesh_shape =
          std::vector<std::uint32_t>(input_sharding.getMeshShape().begin(),
                                     input_sharding.getMeshShape().end());
      return;
    }
  }

  // Assuming single device if there are no inputs sharded on device.
  executable->m_devices_mesh_shape = {1, 1};
}

void ModuleBuilder::collectNumDevicesToUtilize(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    ExecutableImage *executable) {
  auto num_partitions_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_partitions");
  // Assuming one partition by default.
  executable->m_num_partitions = 1;
  if (num_partitions_attr) {
    executable->m_num_partitions =
        static_cast<size_t>(num_partitions_attr.getInt());
  } else {
    DLOG_F(WARNING,
           "`mhlo.num_partitions` attribute not found, assuming default number "
           "of partitions: %zu",
           executable->m_num_partitions);
  }

  auto num_replicas_attr =
      mlir_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "mhlo.num_replicas");
  // Assuming one replica by default.
  executable->m_num_replicas = 1;
  if (num_replicas_attr) {
    executable->m_num_replicas =
        static_cast<size_t>(num_replicas_attr.getInt());
  } else {
    DLOG_F(WARNING,
           "`mhlo.num_replicas` attribute not found, assuming default number "
           "of replicas: %zu",
           executable->m_num_replicas);
  }

  if (!num_partitions_attr && !num_replicas_attr) {
    // When both mhlo.num_partitions and mhlo.num_replicas are not populated
    // (torch_xla doesn't populate them), we estimate the number of devices from
    // the mesh shape.
    DLOG_F(WARNING, "Num replicas and num partitions are not set, inferring "
                    "the number of devices from mesh shape");
    executable->m_num_devices_to_utilize = std::accumulate(
        executable->m_devices_mesh_shape.begin(),
        executable->m_devices_mesh_shape.end(), 1, std::multiplies<>());
  } else {
    // If at least one mhlo parameter is populated we assume the default value
    // of the other one.
    executable->m_num_devices_to_utilize =
        executable->m_num_partitions * executable->m_num_replicas;
  }
}

tt_pjrt_status ModuleBuilder::convertFromTTIRToTTNN(
    const std::string &system_descriptor_path,
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const CompileOptions &compile_options, ExecutableImage *executable) {
  mlir::PassManager ttir_to_ttnn_pm(mlir_module.get()->getName());

  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;

  options.optimizerPassEnabled = compile_options.enable_optimizer;
  options.memoryLayoutAnalysisEnabled = compile_options.enable_optimizer;
  options.enableBfp8Conversion = compile_options.enable_bfp8_conversion;
  options.systemDescPath = system_descriptor_path.data();

  if (executable->m_devices_mesh_shape.size() != 2) {
    DLOG_F(ERROR,
           "Invalid mesh shape size: %zu. Shape must have two dimensions!",
           executable->m_devices_mesh_shape.size());
    return tt_pjrt_status::kInternal;
  }

  options.meshShape = {executable->m_devices_mesh_shape[0],
                       executable->m_devices_mesh_shape[1]};
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(ttir_to_ttnn_pm, options);

  enableVerboseIRPrinting(ttir_to_ttnn_pm);

  // Run the pass manager.
  if (mlir::failed(ttir_to_ttnn_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from TTIR to TTNN module");
    return tt_pjrt_status::kInternal;
  }

  executable->m_ttnn_mlir = getMlirCode(mlir_module);

  DLOG_F(LOG_DEBUG, "TTNN Module:");
  printModule(mlir_module);

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::createFlatbufferBinary(
    const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    FlatbufferExecutableImage *executable) {
  executable->m_flatbuffer_binary =
      mlir::tt::ttnn::ttnnToFlatbuffer(mlir_module.get());

  return verifyCreatedFlatbufferBinary(executable);
}

tt_pjrt_status ModuleBuilder::verifyCreatedFlatbufferBinary(
    FlatbufferExecutableImage *executable) {
  if (executable->m_flatbuffer_binary.handle == nullptr) {
    DLOG_F(ERROR, "Failed to generate flatbuffer binary");
    return tt_pjrt_status::kInternal;
  }

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;
  size_t num_inputs =
      executable->m_flatbuffer_binary.getProgramInputs(program_index).size();
  std::vector<tt::runtime::TensorDesc> output_specs =
      executable->m_flatbuffer_binary.getProgramOutputs(program_index);
  size_t num_outputs = output_specs.size();

  if (num_inputs != executable->m_input_sharding.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of inputs %zu "
           "than expected from the m_input_shardings %zu",
           num_inputs, executable->m_input_sharding.size());
    return tt_pjrt_status::kInternal;
  }

  if (num_outputs != executable->m_output_sharding.size()) {
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of outputs %zu "
           "than expected from the m_output_shardings %zu",
           num_outputs, executable->m_output_sharding.size());
    return tt_pjrt_status::kInternal;
  }

  return checkOutputShardingShapes(output_specs, executable);
}

tt_pjrt_status ModuleBuilder::checkOutputShardingShapes(
    const std::vector<tt::runtime::TensorDesc> &output_specs,
    ExecutableImage *executable) {
  for (size_t output_index = 0; output_index < output_specs.size();
       ++output_index) {
    const mlir::tt::sharding_utils::MeshSharding &output_sharding =
        executable->m_output_sharding[output_index];
    if (output_sharding.getShardType() ==
            mlir::tt::ttcore::MeshShardType::Identity ||
        output_sharding.getShardType() ==
            mlir::tt::ttcore::MeshShardType::Replicate) {
      continue;
    }

    const llvm::SmallVector<int64_t> shard_shape =
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
    const mlir::OwningOpRef<mlir::ModuleOp> &module,
    ExecutableImage *executable) {
  llvm::StringMap<llvm::SmallVector<mlir::tt::ttcore::ArgumentType>>
      argTypesMap;

  if (executable->m_input_argument_roles.empty()) {
    return argTypesMap;
  }

  std::vector<mlir::func::FuncOp> publicFuncOps = getPublicFuncOps(module);
  size_t arg_offset = 0;

  for (mlir::func::FuncOp &func_op : publicFuncOps) {
    llvm::SmallVector<mlir::tt::ttcore::ArgumentType> argTypes;
    for (unsigned int i = 0; i < func_op.getNumArguments(); ++i) {
      assert(arg_offset + i < executable->m_input_argument_roles.size() &&
             "TTIR module should have the same number of input arguments as "
             "the SHLO module");
      if (executable->m_input_argument_roles[arg_offset + i] ==
          tt::pjrt::InputArgumentRole::kWeight) {
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

std::string ModuleBuilder::findTTAlchemistLibraryPath() {
  // HACK!

  // Option 1: Use VIRTUAL_ENV if available
  if (const char *venv = std::getenv("VIRTUAL_ENV")) {
    std::string venv_path(venv);
    // We can't assume it will be a python3.11 venv
    for (const auto &entry :
         std::filesystem::directory_iterator(venv_path + "/lib")) {
      if (entry.is_directory() &&
          entry.path().filename().string().find("python") == 0) {
        std::string python_dir_path =
            entry.path().string() +
            "/site-packages/tt_alchemist/lib/libtt-alchemist-lib.so";
        if (std::filesystem::exists(python_dir_path)) {
          return python_dir_path;
        }
      }
    }
  }

  return ""; // Not found
}

void ModuleBuilder::loadTTAlchemistFunctions() {
  std::string so_path = findTTAlchemistLibraryPath();
  if (so_path.empty()) {
    DLOG_F(WARNING, "tt-alchemist library not found in Python environment");
    m_alchemist_available = false;
  }

  dlerror(); // Clear any existing error
  m_tt_alchemist_handle = dlopen(so_path.c_str(), RTLD_LAZY);
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    return;
  }

  m_tt_alchemist_get_instance = (void *(*)())dlsym(
      m_tt_alchemist_handle, "tt_alchemist_TTAlchemist_getInstance");

  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    return;
  }

  m_tt_alchemist_generate_python =
      (bool (*)(void *, const char *, const char *, bool, const char *))dlsym(
          m_tt_alchemist_handle, "tt_alchemist_TTAlchemist_generatePython");

  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    return;
  }

  m_tt_alchemist_generate_cpp =
      (bool (*)(void *, const char *, const char *, bool, const char *))dlsym(
          m_tt_alchemist_handle, "tt_alchemist_TTAlchemist_generateCpp");

  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    return;
  }

  m_alchemist_available = true;
}

} // namespace tt::pjrt::module_builder
