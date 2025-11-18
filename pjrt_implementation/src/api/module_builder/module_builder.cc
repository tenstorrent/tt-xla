// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/module_builder/module_builder.h"

// c++ standard library includes
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>

// POSIX includes
#include <dlfcn.h>

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
#include "mlir/IR/BuiltinOps.h"
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
#include "ttmlir/Target/Python/PythonEmitter.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

// tt-xla includes
#include "api/client_instance.h"
#include "api/compile_options.h"
#include "api/executable_image.h"
#include "api/memory_instance.h"
#include "api/module_builder/frontend_passes/shlo_input_role_propagation.h"
#include "api/module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.h"
#include "utils/data_type_utils.h"
#include "utils/logging.h"

namespace tt::pjrt::module_builder {

const std::string c_mlir_format_name = "mlir";

// Helper function to get current timestamp in milliseconds.
static std::string getCurrentTimeStamp() {
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
  return std::to_string(ms);
}

// TTAlchemistHandler implementation

TTAlchemistHandler::TTAlchemistHandler()
    : m_initialized(false), m_handle(nullptr), m_get_instance(nullptr),
      m_generate_python(nullptr), m_generate_cpp(nullptr) {}

TTAlchemistHandler::~TTAlchemistHandler() {
  if (m_handle != nullptr) {
    dlclose(m_handle);
  }
}

std::optional<std::string> TTAlchemistHandler::findTTAlchemistLibraryPath() {
  // HACK: Currently tt-alchemist is packaged as a python package that contains
  // an .so file. We rely on python venv activate script always exporting
  // VIRTUAL_ENV environment variable to find the .so file. Long term, this code
  // should be made redundant once we think of a better way to package
  // tt-alchemist. Packaging tracked at:
  // https://github.com/tenstorrent/tt-mlir/issues/5250

  const char *venv_cstr = std::getenv("VIRTUAL_ENV");
  if (venv_cstr == nullptr) {
    return std::nullopt;
  }
  std::string venv(venv_cstr);
  if (venv.empty()) {
    return std::nullopt;
  }

  // We can't assume it will be a python3.11 venv.
  for (const auto &entry : std::filesystem::directory_iterator(venv + "/lib")) {
    if (entry.is_directory() &&
        entry.path().filename().string().find("python") == std::string::npos) {
      continue;
    }

    std::string python_dir_path =
        entry.path().string() +
        "/site-packages/tt_alchemist/lib/libtt-alchemist-lib.so";
    if (std::filesystem::exists(python_dir_path)) {
      return python_dir_path;
    }
  }

  return std::nullopt;
}

void TTAlchemistHandler::initialize() {
  std::optional<std::string> maybe_so_path = findTTAlchemistLibraryPath();
  if (!maybe_so_path.has_value()) {
    DLOG_F(WARNING, "tt-alchemist library not found in Python environment");
    return;
  }
  std::string so_path = maybe_so_path.value();

  dlerror(); // Clear any existing error
  m_handle = dlopen(so_path.c_str(), RTLD_LAZY);
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    return;
  }

  m_get_instance =
      (void *(*)())dlsym(m_handle, "tt_alchemist_TTAlchemist_getInstance");

  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_generate_python =
      (bool (*)(void *, const char *, const char *, bool, const char *))dlsym(
          m_handle, "tt_alchemist_TTAlchemist_generatePython");

  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_generate_cpp =
      (bool (*)(void *, const char *, const char *, bool, const char *))dlsym(
          m_handle, "tt_alchemist_TTAlchemist_generateCpp");

  dlsym_error = dlerror();
  if (dlsym_error) {
    DLOG_F(WARNING, "dlsym error while loading tt-alchemist library: %s",
           dlsym_error);
    dlclose(m_handle);
    m_handle = nullptr;
    return;
  }

  m_initialized = true;
}

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

  // Try to load tt-alchemist library and function pointers.
  m_tt_alchemist_handler.initialize();
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
  status =
      createVHLOModule(mlir_code, mlir_module, compile_options.export_path);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "0_CreateVHLOModule");

  std::string original_mlir_code(mlir_code);

  status = convertFromVHLOToSHLO(mlir_module, compile_options.export_path);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "1_VHLOToSHLO");

  status = runFrontendSHLOPipeline(mlir_module, compile_options.export_path);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "2_FrontendSHLOPipeline");

  std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings;
  status = collectInputShardings(mlir_module, input_shardings);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "3_InputShardings");

  std::vector<mlir::tt::sharding_utils::MeshSharding> output_shardings;
  status = collectOutputShardings(mlir_module, output_shardings);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "4_OutputShardings");

  NumArgumentsResult num_arguments;
  status = collectNumArguments(mlir_module, num_arguments);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "5_NumArguments");

  std::vector<PJRT_Buffer_Type> output_types = collectOutputTypes(mlir_module);

  status =
      runCompilerStableHLOPipeline(mlir_module, compile_options.export_path);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "6_CompilerStableHLOPipeline");

  LOG_BRINGUP_STAGE("TTMLIR_COMPILATION_START");
  std::string ttir_mlir;
  status = convertFromSHLOToTTIR(mlir_module, ttir_mlir,
                                 compile_options.export_path);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "7_ConvertFromSHLOToTTIR");

  std::vector<std::uint32_t> mesh_shape =
      collectMeshShape(mlir_module, input_shardings);

  NumDevicesResult num_devices_result =
      collectNumDevicesToUtilize(mlir_module, mesh_shape);

  // Collect memory kinds for output buffers
  std::vector<const char *> output_memory_kinds;
  std::vector<size_t> output_memory_kinds_sizes;
  collectMemoryKinds(num_arguments.num_outputs, output_memory_kinds,
                     output_memory_kinds_sizes);

  std::string ttnn_mlir;
  status = convertFromTTIRToTTNN(system_descriptor_path, mlir_module,
                                 compile_options, client_instance, mesh_shape,
                                 ttnn_mlir);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }
  printModule(mlir_module, ".", "8_ConvertFromTTIRToTTNN");
  // TODO(mrakita): Use the VHLO module name from the module builder, if it has
  // a name, otherwise some default string like the current one.
  std::string executable_name = "tt_executable";

  if (compile_options.backend == BackendRuntime::TTNNFlatbuffer) {
    return buildModuleForTTNNRuntime(
        mlir_module, std::move(original_mlir_code), std::move(ttir_mlir),
        std::move(ttnn_mlir), std::move(executable_name),
        std::move(num_arguments), num_devices_result, mesh_shape,
        input_shardings, output_shardings, output_types,
        std::move(output_memory_kinds), std::move(output_memory_kinds_sizes),
        std::move(compile_options));
  } else if (compile_options.backend == BackendRuntime::TTNNCodegenCpp ||
             compile_options.backend == BackendRuntime::TTNNCodegenPy) {
    return buildModuleForTTNNCodegen(
        mlir_module, std::move(original_mlir_code), std::move(ttir_mlir),
        std::move(ttnn_mlir), std::move(executable_name),
        std::move(num_arguments), num_devices_result, mesh_shape,
        input_shardings, output_shardings, output_types,
        std::move(output_memory_kinds), std::move(output_memory_kinds_sizes),
        std::move(compile_options));
  }

  DLOG_F(ERROR, "Unsupported backend option");
  return {tt_pjrt_status::kInvalidArgument, nullptr};
}

tt_pjrt_status
ModuleBuilder::createVHLOModule(const std::string_view &mlir_code,
                                mlir::OwningOpRef<mlir::ModuleOp> &vhlo_module,
                                const std::optional<std::string> &export_path) {
  vhlo_module = mlir::parseSourceString<mlir::ModuleOp>(
      llvm::StringRef(mlir_code.data(), mlir_code.size()),
      mlir::ParserConfig{m_context.get(), /*verifyAfterParse=*/true});

  if (!vhlo_module) {
    DLOG_F(ERROR, "Failed to create VHLO module from the input program code");
    return tt_pjrt_status::kInternal;
  }

  printModule(vhlo_module, export_path, "vhlo");

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::convertFromVHLOToSHLO(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const std::optional<std::string> &export_path) {
  mlir::PassManager vhlo_to_shlo_pm(mlir_module.get()->getName());

  mlir::stablehlo::createStablehloDeserializePipeline(vhlo_to_shlo_pm);

  enableVerboseIRPrinting(vhlo_to_shlo_pm);

  if (mlir::failed(vhlo_to_shlo_pm.run(mlir_module.get()))) {
    DLOG_F(ERROR, "Failed to convert from VHLO to SHLO module");
    return tt_pjrt_status::kInternal;
  }

  printModule(mlir_module, export_path, "shlo");

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::runFrontendSHLOPipeline(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const std::optional<std::string> &export_path) {

  tt_pjrt_status status =
      frontend_passes::annotateArgumentAttributes(mlir_module);

  printModule(mlir_module, export_path, "shlo_frontend");

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
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    const std::optional<std::string> &export_path) {
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

  printModule(mlir_module, export_path, "shlo_compiler");

  if (!tt_pjrt_status_is_ok(
          frontend_passes::setProperSdyMeshAttributeInSpmdMode(mlir_module))) {
    DLOG_F(ERROR, "Failed to set proper sdy.mesh attribute in SPMD mode");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ModuleBuilder::convertFromSHLOToTTIR(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module, std::string &ttir_mlir,
    const std::optional<std::string> &export_path) {
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

  printModule(mlir_module, export_path, "ttir");

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
    std::vector<std::uint32_t> devices_mesh_shape, std::string &ttnn_mlir) {
  mlir::PassManager ttir_to_ttnn_pm(mlir_module.get()->getName());

  mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions options;

  // Optimizer passes are not supported in distributed runtime.
  if (tt::runtime::getCurrentHostRuntime() ==
          tt::runtime::HostRuntime::Distributed &&
      compile_options.optimization_level > 0) {
    DLOG_F(ERROR, "Optimizer passes are not supported in distributed runtime");
    return tt_pjrt_status::kInternal;
  }

  options.optimizationLevel = compile_options.optimization_level;
  options.enableBfp8Conversion = compile_options.enable_bfp8_conversion;
  options.enableTrace = compile_options.enable_trace;
  options.systemDescPath = system_descriptor_path.data();
  options.enableConstEval = compile_options.enable_const_eval;

  if (devices_mesh_shape.size() != 2) {
    DLOG_F(ERROR,
           "Invalid mesh shape size: %zu. Shape must have two dimensions!",
           devices_mesh_shape.size());
    return tt_pjrt_status::kInternal;
  }

  options.meshShape = {devices_mesh_shape[0], devices_mesh_shape[1]};

  // Use the `options.devicePtr` to pass the device pointer to the optimizer in
  // order to avoid closing and reopening the device afterwards.
  // Optimizer is enabled for optimization_level >= 1
  if (compile_options.optimization_level >= 1) {
    tt::runtime::Device submesh_for_optim =
        client_instance->getOrCreateOptimizerSubmesh(devices_mesh_shape);
    options.devicePtr =
        std::static_pointer_cast<tt::tt_metal::distributed::MeshDevice>(
            submesh_for_optim.handle);
  }
  mlir::tt::ttnn::createTTIRToTTNNBackendPipeline(ttir_to_ttnn_pm, options);

  enableVerboseIRPrinting(ttir_to_ttnn_pm);

  // Run the pass manager.
  mlir::LogicalResult mlir_result = ttir_to_ttnn_pm.run(mlir_module.get());

  // Close the optimizer submesh now that the compilation is complete.
  client_instance->closeOptimizerSubmesh();

  if (mlir::failed(mlir_result)) {
    DLOG_F(ERROR, "Failed to convert from TTIR to TTNN module");
    return tt_pjrt_status::kInternal;
  }

  ttnn_mlir = getMlirCode(mlir_module);

  printModule(mlir_module, compile_options.export_path, "ttnn");

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

void ModuleBuilder::printModule(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                                const std::optional<std::string> &export_path,
                                const std::string &stage_name) {
  if (loguru::g_stderr_verbosity >= LOG_DEBUG) {
    VLOG_F(LOG_DEBUG, "MLIR Module %s:", stage_name.c_str());
    mlir_module->print(llvm::errs(), mlir::OpPrintingFlags().enableDebugInfo());
    llvm::errs()
        << "------------------ END OF MLIR MODULE ------------------\n";
  }

  if (!export_path.has_value()) {
    return;
  }

  std::filesystem::path ir_dump_dir =
      std::filesystem::path(export_path.value()) / "irs";
  std::filesystem::create_directories(ir_dump_dir);

  std::string filename = stage_name + "_" + getCurrentTimeStamp() + ".mlir";
  std::filesystem::path ir_file_path = ir_dump_dir / filename;

  std::error_code err_code;
  llvm::raw_fd_ostream out_stream(ir_file_path.string(), err_code);
  mlir_module->print(out_stream, mlir::OpPrintingFlags().enableDebugInfo());
  out_stream.close();
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

std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
ModuleBuilder::buildModuleForTTNNRuntime(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    std::string &&original_mlir_code, std::string &&ttir_mlir,
    std::string &&ttnn_mlir, std::string &&executable_name,
    NumArgumentsResult &&num_arguments,
    const NumDevicesResult &num_devices_result,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings,
    const std::vector<PJRT_Buffer_Type> &output_types,
    std::vector<const char *> &&output_memory_kinds,
    std::vector<size_t> &&output_memory_kinds_sizes,
    CompileOptions &&compile_options) {
  tt::runtime::Binary flatbuffer(nullptr);
  tt_pjrt_status status = createFlatbufferBinary(mlir_module, input_shardings,
                                                 output_shardings, flatbuffer);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  if (compile_options.export_path.has_value()) {
    std::string filename = "fb_" + getCurrentTimeStamp() + ".ttnn";
    std::filesystem::path output_path =
        std::filesystem::path(compile_options.export_path.value()) / filename;
    flatbuffer.store(output_path.string().c_str());
  }

  auto executable_image = FlatbufferExecutableImage::createInstance(
      flatbuffer, std::move(original_mlir_code), std::move(ttir_mlir),
      std::move(ttnn_mlir), std::move(executable_name),
      num_arguments.num_inputs, num_arguments.num_outputs,
      std::move(num_arguments.output_dimensions),
      std::move(num_arguments.output_ranks),
      std::move(num_arguments.output_dimensions_flat),
      num_devices_result.num_partitions, num_devices_result.num_replicas,
      num_devices_result.num_devices_to_utilize, mesh_shape, input_shardings,
      output_shardings, output_types, std::move(output_memory_kinds),
      std::move(output_memory_kinds_sizes), std::move(compile_options));
  return {tt_pjrt_status::kSuccess, executable_image};
}

std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
ModuleBuilder::buildModuleForTTNNCodegen(
    mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
    std::string &&original_mlir_code, std::string &&ttir_mlir,
    std::string &&ttnn_mlir, std::string &&executable_name,
    NumArgumentsResult &&num_arguments,
    const NumDevicesResult &num_devices_result,
    const std::vector<std::uint32_t> &mesh_shape,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings,
    const std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings,
    const std::vector<PJRT_Buffer_Type> &output_types,
    std::vector<const char *> &&output_memory_kinds,
    std::vector<size_t> &&output_memory_kinds_sizes,
    CompileOptions &&compile_options) {
  tt_pjrt_status status = performCodegen(ttnn_mlir, compile_options);
  if (!tt_pjrt_status_is_ok(status)) {
    return {status, nullptr};
  }

  auto executable_image = SOExecutableImage::createInstance(
      std::move(original_mlir_code), std::move(ttir_mlir), std::move(ttnn_mlir),
      std::move(executable_name), num_arguments.num_inputs,
      num_arguments.num_outputs, std::move(num_arguments.output_dimensions),
      std::move(num_arguments.output_ranks),
      std::move(num_arguments.output_dimensions_flat),
      num_devices_result.num_partitions, num_devices_result.num_replicas,
      num_devices_result.num_devices_to_utilize, mesh_shape, input_shardings,
      output_shardings, output_types, std::move(output_memory_kinds),
      std::move(output_memory_kinds_sizes), std::move(compile_options));
  return {tt_pjrt_status::kSuccess, executable_image};
}

tt_pjrt_status
ModuleBuilder::performCodegen(std::string_view ttnn_mlir,
                              const CompileOptions &compile_options) {
  assert(compile_options.export_path.has_value() &&
         "export_path compile option is not set.");

  if (!m_tt_alchemist_handler.isInitialized()) {
    DLOG_F(ERROR, "tt-alchemist library or functions not available");
    return tt_pjrt_status::kInternal;
  }

  std::string folder = compile_options.export_path.value();
  std::filesystem::create_directories(folder);

  std::ofstream ttnn_file(folder + "/ttnn.mlir");
  ttnn_file << ttnn_mlir;
  ttnn_file.close();

  void *instance = m_tt_alchemist_handler.getInstanceFunc()();
  if (!instance) {
    DLOG_F(ERROR, "Failed to get tt-alchemist instance");
    return tt_pjrt_status::kInternal;
  }

  std::string input_file = folder + "/ttnn.mlir";
  // Controls wether the generated solution is designed
  // for standalone execution(is_local=false)
  // or for execution within an existing development environment that already
  // has prerequisites installed(is_local=true).
  bool is_local = false;
  // Alchemist specific options are passed here.
  // Other options are ingested during TTIR->TTNN conversion.
  std::string should_load = compile_options.export_tensors ? "true" : "false";
  std::string pipeline_options = "load-input-tensors-from-disk=" + should_load +
                                 " "
                                 "tensor-load-directory='./tensors'";
  bool result;

  if (compile_options.backend == BackendRuntime::TTNNCodegenCpp) {
    result = m_tt_alchemist_handler.generateCppFunc()(
        instance, input_file.c_str(), folder.c_str(), is_local,
        pipeline_options.c_str());
  } else if (compile_options.backend == BackendRuntime::TTNNCodegenPy) {
    // For Python specifically,
    // standalone is currently marked as unsupported, and setting it only
    // results in copying of one extra blank file. As per offline discussion
    // with mlir-core, we should set local to true for Python.
    is_local = true;
    result = m_tt_alchemist_handler.generatePythonFunc()(
        instance, input_file.c_str(), folder.c_str(), is_local,
        pipeline_options.c_str());
  } else {
    assert(false && "Unsupported backend when doing codegen");
  }

  if (!result) {
    DLOG_F(ERROR, "tt-alchemist generatePython failed");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt::module_builder
