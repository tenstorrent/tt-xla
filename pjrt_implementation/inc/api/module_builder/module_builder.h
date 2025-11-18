// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_MODULE_BUILDER_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_MODULE_BUILDER_H_

// c++ standard library includes
#include <memory>
#include <optional>
#include <string>
#include <tuple>

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// shardy includes
#include "shardy/dialect/sdy/ir/dialect.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Support/TTPrintIRInstrumentation.h"

// tt-xla includes
#include "api/compile_options.h"
#include "utils/status.h"

namespace tt::pjrt {
class ClientInstance;
class ExecutableImage;
} // namespace tt::pjrt

namespace tt::pjrt::module_builder {

// MLIR program format name. This would ideally be defined in PJRT API header.
extern const std::string c_mlir_format_name;

// Class to hold tt-alchemist library handles and function pointers.
class TTAlchemistHandler {
public:
  // Default constructor leaves the library unitialized.
  TTAlchemistHandler();

  // Destructor closes the handle to .so file.
  ~TTAlchemistHandler();

  // Initializes the tt-alchemist library and function pointers. This function
  // is fallible.
  void initialize();

  // Getter for initialization status.
  bool isInitialized() const { return m_initialized; }

  // Gets the alchemist singleton. The singleton is the first parameter for all
  // alchemist functions.
  void *(*getInstanceFunc() const)() { return m_get_instance; }

  // Gets the function that ingests TTIR and generates a Python solution at the
  // specified path.
  bool (*generatePythonFunc() const)(void *, const char *, const char *, bool,
                                     const char *) {
    return m_generate_python;
  }

  // Gets the function that ingests TTIR and generates a C++ solution at the
  // specified path.
  bool (*generateCppFunc() const)(void *, const char *, const char *, bool,
                                  const char *) {
    return m_generate_cpp;
  }

private:
  // Finds tt-alchemist library path using environment variables
  std::optional<std::string> findTTAlchemistLibraryPath();

  // Initialization status. It is required to be checked before using the
  // library, as initializing the library is fallible.
  bool m_initialized;

  // The handle to the alchemist .so.
  void *m_handle;

  // Function pointer to the get_instance function in the alchemist .so.
  void *(*m_get_instance)();

  // Function pointer to the generate_python function in the alchemist .so.
  bool (*m_generate_python)(void *instance, const char *input_file,
                            const char *output_dir, bool is_local,
                            const char *pipeline_options);

  // Function pointer to the generate_cpp function in the alchemist .so.
  bool (*m_generate_cpp)(void *instance, const char *input_file,
                         const char *output_dir, bool is_local,
                         const char *pipeline_options);
};

struct NumArgumentsResult {
  size_t num_inputs;
  size_t num_outputs;
  std::vector<std::vector<std::uint32_t>> output_dimensions;
  std::vector<size_t> output_ranks;
  std::vector<std::int64_t> output_dimensions_flat;
};

struct NumDevicesResult {
  size_t num_partitions;
  size_t num_replicas;
  size_t num_devices_to_utilize;
};

class ModuleBuilder {
public:
  ModuleBuilder();

  // Compiles given mlir module code and returns produced executable image
  // for execution on a given system, together with the compilation status
  // for error checking.
  std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>> buildModule(
      const std::string_view &mlir_code,
      const std::string &system_descriptor_path,
      const std::unordered_map<std::string, std::string> &compile_options,
      tt::pjrt::ClientInstance *client_instance);

  // Gets the first sdy.Mesh op of a mlir module with shardy dialect enbaled.
  // Could be used to extract mesh attribute from the module so we can use it as
  // a utility function.
  static std::optional<mlir::sdy::MeshOp>
  getFirstShardyMeshOp(const mlir::OwningOpRef<mlir::ModuleOp> &module);

private:
  // Creates VHLO module from the input program code.
  tt_pjrt_status
  createVHLOModule(const std::string_view &code,
                   mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                   const std::optional<std::string> &export_path);

  // Converts VHLO module to StableHLO module.
  tt_pjrt_status
  convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                        const std::optional<std::string> &export_path);

  // Runs frontend specific SHLO pipeline on the MLIR module.
  tt_pjrt_status
  runFrontendSHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                          const std::optional<std::string> &export_path);

  // Collects the information about output types.
  static std::vector<PJRT_Buffer_Type>
  collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the number of input and output arguments from the VHLO module.
  static tt_pjrt_status
  collectNumArguments(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                      NumArgumentsResult &result);

  // Collects the information about the sharding of specific inputs.
  tt_pjrt_status collectInputShardings(
      const mlir::OwningOpRef<mlir::ModuleOp> &module,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings);

  // Collects the information about the sharding of specific outputs.
  tt_pjrt_status collectOutputShardings(
      const mlir::OwningOpRef<mlir::ModuleOp> &module,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings);

  // Runs compiler StableHLO pipeline on the MLIR module.
  tt_pjrt_status
  runCompilerStableHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                               const std::optional<std::string> &export_path);

  // Converts StableHLO module to TTIR module.
  tt_pjrt_status
  convertFromSHLOToTTIR(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                        std::string &ttir_code,
                        const std::optional<std::string> &export_path);

  // Collects the information about the mesh shape the module is intended to run
  // on.
  static std::vector<std::uint32_t> collectMeshShape(
      const mlir::OwningOpRef<mlir::ModuleOp> &module,
      std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings);

  // Estimates devices mesh shape from input shardings in case the mesh
  // attribute is not set on the module.
  static std::vector<std::uint32_t> estimateMeshShape(
      std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings);

  // Gets the number of devices the binary is intended to run on from the VHLO
  // module.
  static NumDevicesResult
  collectNumDevicesToUtilize(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                             std::vector<std::uint32_t> devices_mesh_shape);

  // Converts TTIR module to TTNN module.
  tt_pjrt_status convertFromTTIRToTTNN(
      const std::string &system_descriptor_path,
      mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
      const CompileOptions &compile_options, ClientInstance *client_instance,
      std::vector<std::uint32_t> devices_mesh_shape, std::string &ttnn_code);

  // Creates flatbuffer binary from the built TTNN module.
  tt_pjrt_status createFlatbufferBinary(
      const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &input_shardings,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings,
      tt::runtime::Binary &flatbuffer_binary);

  // Verifies that creates flatbuffer binary satisfies conditions estimated by
  // the compiler from the input graph.
  tt_pjrt_status verifyCreatedFlatbufferBinary(
      const tt::runtime::Binary &flatbuffer_binary,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &input_shardings,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings);

  // Checks if the resulting outputs and their shardings are valid.
  static tt_pjrt_status checkOutputShardingShapes(
      const std::vector<tt::runtime::TensorDesc> &output_specs,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings);

  // Prints module to console for debug purposes.
  // If export_path is set, also dumps the IR to disk with the given stage name.
  static void printModule(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                          const std::optional<std::string> &export_path,
                          const std::string &stage_name);

  // Enables IR printing between passes with VERBOSE or higher logger level.
  static void enableVerboseIRPrinting(mlir::PassManager &pm,
                                      const std::string &source_name,
                                      const std::string &pipeline_name,
                                      bool dump_initial = false);

  // Parse EXPLORER_EXPORT_LEVEL keyword to determine dump level.
  static mlir::tt::TTPrintIRInstrumentation::DumpLevel
  parseExplorerDumpLevel(const char *level_str);

  // Extract source file name from MLIR module location information.
  std::string
  extractSourceName(const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) const;

  // Checks if a particular type is scalar.
  static bool isScalarType(mlir::Type type);

  // Converts a MLIR module into it's textual representation
  static std::string
  getMlirCode(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collect input sharding if we are using GSPMD.
  tt_pjrt_status collectInputShardingsGSPMD(
      const mlir::OwningOpRef<mlir::ModuleOp> &module,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &input_shardings);

  // Collect output sharding if we are using GSPMD.
  tt_pjrt_status collectOutputShardingsGSPMD(
      const mlir::OwningOpRef<mlir::ModuleOp> &module,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &output_shardings);

  // Collect input sharding if we are using Shardy.
  std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
  collectInputShardingsShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collect output sharding if we are using Shardy.
  std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
  collectOutputShardingsShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Checks if the StableHLO code is using the Shardy mlir dialect.
  bool isUsingShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Checks if the StableHLO code is using manual computation ops of the Shardy
  // mlir dialect.
  bool isUsingShardyManualComputation(
      const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Takes a vector of string attributes representing GSPMD sharding and fills
  // the vector of tt_mlir Sharding with the appropriate corresponding values.
  static mlir::LogicalResult createShardingsFromGSPMD(
      const std::vector<mlir::StringAttr> &gspmd_attributes,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings);

  // Takes a vector of Shardy sharding attributes, the overall Shardy mesh and
  // fills the vector of tt_mlir MeshSharding objects with the appropriate
  // corresponding values.
  static mlir::LogicalResult createShardingsFromShardy(
      std::vector<mlir::sdy::TensorShardingAttr> &shardy_attributes,
      const mlir::sdy::MeshAttr &shardy_mesh,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings);

  // Collects memory kinds for output buffers.
  static void collectMemoryKinds(size_t num_outputs,
                                 std::vector<const char *> &memory_kinds,
                                 std::vector<size_t> &memory_kind_sizes);

  // Gets all public functions from the module.
  static std::vector<mlir::func::FuncOp>
  getPublicFuncOps(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Builds module for TTNN Flatbuffer backend runtime.
  std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
  buildModuleForTTNNRuntime(
      mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
      std::string &&original_mlir_code, std::string &&ttir_mlir,
      std::string &&ttnn_mlir, std::string &&executable_name,
      NumArgumentsResult &&num_arguments,
      const NumDevicesResult &num_devices_result,
      const std::vector<std::uint32_t> &mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &input_shardings,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings,
      const std::vector<PJRT_Buffer_Type> &output_types,
      std::vector<const char *> &&output_memory_kinds,
      std::vector<size_t> &&output_memory_kinds_sizes,
      CompileOptions &&compile_options);

  // Builds module for TTNN Codegen C++ backend runtime.
  std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>>
  buildModuleForTTNNCodegen(
      mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
      std::string &&original_mlir_code, std::string &&ttir_mlir,
      std::string &&ttnn_mlir, std::string &&executable_name,
      NumArgumentsResult &&num_arguments,
      const NumDevicesResult &num_devices_result,
      const std::vector<std::uint32_t> &mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &input_shardings,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings,
      const std::vector<PJRT_Buffer_Type> &output_types,
      std::vector<const char *> &&output_memory_kinds,
      std::vector<size_t> &&output_memory_kinds_sizes,
      CompileOptions &&compile_options);

  // Invokes tt-alchemist to generate a ready-to-run solution (C++ or Python)
  // independently of the frontend. In the future, this will also prepare
  // everything to generate an .so file for execution.
  tt_pjrt_status performCodegen(std::string_view ttnn_mlir,
                                const CompileOptions &compile_options);

  // MLIR context handle.
  std::unique_ptr<mlir::MLIRContext> m_context;

  // tt-alchemist library handler.
  TTAlchemistHandler m_tt_alchemist_handler;
};

} // namespace tt::pjrt::module_builder

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_MODULE_BUILDER_MODULE_BUILDER_H_
