// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_

// c++ standard library includes
#include <memory>
#include <string>
#include <tuple>

// llvm mlir includes
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// shardy includes
#include "shardy/dialect/sdy/ir/dialect.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

// tt-xla includes
#include "common/pjrt_implementation/input_argument_role.h"
#include "common/status.h"
#include "compile_options.h"

// Forward declarations
namespace tt::pjrt {
class ExecutableImage;
class FlatbufferExecutableImage;
class SOExecutableImage;
} // namespace tt::pjrt

namespace tt::pjrt::module_builder {

// MLIR program format name. This would ideally be defined in PJRT API header.
extern const std::string c_mlir_format_name;

class ModuleBuilder {
public:
  ModuleBuilder();
  ~ModuleBuilder();

  // Compiles given mlir module code and produces flatbuffer to execute on a
  // given system.
  std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>> buildModule(
      const std::string_view &mlir_code,
      const std::string &system_descriptor_path,
      const std::unordered_map<std::string, std::string> &compile_options);

private:
  // Logic for buildModule that is common to both the flatbuffer and codegen
  // paths
  std::tuple<tt_pjrt_status, mlir::OwningOpRef<mlir::ModuleOp>>
  buildCommon(const std::string_view &mlir_code, ExecutableImage *executable);

  // Logic for buildModule that is specific to the flatbuffer backend
  tt_pjrt_status buildFlatbuffer(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                                 const std::string &system_descriptor_path,
                                 FlatbufferExecutableImage *executable);

  // Logic for buildModule that is specific to the codegen backend for C++
  tt_pjrt_status
  buildForCodegenCpp(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                     SOExecutableImage *executable);

  // Logic for buildModule that is specific to the codegen backend for Python
  tt_pjrt_status
  buildForCodegenPy(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                    SOExecutableImage *executable);

  // Creates VHLO module from the input program code.
  std::tuple<tt_pjrt_status, mlir::OwningOpRef<mlir::ModuleOp>>
  createVHLOModule(const std::string_view &code);

  // Converts VHLO module to StableHLO module.
  tt_pjrt_status
  convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Runs frontend specific SHLO pipeline on the MLIR module.
  tt_pjrt_status
  runFrontendSHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collects the information about output types.
  void collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                          ExecutableImage *executable);

  // Collects the number of input and output arguments from the VHLO module.
  void collectNumArguments(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                           ExecutableImage *executable);

  // Collects the information about the sharding of specific inputs.
  tt_pjrt_status
  collectInputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                        ExecutableImage *executable);

  // Collects the information about the sharding of specific outputs.
  tt_pjrt_status
  collectOutputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                         ExecutableImage *executable);

  // Collects the information about input argument roles (weight vs input).
  void
  collectInputArgumentRoles(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                            ExecutableImage *executable);

  // Runs compiler StableHLO pipeline on the MLIR module.
  tt_pjrt_status
  runCompilerStableHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts StableHLO module to TTIR module.
  tt_pjrt_status
  convertFromSHLOToTTIR(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                        ExecutableImage *executable);

  // Collects the information about the mesh shape the module is intended to run
  // on.
  void collectMeshShape(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                        ExecutableImage *executable);

  // Estimates devices mesh shape from input shardings in case the mesh
  // attribute is not set on the module.
  void estimateMeshShape(ExecutableImage *executable);

  // Gets the number of devices the binary is intended to run on from the VHLO
  // module.
  void
  collectNumDevicesToUtilize(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                             ExecutableImage *executable);

  // Converts TTIR module to TTNN module.
  tt_pjrt_status
  convertFromTTIRToTTNN(const std::string &system_descriptor_path,
                        mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                        const CompileOptions &compile_options,
                        ExecutableImage *executable);

  // Creates flatbuffer binary from the built TTNN module.
  tt_pjrt_status
  createFlatbufferBinary(const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                         FlatbufferExecutableImage *executable);

  // Verifies that creates flatbuffer binary satisfies conditions estimated by
  // the compiler from the input graph.
  tt_pjrt_status
  verifyCreatedFlatbufferBinary(FlatbufferExecutableImage *executable);

  // Checks if the resulting outputs and their shardings are valid.
  tt_pjrt_status checkOutputShardingShapes(
      const std::vector<tt::runtime::TensorDesc> &output_specs,
      ExecutableImage *executable);

  // Prints module to console for debug purposes.
  static void printModule(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Enables IR printing between passes with VERBOSE or higher logger level.
  static void enableVerboseIRPrinting(mlir::PassManager &pm);

  // Checks if a particular type is scalar.
  bool isScalarType(mlir::Type type);

  // Converts a MLIR module into it's textual representation
  std::string getMlirCode(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collect input sharding if we are using GSPMD.
  tt_pjrt_status
  collectInputShardingsGSPMD(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                             ExecutableImage *executable);

  // Collect output sharding if we are using GSPMD.
  tt_pjrt_status
  collectOutputShardingsGSPMD(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                              ExecutableImage *executable);

  // Collect input sharding if we are using Shardy.
  tt_pjrt_status
  collectInputShardingsShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                              ExecutableImage *executable);

  // Collect output sharding if we are using Shardy.
  tt_pjrt_status
  collectOutputShardingsShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                               ExecutableImage *executable);

  // Checks if the StableHLO code is using the Shardy mlir dialect.
  bool isUsingShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Checks if the StableHLO code is using manual computation ops of the Shardy
  // mlir dialect.
  bool isUsingShardyManualComputation(
      const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Takes a vector of string attributes representing GSPMD sharding and fills
  // the vector of tt_mlir Sharding with the appropriate corresponding values.
  mlir::LogicalResult createShardingsFromGSPMD(
      const std::vector<mlir::StringAttr> &gspmd_attributes,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings);

  // Takes a vector of Shardy sharding attributes, the overall Shardy mesh and
  // fills the vector of tt_mlir MeshSharding objects with the appropriate
  // corresponding values.
  mlir::LogicalResult createShardingsFromShardy(
      std::vector<mlir::sdy::TensorShardingAttr> &shardy_attributes,
      const mlir::sdy::MeshAttr &shardy_mesh,
      std::vector<mlir::tt::sharding_utils::MeshSharding> &shardings);

  // Gets all public functions from the module.
  std::vector<mlir::func::FuncOp>
  getPublicFuncOps(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Gets the first sdy.Mesh op of a mlir module with shardy dialect enbaled.
  std::optional<mlir::sdy::MeshOp>
  getFirstShardyMeshOp(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Creates argument type map based on collected input argument roles.
  llvm::StringMap<llvm::SmallVector<mlir::tt::ttcore::ArgumentType>>
  createArgumentTypeMap(const mlir::OwningOpRef<mlir::ModuleOp> &module,
                        ExecutableImage *executable);

  // Finds tt-alchemist library path using environment variables
  std::string findTTAlchemistLibraryPath();

  // Loads tt-alchemist library and function pointers
  void loadTTAlchemistFunctions();

  // MLIR context handle.
  std::unique_ptr<mlir::MLIRContext> m_context;

  // tt-alchemist library handle and function pointers
  void *m_tt_alchemist_handle;
  bool m_alchemist_available;
  void *(*m_tt_alchemist_get_instance)();
  bool (*m_tt_alchemist_generate_python)(void *instance, const char *input_file,
                                         const char *output_dir, bool is_local,
                                         const char *pipeline_options);
  bool (*m_tt_alchemist_generate_cpp)(void *instance, const char *input_file,
                                      const char *output_dir, bool is_local,
                                      const char *pipeline_options);
};

} // namespace tt::pjrt::module_builder

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_
