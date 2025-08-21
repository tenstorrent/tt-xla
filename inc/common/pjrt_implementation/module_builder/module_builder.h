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
#include "common/pjrt_implementation/executable_image.h"
#include "common/status.h"
#include "compile_options.h"

namespace tt::pjrt::module_builder {

// MLIR program format name. This would ideally be defined in PJRT API header.
extern const std::string c_mlir_format_name;

struct NumArgumentsResult {
  size_t num_inputs;
  size_t num_outputs;
  std::vector<std::vector<std::uint32_t>> m_output_dimensions;
  std::vector<size_t> m_output_ranks;
  std::vector<std::int64_t> m_output_dimensions_flat;
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
      const std::unordered_map<std::string, std::string> &compile_options);

private:
  // Creates VHLO module from the input program code.
  std::optional<mlir::OwningOpRef<mlir::ModuleOp>>
  createVHLOModule(const std::string_view &code);

  // Converts VHLO module to StableHLO module.
  tt_pjrt_status
  convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Runs frontend specific SHLO pipeline on the MLIR module.
  tt_pjrt_status
  runFrontendSHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collects the information about output types.
  std::vector<PJRT_Buffer_Type>
  collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the number of input and output arguments from the VHLO module.
  NumArgumentsResult
  collectNumArguments(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific inputs.
  std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
  collectInputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific outputs.
  std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
  collectOutputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Runs compiler StableHLO pipeline on the MLIR module.
  tt_pjrt_status
  runCompilerStableHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts StableHLO module to TTIR module.
  std::optional<std::string>
  convertFromSHLOToTTIR(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collects the information about the mesh shape the module is intended to run
  // on.
  std::vector<std::uint32_t> collectMeshShape(
      const mlir::OwningOpRef<mlir::ModuleOp> &module,
      std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings);

  // Estimates devices mesh shape from input shardings in case the mesh
  // attribute is not set on the module.
  std::vector<std::uint32_t> estimateMeshShape(
      std::vector<mlir::tt::sharding_utils::MeshSharding> input_shardings);

  // Gets the number of devices the binary is intended to run on from the VHLO
  // module.
  std::tuple<size_t, size_t, size_t>
  collectNumDevicesToUtilize(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                             std::vector<std::uint32_t> devices_mesh_shape);

  // Converts TTIR module to TTNN module.
  std::optional<std::string>
  convertFromTTIRToTTNN(const std::string &system_descriptor_path,
                        mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                        const CompileOptions &compile_options,
                        std::vector<std::uint32_t> devices_mesh_shape);

  // Creates flatbuffer binary from the built TTNN module.
  std::optional<tt::runtime::Binary> createFlatbufferBinary(
      const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &input_shardings,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings);

  // Verifies that creates flatbuffer binary satisfies conditions estimated by
  // the compiler from the input graph.
  tt_pjrt_status verifyCreatedFlatbufferBinary(
      const tt::runtime::Binary &flatbuffer_binary,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &input_shardings,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings);

  // Checks if the resulting outputs and their shardings are valid.
  tt_pjrt_status checkOutputShardingShapes(
      const std::vector<tt::runtime::TensorDesc> &output_specs,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_shardings);

  // Prints module to console for debug purposes.
  static void printModule(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Enables IR printing between passes with VERBOSE or higher logger level.
  static void enableVerboseIRPrinting(mlir::PassManager &pm);

  // Checks if a particular type is scalar.
  bool isScalarType(mlir::Type type);

  // Converts a MLIR module into it's textual representation
  std::string getMlirCode(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collect input sharding if we are using GSPMD.
  std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
  collectInputShardingsGSPMD(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collect output sharding if we are using GSPMD.
  std::optional<std::vector<mlir::tt::sharding_utils::MeshSharding>>
  collectOutputShardingsGSPMD(const mlir::OwningOpRef<mlir::ModuleOp> &module);

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

  // MLIR context handle.
  std::unique_ptr<mlir::MLIRContext> m_context;
};

} // namespace tt::pjrt::module_builder

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_
