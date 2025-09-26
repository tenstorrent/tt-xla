// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_

// c++ standard library includes
#include <memory>
#include <string>

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
#include "common/status.h"
#include "compile_options.h"

namespace tt::pjrt {
class ClientInstance;
}

namespace tt::pjrt::module_builder {

// MLIR program format name. This would ideally be defined in PJRT API header.
extern const std::string c_mlir_format_name;

// Enum to represent the role of input arguments
enum class InputArgumentRole {
  kInput, // Regular input data
  kWeight // Weight/parameter data
};

class ModuleBuilder {
public:
  ModuleBuilder();

  // Compiles given mlir module code and produces flatbuffer to execute on a
  // given system.
  tt_pjrt_status buildModule(
      const std::string_view &mlir_code,
      const std::string &system_descriptor_path,
      const std::unordered_map<std::string, std::string> &compile_options,
      tt::pjrt::ClientInstance *client_instance);

  // Returns compiled flatbuffer binary.
  const tt::runtime::Binary &getFlatbufferBinary() const {
    return m_flatbuffer_binary;
  }

  // Returns TTIR MLIR code.
  const std::string &getTTIRMlirCode() const { return m_ttir_mlir; }

  // Returns TTNN MLIR code.
  const std::string &getTTNNMlirCode() const { return m_ttnn_mlir; }

  // Returns vector of boolean values determining if each output is scalar.
  const std::vector<bool> &getIsOutputScalar() const {
    return m_is_output_scalar;
  };

  // Returns a vector of PJRT_Buffer_Type enums corresponding to the data types
  // of the outputs of the module.
  const std::vector<PJRT_Buffer_Type> &getOutputDataTypes() const {
    return m_output_data_types;
  };

  // Returns number of partitions defined for the program module.
  size_t getNumPartitions() const { return m_num_partitions; }

  // Returns number of replicas defined for the program module.
  size_t getNumReplicas() const { return m_num_replicas; }

  // Returns number of devices the binary is intended to run on, estimated from
  // the compiled graph.
  size_t getNumDevicesToUtilize() const { return m_num_devices_to_utilize; }

  // Returns devices mesh shape the binary is intended to run on, estimated from
  // the compiled graph.
  const std::vector<std::uint32_t> &getDevicesMeshShape() const {
    return m_devices_mesh_shape;
  }

  // Returns sharding information for inputs.
  const std::vector<mlir::tt::sharding_utils::MeshSharding> &
  getInputShardings() const {
    return m_input_shardings;
  }

  // Returns sharding information for outputs.
  const std::vector<mlir::tt::sharding_utils::MeshSharding> &
  getOutputShardings() const {
    return m_output_shardings;
  }

  // Returns input argument roles (weight vs input).
  const std::vector<InputArgumentRole> &getInputArgumentRoles() const {
    return m_input_argument_roles;
  }

private:
  // Creates VHLO module from the input program code.
  mlir::OwningOpRef<mlir::ModuleOp>
  createVHLOModule(const std::string_view &code);

  // Converts VHLO module to StableHLO module.
  void convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Runs frontend specific SHLO pipeline on the MLIR module.
  void runFrontendSHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Fills up the m_is_output_scalar array with information is the output type
  // scalar or not.
  void collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific inputs.
  void collectInputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific outputs.
  void collectOutputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about input argument roles (weight vs input).
  void
  collectInputArgumentRoles(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Runs compiler StableHLO pipeline on the MLIR module.
  void
  runCompilerStableHLOPipeline(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts StableHLO module to TTIR module.
  void convertFromSHLOToTTIR(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collects the information about the mesh shape the module is intended to run
  // on.
  void collectMeshShape(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Estimates devices mesh shape from input shardings in case the mesh
  // attribute is not set on the module.
  void estimateMeshShape();

  // Gets the number of devices the binary is intended to run on from the VHLO
  // module.
  void
  collectNumDevicesToUtilize(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts TTIR module to TTNN module.
  void convertFromTTIRToTTNN(const std::string &system_descriptor_path,
                             mlir::OwningOpRef<mlir::ModuleOp> &mlir_module,
                             const CompileOptions &compile_options,
                             tt::pjrt::ClientInstance *client_instance);

  // Creates flatbuffer binary from the built TTNN module.
  void
  createFlatbufferBinary(const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Verifies that creates flatbuffer binary satisfies conditions estimated by
  // the compiler from the input graph.
  void verifyCreatedFlatbufferBinary();

  // Checks if the resulting outputs and their shardings are valid.
  void checkOutputShardingShapes(
      const std::vector<tt::runtime::TensorDesc> &output_specs);

  // Prints module to console for debug purposes.
  static void printModule(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Enables IR printing between passes with VERBOSE or higher logger level.
  static void enableVerboseIRPrinting(mlir::PassManager &pm);

  // Checks if a particular type is scalar.
  bool isScalarType(mlir::Type type);

  // Converts a MLIR module into it's textual representation
  std::string getMlirCode(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Collect input sharding if we are using GSPMD.
  void
  collectInputShardingsGSPMD(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collect output sharding if we are using GSPMD.
  void
  collectOutputShardingsGSPMD(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collect input sharding if we are using Shardy.
  void
  collectInputShardingsShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collect output sharding if we are using Shardy.
  void
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

  // Compiled flatbuffer binary.
  tt::runtime::Binary m_flatbuffer_binary;

  // TTIR MLIR code.
  std::string m_ttir_mlir;

  // TTNN MLIR code.
  std::string m_ttnn_mlir;

  // Holds status of the last builder action.
  tt_pjrt_status m_status;

  // For every output, holds if the type is a scalar or not.
  std::vector<bool> m_is_output_scalar;

  // For every output, stores the expected data type.
  std::vector<PJRT_Buffer_Type> m_output_data_types;

  // Number of partitions defined for the program module.
  size_t m_num_partitions;

  // Number of replicas defined for the program module.
  size_t m_num_replicas;

  // Number of devices the binary is intended to run on, estimated from the
  // compiled graph.
  size_t m_num_devices_to_utilize;

  // Devices mesh shape the binary is intended to run on, estimated from the
  // compiled graph.
  std::vector<std::uint32_t> m_devices_mesh_shape;

  // For every input, holds the sharding information.
  std::vector<mlir::tt::sharding_utils::MeshSharding> m_input_shardings;

  // For every output, holds the sharding information.
  std::vector<mlir::tt::sharding_utils::MeshSharding> m_output_shardings;

  // For every input, holds the argument role (weight vs input).
  std::vector<InputArgumentRole> m_input_argument_roles;
};

} // namespace tt::pjrt::module_builder

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_MODULE_BUILDER_MODULE_BUILDER_H_
