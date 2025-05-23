// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_SRC_COMMON_MODULE_BUILDER_H_
#define TT_XLA_SRC_COMMON_MODULE_BUILDER_H_

// c++ standard library includes
#include <memory>
#include <string>

// llvm mlir includes
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

// tt-mlir includes
#include "tt/runtime/types.h"

#define TTMLIR_ENABLE_STABLEHLO 1
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"

// tt-xla includes
#include "status.h"

namespace tt::pjrt {

class ModuleBuilder {
public:
  ModuleBuilder();

  // Compiles given mlir module code and produces flatbuffer to execute on a
  // given system.
  tt_pjrt_status buildModule(const std::string_view &mlir_code,
                             const std::string &system_descriptor_path);

  // Returns compiled flatbuffer binary.
  const tt::runtime::Binary &getFlatbufferBinary() const {
    return m_flatbuffer_binary;
  }

  // Returns vector of boolean values determining if each output is scalar.
  const std::vector<bool> &getIsOutputScalar() const {
    return m_is_output_scalar;
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

  // MLIR program format name. This would ideally be defined in PJRT API header.
  static const std::string c_mlir_format_name;

private:
  // Creates VHLO module from the input program code.
  mlir::OwningOpRef<mlir::ModuleOp>
  createVHLOModule(const std::string_view &code);

  // Converts VHLO module to StableHLO module.
  void convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Fills up the m_is_output_scalar array with information is the output type
  // scalar or not.
  void collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific inputs.
  void collectInputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific outputs.
  void collectOutputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Gets shardy mesh attribute from the mesh op and adjusts it to 2D mesh in
  // case of 1D mesh so that the rest of our compiler logic can assume 2D mesh.
  mlir::sdy::MeshAttr getAdjustedShardyMeshAttribute(mlir::sdy::MeshOp mesh_op);

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
                             mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

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

  // Checks if a particular type is scalar.
  bool isScalarType(mlir::Type type);

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

  // Holds status of the last builder action.
  tt_pjrt_status m_status;

  // For every output, holds if the type is a scalar or not.
  std::vector<bool> m_is_output_scalar;

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
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_MODULE_BUILDER_H_
