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

  tt_pjrt_status buildModule(const std::string_view &code,
                             const std::string_view &format,
                             const std::string &system_descriptor_path);

  const tt::runtime::Binary &getBinary() const { return m_flatbuffer_binary; }

  const std::vector<bool> &getIsOutputScalar() const {
    return m_is_output_scalar;
  };

  size_t getNumDevicesToUtilize() const { return m_num_devices_to_utilize; }

  const std::vector<mlir::tt::sharding_utils::MeshSharding> &
  getInputShardings() const {
    return m_input_shardings;
  }

  const std::vector<mlir::tt::sharding_utils::MeshSharding> &
  getOutputShardings() const {
    return m_output_shardings;
  }

private:
  // Creates VHLO module from the input program code.
  mlir::OwningOpRef<mlir::ModuleOp>
  createVHLOModule(const std::string_view &code);

  // Gets the number of devices the binary is intended to run on from the VHLO
  // module.
  void
  collectNumDevicesToUtilize(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts VHLO module to StableHLO module.
  void convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Fills up the m_is_output_scalar array with information is the output type
  // scalar or not.
  void collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific inputs.
  void collectInputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Collects the information about the sharding of specific outputs.
  void collectOutputShardings(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Converts StableHLO module to TTIR module.
  void convertFromSHLOToTTIR(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts TTIR module to TTNN module.
  void convertFromTTIRToTTNN(const std::string &system_descriptor_path,
                             mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Creates flatbuffer binary from the built TTNN module.
  void
  createFlatbufferBinary(const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

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

  // Checks if the jax is using the Shardy mlir dialect.
  bool isUsingShardy(const mlir::OwningOpRef<mlir::ModuleOp> &module);

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

  // Flatbuffer binary.
  tt::runtime::Binary m_flatbuffer_binary;

  // Holds status of the last builder action.
  tt_pjrt_status m_status;

  // For every output, holds if the type is a scalar or not.
  std::vector<bool> m_is_output_scalar;

  // Number of devices the binary is intended to run on.
  size_t m_num_devices_to_utilize;

  // For every input, holds the sharding information.
  std::vector<mlir::tt::sharding_utils::MeshSharding> m_input_shardings;

  // For every output, holds the sharding information.
  std::vector<mlir::tt::sharding_utils::MeshSharding> m_output_shardings;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_MODULE_BUILDER_H_
