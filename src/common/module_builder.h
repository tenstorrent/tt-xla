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

// tt-xla includes
#include "status.h"

namespace tt::pjrt {

class ModuleBuilder {
public:
  ModuleBuilder();

  tt_pjrt_status buildModule(const std::string_view &code,
                             const std::string_view &format);

  std::shared_ptr<void> getBinary() const { return m_flatbuffer_binary; }

  size_t getNumInputs() const { return m_num_inputs; };

  size_t getNumOutputs() const { return m_num_outputs; };

  bool isOutputScalar(size_t index) const;

private:
  // Creates VHLO module from the input program code.
  mlir::OwningOpRef<mlir::ModuleOp>
  createVHLOModule(const std::string_view &code);

  // Converts VHLO module to StableHLO module.
  void convertFromVHLOToSHLO(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Fills up the m_is_output_scalar array with information is the output type
  // scalar or not.
  void collectOutputTypes(const mlir::OwningOpRef<mlir::ModuleOp> &module);

  // Converts StableHLO module to TTIR module.
  void convertFromSHLOToTTIR(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Converts TTIR module to TTNN module.
  void convertFromTTIRToTTNN(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Creates flatbuffer binary from the built TTNN module.
  void
  createFlatbufferBinary(const mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Prints module to console for debug purposes.
  static void printModule(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module);

  // Checks if a particular type is scalar.
  bool isScalarType(mlir::Type type);

  // MLIR context handle.
  std::unique_ptr<mlir::MLIRContext> m_context;

  // Flatbuffer binary handle.
  std::shared_ptr<void> m_flatbuffer_binary;

  // Number of binary program inputs.
  size_t m_num_inputs;

  // Number of binary program outputs.
  size_t m_num_outputs;

  // Holds status of the last builder action.
  tt_pjrt_status m_status;

  // For every output, holds if the type is a scalar or not.
  std::vector<bool> m_is_output_scalar;
};

} // namespace tt::pjrt

#endif // TT_XLA_SRC_COMMON_MODULE_BUILDER_H_
