// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>
#include <string>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

namespace tt::pjrt {

// Represents compiled image containing all the required information for its
// execution.
class ExecutableImage {

public:
  // Creates new executable image instance from the information given by the
  // compiler.
  static std::shared_ptr<ExecutableImage> createInstance(
      const tt::runtime::Binary &flatbuffer_binary,
      std::string &&optimized_mlir_code, std::string &&executable_name,
      size_t num_partitions, size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<bool> &is_output_scalar);

  // Returns flatbuffer binary produced by the compiler.
  const tt::runtime::Binary &getFlatbufferBinary() const {
    return m_flatbuffer_binary;
  }

  // Returns optimized mlir code produced by the compiler.
  const std::string &getOptimizedMlirCode() const {
    return m_optimized_mlir_code;
  }

  // Returns a name that identifies the executable.
  const std::string &getExecutableName() const { return m_executable_name; }

  // Returns number of replicas of the executable.
  size_t getNumReplicas() const { return m_num_replicas; }

  // Returns number of partitions of the executable.
  size_t getNumPartitions() const { return m_num_partitions; }

  // Returns number of devices this executable should run on.
  size_t getNumDevicesToUtilize() const { return m_num_devices_to_utilize; }

  // Returns devices mesh shape this executable should run on.
  const std::vector<std::uint32_t> &getDevicesMeshShape() const {
    return m_devices_mesh_shape;
  }

  // Returns number of input buffers per device this executable requires.
  const size_t getNumInputs() const { return m_num_inputs; }

  // Returns number of output buffers per device produced by this executable.
  const size_t getNumOutputs() const { return m_num_outputs; }

  // Returns raw pointer to data types for each output buffer.
  PJRT_Buffer_Type *getOutputTypesRaw() { return m_output_types.data(); }

  // Returns the shape for the output buffer with a given index.
  const std::vector<std::uint32_t> &getOutputShape(size_t output_index) const;

  // Returns raw pointer to ranks for each output buffer.
  const size_t *getOutputRanksRaw() const { return m_output_ranks.data(); }

  // Returns raw pointer to output dimensions concatenated in a flat array.
  const std::int64_t *getOutputDimensionsFlatRaw() const {
    return m_output_dimensions_flat.data();
  }

  // Returns the sharding information for the input buffer with a given index.
  const mlir::tt::sharding_utils::MeshSharding &
  getInputSharding(size_t input_index) const;

  // Returns the sharding information for the output buffer with a given index.
  const mlir::tt::sharding_utils::MeshSharding &
  getOutputSharding(size_t output_index) const;

  // Gets the vector of memory kinds for each output.
  const std::vector<const char *> &getOutputMemoryKinds() const {
    return m_output_memory_kinds;
  }

  // Gets the vector of sizes of the memory kinds for each output.
  const std::vector<size_t> &getOutputMemoryKindsSizes() const {
    return m_output_memory_kinds_sizes;
  }

private:
  // Constructs executable image instance from the information given by the
  // compiler.
  ExecutableImage(
      const tt::runtime::Binary &flatbuffer_binary,
      std::string &&optimized_mlir_code, std::string &&executable_name,
      size_t num_partitions, size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<bool> &is_output_scalar);

  // Flatbuffer binary produced by the compiler.
  tt::runtime::Binary m_flatbuffer_binary;

  // Optimized mlir code produced by the compiler, stored for debugging
  // purposes.
  std::string m_optimized_mlir_code;

  // A name that identifies the executable.
  std::string m_executable_name;

  // Number of partitions of the executable.
  size_t m_num_partitions;

  // Number of replicas of the executable.
  size_t m_num_replicas;

  // Number of devices this executable should run on, estimated from the
  // compiled code.
  size_t m_num_devices_to_utilize;

  // Devices mesh shape this executable should run on, estimated from the
  // compiled code.
  const std::vector<std::uint32_t> m_devices_mesh_shape;

  // Number of input buffers per device this executable requires.
  size_t m_num_inputs;

  // Number of output buffers per device produced by this executable.
  size_t m_num_outputs;

  // Holds data type for each output buffer.
  std::vector<PJRT_Buffer_Type> m_output_types;

  // Holds dimensions for each output buffer.
  std::vector<std::vector<std::uint32_t>> m_output_dimensions;

  // Stores rank (number of dimensions) of each output. It could be deduced from
  // the output dimensions vector, but we need pointer to data to return back in
  // `PJRT_Executable_OutputDimensions` API function.
  std::vector<size_t> m_output_ranks;

  // Stores all output dimensions concatenated in a flat array.
  std::vector<std::int64_t> m_output_dimensions_flat;

  // Hold the sharding information for each input.
  const std::vector<mlir::tt::sharding_utils::MeshSharding> m_input_sharding;

  // Hold the sharding information for each output.
  const std::vector<mlir::tt::sharding_utils::MeshSharding> m_output_sharding;

  // Holds the information on memory kind of the output.
  std::vector<const char *> m_output_memory_kinds;

  // Holds the information about the individual sizes of the memory kind strings
  // of the outputs.
  std::vector<size_t> m_output_memory_kinds_sizes;
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
