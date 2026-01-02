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
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXECUTABLE_IMAGE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXECUTABLE_IMAGE_H_

// tt-xla includes
#include "api/compile_options.h"
#include "api/module_builder/module_builder.h"

namespace tt::pjrt {

// Represents compiled image containing all the required information for its
// execution.
class ExecutableImage {

public:
  // Virtual destructor for proper cleanup of derived classes
  virtual ~ExecutableImage() = default;

  // Returns original mlir code produced by the xla plugin.
  const std::string &getOriginalMlirCode() const {
    return m_original_mlir_code;
  }

  const std::string &getTTIRMlirCode() const { return m_ttir_mlir; }

  const std::string &getTTNNMlirCode() const { return m_ttnn_mlir; }

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

  // Returns the vector of output data types.
  std::vector<PJRT_Buffer_Type> &getOutputTypes() { return m_output_types; }

  // Returns the vector of output dimensions.
  std::vector<std::vector<std::uint32_t>> &getOutputDimensions() {
    return m_output_dimensions;
  }

  // Returns the vector of output ranks.
  std::vector<size_t> &getOutputRanks() { return m_output_ranks; }

  // Returns the vector of output dimensions concatenated in a flat array.
  std::vector<std::int64_t> &getOutputDimensionsFlat() {
    return m_output_dimensions_flat;
  }

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

  // Returns the compile options used to create this executable.
  const CompileOptions &getCompileOptions() const { return m_compile_options; }

  // Returns the fingerprint for this executable.
  const std::string &getFingerprint() const { return m_fingerprint; }

  // Returns sanitized MLIR code cleaned for XLA ingestion.
  const std::string &getSanitizedMlirCode() const {
    return m_sanitized_mlir_code;
  }

  // Creates a LoadedExecutableInstance from this executable image.
  virtual std::unique_ptr<class LoadedExecutableInstance> toExecutableInstance(
      std::vector<class DeviceInstance *> &&addressable_devices,
      ClientInstance *client_instance) = 0;
  
  std::string m_checkpointed_mlir_code;

protected:
  // Constructs executable image instance from the information given by the
  // compiler.
  ExecutableImage(
      std::string &&original_mlir_code, std::string &&ttir_mlir_code,
      std::string &&ttnn_mlir_code, std::string &&executable_name,
      size_t num_inputs, size_t num_outputs,
      std::vector<std::vector<std::uint32_t>> &&output_dimensions,
      std::vector<size_t> &&output_ranks,
      std::vector<std::int64_t> &&output_dimensions_flat, size_t num_partitions,
      size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
      std::vector<const char *> &&output_memory_kinds,
      std::vector<size_t> &&output_memory_kinds_sizes,
      std::string &&sanitized_mlir_code,
      CompileOptions &&compile_options);

  // Generates the fingerprint for this executable based on compilation inputs.
  virtual std::string generateFingerprint() const;

  // Cached fingerprint for this executable.
  std::string m_fingerprint;

private:
  // Original mlir code produced by the compiler, stored for debugging
  // purposes.
  std::string m_original_mlir_code;

  // TTIR MLIR code produced by the compiler, stored for debugging purposes.
  std::string m_ttir_mlir;

  // TTNN MLIR code produced by the compiler, stored for debugging purposes.
  std::string m_ttnn_mlir;

  // Sanitized MLIR code cleaned for XLA ingestion, stored for debugging purposes.
  std::string m_sanitized_mlir_code;

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

  // Compile options used to create this executable.
  CompileOptions m_compile_options;
};

/*
Brief context:
Our primary flow involves exporting TTNN MLIR to a flatbuffer which is then
basically interpreted op-by-op by the mlir runtime. We have an alternative flow
where we export TTNN MLIR as compileable code(either C++ or Python) targetting
TT-NN library(a part of Metalium). These two can be considered separate paths
from tt-xla to our hardware. ExecutableImage and LoadedExecutableInstance
classes have been split up into respective versions for each path, with
Flatbuffer* versions denoting the primary flow going trough flatbuffers and
runtime and SO* versions denoting the flow where we compile to shared object
files. SO emitting is not implemented yet.
*/

// Derived class for executables going trough the default path of packing ops
// into a flatbuffer.
class FlatbufferExecutableImage
    : public ExecutableImage,
      public std::enable_shared_from_this<FlatbufferExecutableImage> {
public:
  // Creates new executable image instance from the information given by the
  // compiler.
  static std::shared_ptr<FlatbufferExecutableImage> createInstance(
      const tt::runtime::Binary &flatbuffer_binary,
      std::string &&original_mlir_code, std::string &&checkpointed_mlir_code,
      std::string &&ttir_mlir_code, std::string &&ttnn_mlir_code,
      std::string &&executable_name, size_t num_inputs, size_t num_outputs,
      std::vector<std::vector<std::uint32_t>> &&output_dimensions,
      std::vector<size_t> &&output_ranks,
      std::vector<std::int64_t> &&output_dimensions_flat, size_t num_partitions,
      size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
      std::vector<const char *> output_memory_kinds,
      std::vector<size_t> output_memory_kinds_sizes,
      std::string &&sanitized_mlir_code,
      CompileOptions &&compile_options);

  // Returns flatbuffer binary produced by the compiler.
  const tt::runtime::Binary &getFlatbufferBinary() const {
    return m_flatbuffer_binary;
  }

  // Creates a LoadedExecutableInstance from this executable image.
  std::unique_ptr<class LoadedExecutableInstance> toExecutableInstance(
      std::vector<class DeviceInstance *> &&addressable_devices,
      ClientInstance *client_instance) override;

private:
  // Constructs executable image instance from the information given by the
  // compiler.
  FlatbufferExecutableImage(
      const tt::runtime::Binary &flatbuffer_binary,
      std::string &&original_mlir_code, std::string &&checkpointed_mlir_code,
      std::string &&ttir_mlir_code, std::string &&ttnn_mlir_code,
      std::string &&executable_name, size_t num_inputs, size_t num_outputs,
      std::vector<std::vector<std::uint32_t>> output_dimensions,
      std::vector<size_t> output_ranks,
      std::vector<std::int64_t> output_dimensions_flat, size_t num_partitions,
      size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
      std::vector<const char *> &&output_memory_kinds,
      std::vector<size_t> &&output_memory_kinds_sizes,
      std::string &&sanitized_mlir_code,
      CompileOptions &&compile_options);

  // Generates the fingerprint for this executable based on compilation inputs.
  std::string generateFingerprint() const final;

  // Flatbuffer binary produced by the compiler.
  tt::runtime::Binary m_flatbuffer_binary;
};

// Derived class for executables backed by shared object files emitted via
// TT-Alchemist.
class SOExecutableImage
    : public ExecutableImage,
      public std::enable_shared_from_this<SOExecutableImage> {
public:
  static std::shared_ptr<SOExecutableImage> createInstance(
      std::string original_mlir_code, std::string ttir_mlir_code,
      std::string ttnn_mlir_code, std::string executable_name,
      size_t num_inputs, size_t num_outputs,
      std::vector<std::vector<std::uint32_t>> output_dimensions,
      std::vector<size_t> output_ranks,
      std::vector<std::int64_t> output_dimensions_flat, size_t num_partitions,
      size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
      std::vector<const char *> output_memory_kinds,
      std::vector<size_t> output_memory_kinds_sizes,
      std::string &&sanitized_mlir_code,
      CompileOptions &&compile_options);

  // Creates a LoadedExecutableInstance from this executable image.
  std::unique_ptr<class LoadedExecutableInstance> toExecutableInstance(
      std::vector<class DeviceInstance *> &&addressable_devices,
      ClientInstance *client_instance) override;

private:
  // Constructs executable image instance from the information given by the
  // compiler.
  SOExecutableImage(
      std::string &&original_mlir_code, std::string &&ttir_mlir_code,
      std::string &&ttnn_mlir_code, std::string &&executable_name,
      size_t num_inputs, size_t num_outputs,
      std::vector<std::vector<std::uint32_t>> output_dimensions,
      std::vector<size_t> output_ranks,
      std::vector<std::int64_t> output_dimensions_flat, size_t num_partitions,
      size_t num_replicas, size_t num_devices_to_utilize,
      const std::vector<std::uint32_t> &devices_mesh_shape,
      const std::vector<mlir::tt::sharding_utils::MeshSharding> &input_sharding,
      const std::vector<mlir::tt::sharding_utils::MeshSharding>
          &output_sharding,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types,
      std::vector<const char *> &&output_memory_kinds,
      std::vector<size_t> &&output_memory_kinds_sizes,
      std::string &&sanitized_mlir_code,
      CompileOptions &&compile_options);

  // Generates the fingerprint for this executable based on compilation inputs.
  std::string generateFingerprint() const final;

  // Logically so_path is part of SOExecutableImage, but it is already stored in
  // compile_options.export_path
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXECUTABLE_IMAGE_H_
