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
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

// tt-xla includes
#include "common/pjrt_implementation/input_argument_role.h"
#include "common/pjrt_implementation/module_builder/compile_options.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_

// tt-xla includes
#include "common/pjrt_implementation/module_builder/compile_options.h"

namespace tt::pjrt {

namespace module_builder {
class ModuleBuilder;
}

// Base class representing compiled image containing all the required
// information for its execution, except for the specific executable binary.
class ExecutableImage {

  friend class module_builder::ModuleBuilder;

public:
  // Virtual destructor for proper cleanup of derived classes
  virtual ~ExecutableImage() = default;

  // Validate the executable image after all the fields have been filled.
  virtual void validate() = 0;

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

  // Returns the fingerprint for this executable.
  const std::string &getFingerprint() const { return m_fingerprint; }

  const CompileOptions &getCompileOptions() const { return m_compile_options; }
  
  const std::vector<std::string> getInputArgumentNames() const {
    return m_input_argument_names;
  }

  // Creates a LoadedExecutableInstance from this executable image.
  virtual std::unique_ptr<class LoadedExecutableInstance> toExecutableInstance(
      std::vector<class DeviceInstance *> &&addressable_devices) = 0;

protected:
  ExecutableImage();

  // Original mlir code produced by the compiler, stored for debugging
  // purposes.
  std::string m_original_mlir_code;

  // TTIR MLIR code produced by the compiler, stored for debugging purposes.
  std::string m_ttir_mlir;

  // TTNN MLIR code produced by the compiler, stored for debugging purposes.
  std::string m_ttnn_mlir;

  // TODO(mrakita): Use the VHLO module name from the module builder, if it has
  // a name, otherwise some default string like the current one.
  // A name that identifies the executable.
  std::string m_executable_name = "tt_executable";

  // Number of partitions of the executable.
  size_t m_num_partitions;

  // Number of replicas of the executable.
  size_t m_num_replicas;

  // Number of devices this executable should run on, estimated from the
  // compiled code.
  size_t m_num_devices_to_utilize;

  // Devices mesh shape this executable should run on, estimated from the
  // compiled code.
  std::vector<std::uint32_t> m_devices_mesh_shape;

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
  std::vector<mlir::tt::sharding_utils::MeshSharding> m_input_sharding;

  // Hold the sharding information for each output.
  std::vector<mlir::tt::sharding_utils::MeshSharding> m_output_sharding;

  // Holds the information on memory kind of the output.
  std::vector<const char *> m_output_memory_kinds;

  // Holds the information about the individual sizes of the memory kind strings
  // of the outputs.
  std::vector<size_t> m_output_memory_kinds_sizes;

  // Cached fingerprint for this executable.
  std::string m_fingerprint;

  // For every input, holds the argument role (weight vs input).
  std::vector<tt::pjrt::InputArgumentRole> m_input_argument_roles;

  CompileOptions m_compile_options;

  // Generates the fingerprint for this executable based on compilation inputs.
  virtual std::string generateFingerprint() const;

  // For every input, holds the argument name.
  std::vector<std::string> m_input_argument_names;
};

// Derived class for executables backed by flatbuffer binaries
class FlatbufferExecutableImage
    : public ExecutableImage,
      public std::enable_shared_from_this<FlatbufferExecutableImage> {

  friend class module_builder::ModuleBuilder;

public:
  // Creates a new blank FlatbufferExecutableImage, ready to be filled by
  // ModuleBuilder.
  template <typename... Args>
  static std::shared_ptr<FlatbufferExecutableImage>
  createInstance(Args &&...args) {
    struct MakeSharedEnabler : public FlatbufferExecutableImage {
      MakeSharedEnabler(Args &&...args)
          : FlatbufferExecutableImage(std::forward<Args>(args)...) {}
    };
    return std::make_shared<MakeSharedEnabler>(std::forward<Args>(args)...);
  }

  // Returns flatbuffer binary produced by the compiler.
  const tt::runtime::Binary &getFlatbufferBinary() const {
    return m_flatbuffer_binary;
  }

  // Validate the executable image after all the fields have been filled.
  void validate() override;

  // Creates a LoadedExecutableInstance from this executable image.
  std::unique_ptr<class LoadedExecutableInstance> toExecutableInstance(
      std::vector<class DeviceInstance *> &&addressable_devices) override;

private:
  // Generates the fingerprint for this executable based on compilation inputs.
  std::string generateFingerprint() const final;

  // Flatbuffer binary produced by the compiler.
  tt::runtime::Binary m_flatbuffer_binary;

protected:
  FlatbufferExecutableImage();
};

// Derived class for executables backed by shared object files
class SOExecutableImage
    : public ExecutableImage,
      public std::enable_shared_from_this<SOExecutableImage> {

  friend class module_builder::ModuleBuilder;

public:
  // Creates a new blank SOExecutableImage, ready to be filled by ModuleBuilder.
  template <typename... Args>
  static std::shared_ptr<SOExecutableImage> createInstance(Args &&...args) {
    struct MakeSharedEnabler : public SOExecutableImage {
      MakeSharedEnabler(Args &&...args)
          : SOExecutableImage(std::forward<Args>(args)...) {}
    };
    return std::make_shared<MakeSharedEnabler>(std::forward<Args>(args)...);
  }

  // Returns path to the shared object file.
  const std::string &getSOPath() const { return m_so_path; }

  // Validate the executable image after all the fields have been filled.
  void validate() override;

  // Creates a LoadedExecutableInstance from this executable image.
  std::unique_ptr<class LoadedExecutableInstance> toExecutableInstance(
      std::vector<class DeviceInstance *> &&addressable_devices) override;

private:
  // Generates the fingerprint for this executable based on compilation inputs.
  std::string generateFingerprint() const final;

  // Path to the shared object file.
  std::string m_so_path;

protected:
  SOExecutableImage();
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_IMAGE_H_
