// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_FLATBUFFER_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_FLATBUFFER_LOADED_EXECUTABLE_INSTANCE_H_

// c++ standard library includes
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "mlir/Support/LogicalResult.h"

// tt-xla includes
#include "common/pjrt_implementation/loaded_executable_instance.h"
#include "common/status.h"

// Forward declarations
namespace tt::runtime {
class Device;
class Tensor;
} // namespace tt::runtime

namespace mlir::tt::sharding_utils {
struct MeshSharding;
} // namespace mlir::tt::sharding_utils

namespace tt::pjrt {
class BufferInstance;
class FlatbufferExecutableImage;
} // namespace tt::pjrt

namespace tt::pjrt {

// Derived class for Flatbuffer-based loaded executables
class FlatbufferLoadedExecutableInstance : public LoadedExecutableInstance {
public:
  // Creates new flatbuffer loaded executable instance.
  static std::unique_ptr<FlatbufferLoadedExecutableInstance>
  createInstance(std::shared_ptr<FlatbufferExecutableImage> executable_image,
                 std::vector<DeviceInstance *> &&addressable_devices,
                 ClientInstance *client_instance);

  // Shares the underlying executable image.
  std::shared_ptr<FlatbufferExecutableImage> getSharedExecutableImage() const;

  // Releases the resources this loaded executable uses.
  void releaseResources();

  // Runs execution of this loaded executable.
  tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args) override;

private:
  // Opens devices on which input arguments are placed, which we assume are the
  // the devices where computation will run, if their count is equal to the
  // corresponding devices count in the mesh shape estimated by the compiler.
  std::optional<tt::runtime::Device>
  getOrCreateMeshDevice(PJRT_Buffer *const *const *argument_lists,
                        size_t num_args, size_t num_devices,
                        PJRT_Device *pjrt_device);

  // Collects device ids from the addressable devices.
  std::unordered_set<int>
  getDeviceIds(PJRT_Buffer *const *const *argument_lists, size_t num_args,
               size_t num_devices);

  // Gets input runtime tensors from the arguments' buffers and converts them to
  // desired layout determined from the compiled graph.
  tt_pjrt_status
  getInputRuntimeTensors(PJRT_Buffer *const *const *argument_lists,
                         size_t num_args, size_t num_devices,
                         const tt::runtime::Device &runtime_device,
                         std::uint32_t program_index,
                         std::vector<tt::runtime::Tensor> &input_tensors);

  // Returns an input tensor constructed from the provided buffer instances,
  // prepared for execution. If we cannot reuse the already prepared tensor
  // contained within the buffer instances, this will involve calling
  // `toLayout()` which in most cases involves moving the data to the device.
  std::optional<tt::runtime::Tensor>
  prepareInputTensor(const std::vector<BufferInstance *> &arg_buffers,
                     tt::runtime::Device device, size_t num_devices,
                     std::uint32_t program_index, size_t arg_index);

  // Fills strategy map from sharding configuration.
  // TODO: This function might be better suited living in the tt-mlir
  // repository. https://github.com/tenstorrent/tt-xla/issues/374
  static mlir::FailureOr<std::unordered_map<std::string, std::string>>
  fillStrategyMapFromSharding(
      const mlir::tt::sharding_utils::MeshSharding &meshSharding,
      size_t num_devices);

  // Either returns single tensor or creates multi-device host tensor from arg
  // tensors, depending on the strategy.
  tt::runtime::Tensor getTensorFromStrategy(
      const std::vector<BufferInstance *> &arg_buffers,
      const std::unordered_map<std::string, std::string> &strategy);

  // Converts input tensor to desired layout. This might move it on device.
  tt::runtime::Tensor
  convertTensorLayout(tt::runtime::Tensor input_tensor,
                      std::uint32_t program_index, size_t arg_index,
                      const tt::runtime::Device &runtime_device);

  // Untilizes output tensors and transfers them from device to host.
  tt_pjrt_status untilizeToHost(
      const std::vector<tt::runtime::Tensor> &output_tensors,
      size_t num_devices,
      std::vector<std::vector<tt::runtime::Tensor>> &untilized_output_tensors);

  // Fills the output lists of the PJRT API with the outputs of tt runtime
  // execution.
  void fillPJRTOutputLists(
      const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs,
      size_t num_devices, PJRT_Buffer **const *output_lists,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types);

  // Returns the shape of the output on the specified index.
  std::vector<std::uint32_t> getOutputShape(size_t output_index);

  // Save all graph inputs as files, in metal's tensorbin format.
  void dumpInputs(const std::vector<tt::runtime::Tensor> &input_tensors);

  // Creates flatbuffer loaded executable instance from the executable image.
  FlatbufferLoadedExecutableInstance(
      std::shared_ptr<FlatbufferExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance);
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_FLATBUFFER_LOADED_EXECUTABLE_INSTANCE_H_
