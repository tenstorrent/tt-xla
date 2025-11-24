// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_FLATBUFFER_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_FLATBUFFER_LOADED_EXECUTABLE_INSTANCE_H_

// c++ standard library includes
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/device_instance.h"
#include "api/executable_image.h"
#include "api/loaded_executable_instance.h"
#include "client_instance.h"
#include "utils/status.h"

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
  void releaseResources() override;

  // Runs execution of this loaded executable.
  tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args) override;

private:
  // Returns an input tensor constructed from the provided buffer instances,
  // prepared for execution. If we cannot reuse the already prepared tensor
  // contained within the buffer instances, this will involve calling
  // `toLayout()` which in most cases involves moving the data to the device.
  std::optional<tt::runtime::Tensor>
  prepareInputTensor(const std::vector<BufferInstance *> &arg_buffers,
                     tt::runtime::Device device, size_t num_devices,
                     std::uint32_t program_index, size_t arg_index) override;

  // Converts input tensor to desired layout. This might move it on device.
  tt::runtime::Tensor
  convertTensorLayout(tt::runtime::Tensor input_tensor,
                      std::uint32_t program_index, size_t arg_index,
                      const tt::runtime::Device &runtime_device);

  // Fills the output lists of the PJRT API with the outputs of tt runtime
  // execution. Creates BufferInstances with device tensors instead of
  // transferring them to host.
  void fillPJRTOutputLists(
      const std::vector<tt::runtime::Tensor> &output_tensors,
      size_t num_devices, PJRT_Buffer **const *output_lists,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types);

  // Returns the shape of the output on the specified index.
  std::vector<std::uint32_t> getOutputShape(size_t output_index);

  // Creates flatbuffer loaded executable instance from the executable image.
  FlatbufferLoadedExecutableInstance(
      std::shared_ptr<FlatbufferExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance);
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_FLATBUFFER_LOADED_EXECUTABLE_INSTANCE_H_
