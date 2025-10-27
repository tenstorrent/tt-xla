// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_

// c++ standard library includes
#include <cstdint>
#include <memory>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "api/loaded_executable_instance.h"
#include "utils/status.h"

// Forward declarations
namespace tt::runtime {
class Device;
class Tensor;
} // namespace tt::runtime

namespace tt::pjrt {
class SOExecutableImage;
} // namespace tt::pjrt

namespace tt::pjrt {

// Derived class for SO-based loaded executables
class SOLoadedExecutableInstance : public LoadedExecutableInstance {
public:
  // Creates new SO loaded executable instance.
  static std::unique_ptr<SOLoadedExecutableInstance>
  createInstance(std::shared_ptr<SOExecutableImage> executable_image,
                 std::vector<DeviceInstance *> &&addressable_devices,
                 ClientInstance *client_instance);

  // Shares the underlying executable image.
  std::shared_ptr<SOExecutableImage> getSharedExecutableImage() const;

  // Releases the resources this loaded executable uses.
  void releaseResources();

  // Runs execution of this loaded executable.
  tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args) override;

protected:
  // Converts input tensor to desired layout. This might move it on device.
  tt::runtime::Tensor
  convertTensorLayout(tt::runtime::Tensor input_tensor,
                      std::uint32_t program_index, size_t arg_index,
                      const tt::runtime::Device &runtime_device);

private:
  // Creates SO loaded executable instance from the executable image.
  SOLoadedExecutableInstance(
      std::shared_ptr<SOExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance);
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_
