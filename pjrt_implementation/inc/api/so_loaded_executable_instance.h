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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "mlir/Support/LogicalResult.h"
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/executable_image.h"
#include "api/loaded_executable_instance.h"
#include "utils/status.h"

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

  // Create default-initialized output buffers for SO execution
  void createDefaultOutputBuffers(PJRT_Buffer **const *output_lists,
                                  size_t num_devices);

private:
  // Creates SO loaded executable instance from the executable image.
  SOLoadedExecutableInstance(
      std::shared_ptr<SOExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance);
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_
