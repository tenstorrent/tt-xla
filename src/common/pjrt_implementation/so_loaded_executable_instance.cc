// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common/pjrt_implementation/so_loaded_executable_instance.h"

// c++ standard library includes
#include <mutex>

// tt-mlir includes
#include "tt/runtime/types.h"

// tt-xla includes
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/status.h"

namespace tt::pjrt {

std::unique_ptr<SOLoadedExecutableInstance>
SOLoadedExecutableInstance::createInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices,
    ClientInstance *client_instance) {
  struct make_unique_enabler : public SOLoadedExecutableInstance {
    make_unique_enabler(std::shared_ptr<SOExecutableImage> executable_image,
                        std::vector<DeviceInstance *> &&addressable_devices,
                        ClientInstance *client_instance)
        : SOLoadedExecutableInstance(std::move(executable_image),
                                     std::move(addressable_devices),
                                     client_instance) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image),
                                               std::move(addressable_devices),
                                               client_instance);
}

SOLoadedExecutableInstance::SOLoadedExecutableInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices,
    ClientInstance *client_instance)
    : LoadedExecutableInstance(std::move(executable_image), addressable_devices,
                               client_instance) {}

std::shared_ptr<SOExecutableImage>
SOLoadedExecutableInstance::getSharedExecutableImage() const {
  return std::static_pointer_cast<SOExecutableImage>(m_executable_image);
}

void SOLoadedExecutableInstance::releaseResources() {
  if (m_deleted) {
    return;
  }

  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  if (m_deleted) {
    return;
  }

  // Release SO-specific resources
  m_executable_image.reset();

  m_deleted = true;
}

tt_pjrt_status
SOLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args *args) {
  // TODO: Implement SO execution logic
  return tt_pjrt_status::kUnimplemented;
}

tt::runtime::Tensor SOLoadedExecutableInstance::convertTensorLayout(
    tt::runtime::Tensor input_tensor, std::uint32_t program_index,
    size_t arg_index, const tt::runtime::Device &runtime_device) {
  // TODO: Implement SO tensor layout conversion
  return input_tensor; // Placeholder
}

} // namespace tt::pjrt
