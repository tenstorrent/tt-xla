// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_LOADED_EXECUTABLE_INSTANCE_H_

// c++ standard library includes
#include <memory>
#include <mutex>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "api/device_instance.h"
#include "api/executable_image.h"
#include "utils/logging.h"
#include "utils/status.h"

namespace tt::pjrt {

// Represents `PJRT_LoadedExecutable` structure and the functionality around it.
// It is the in-memory loaded executable which is ready for input arguments to
// execute.
class LoadedExecutableInstance {
public:
  // Binds PJRT API functions implementation related to PJRT_LoadedExecutable
  // structure.
  static void bindApi(PJRT_Api *api);

  // Virtual destructor for proper cleanup of derived classes
  virtual ~LoadedExecutableInstance();

  // Casts this loaded executable instance to PJRT_LoadedExecutable pointer.
  operator PJRT_LoadedExecutable *() {
    return reinterpret_cast<PJRT_LoadedExecutable *>(this);
  }

  // Casts the PJRT_LoadedExecutable pointer to LoadedExecutableInstance
  // pointer.
  static LoadedExecutableInstance *unwrap(PJRT_LoadedExecutable *executable) {
    return reinterpret_cast<LoadedExecutableInstance *>(executable);
  }

  // Shares the underlying executable image.
  std::shared_ptr<ExecutableImage> getSharedExecutableImage() const {
    return m_executable_image;
  }

  // Returns subset of client's addressable devices that this executable will
  // run on.
  const std::vector<DeviceInstance *> &getAddressableDevices() {
    return m_addressable_devices;
  }

  // Returns true if the executable was deleted.
  bool isDeleted();

  // Releases the resources this loaded executable uses.
  void releaseResources();

  // Runs execution of this loaded executable.
  virtual tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args) = 0;

protected:
  // Creates loaded executable instance from the executable image.
  LoadedExecutableInstance(
      std::shared_ptr<ExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance)
      : m_executable_image(std::move(executable_image)),
        m_addressable_devices(addressable_devices), m_deleted(false),
        m_client_instance(client_instance) {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance constructor called for executable: %s",
           m_executable_image->getExecutableName().c_str());
  }

  // Executable image instance which is shared between executable and loaded
  // executable instances.
  std::shared_ptr<ExecutableImage> m_executable_image;

  // Subset of client's addressable devices that this executable will run on.
  const std::vector<DeviceInstance *> m_addressable_devices;

  // True if loaded executable was deleted, i.e. its resources are released.
  bool m_deleted;

  // Mutex guarding loaded executable deletion.
  std::mutex m_deleted_mutex;

  // Client instance that created this loaded executable.
  ClientInstance *m_client_instance;
};

namespace internal {

// Implements PJRT_LoadedExecutable_Destroy API function.
PJRT_Error *onLoadedExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args *args);

// Implements PJRT_LoadedExecutable_GetExecutable API function.
PJRT_Error *
onLoadedExecutableGetExecutable(PJRT_LoadedExecutable_GetExecutable_Args *args);

// Implements PJRT_LoadedExecutable_AddressableDevices API function.
PJRT_Error *onLoadedExecutableAddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args *args);

// Implements PJRT_LoadedExecutable_Delete API function.
PJRT_Error *onLoadedExecutableDelete(PJRT_LoadedExecutable_Delete_Args *args);

// Implements PJRT_LoadedExecutable_IsDeleted API function.
PJRT_Error *
onLoadedExecutableIsDeleted(PJRT_LoadedExecutable_IsDeleted_Args *args);

// Implements PJRT_LoadedExecutable_Execute API function.
PJRT_Error *onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_LOADED_EXECUTABLE_INSTANCE_H_
