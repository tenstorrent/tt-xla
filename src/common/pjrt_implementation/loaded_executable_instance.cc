// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/loaded_executable_instance.h"

// c++ standard library includes
#include <mutex>

// tracy includes
#include <tracy/Tracy.hpp>

// tt-xla includes
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/pjrt_implementation/executable_instance.h"
#include "common/status.h"

namespace tt::pjrt {

void LoadedExecutableInstance::bindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy = internal::onLoadedExecutableDestroy;
  api->PJRT_LoadedExecutable_GetExecutable =
      internal::onLoadedExecutableGetExecutable;
  api->PJRT_LoadedExecutable_AddressableDevices =
      internal::onLoadedExecutableAddressableDevices;
  api->PJRT_LoadedExecutable_Delete = internal::onLoadedExecutableDelete;
  api->PJRT_LoadedExecutable_IsDeleted = internal::onLoadedExecutableIsDeleted;
  api->PJRT_LoadedExecutable_Execute = internal::onLoadedExecutableExecute;
}

bool LoadedExecutableInstance::isDeleted() {
  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  return m_deleted;
}

void LoadedExecutableInstance::releaseResources() {
  if (m_deleted) {
    return;
  }

  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  if (m_deleted) {
    return;
  }

  // Base implementation just marks as deleted
  // Derived classes should override to release their specific resources
  m_deleted = true;
}

namespace internal {

PJRT_Error *
onLoadedExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Destroy");

  delete LoadedExecutableInstance::unwrap(args->executable);

  return nullptr;
}

PJRT_Error *onLoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_GetExecutable");

  LoadedExecutableInstance *loaded_executable =
      LoadedExecutableInstance::unwrap(args->loaded_executable);

  std::unique_ptr<ExecutableInstance> executable_instance =
      ExecutableInstance::createInstance(
          loaded_executable->getSharedExecutableImage());

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Executable_Destroy` on the executable.
  args->executable = *executable_instance.release();

  return nullptr;
}

PJRT_Error *onLoadedExecutableAddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_AddressableDevices");

  LoadedExecutableInstance *loaded_executable =
      LoadedExecutableInstance::unwrap(args->executable);

  const std::vector<DeviceInstance *> &addressable_devices =
      loaded_executable->getAddressableDevices();

  args->addressable_devices =
      reinterpret_cast<PJRT_Device *const *>(addressable_devices.data());
  args->num_addressable_devices = addressable_devices.size();

  return nullptr;
}

PJRT_Error *onLoadedExecutableDelete(PJRT_LoadedExecutable_Delete_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");

  LoadedExecutableInstance::unwrap(args->executable)->releaseResources();

  return nullptr;
}

PJRT_Error *
onLoadedExecutableIsDeleted(PJRT_LoadedExecutable_IsDeleted_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted");

  args->is_deleted =
      LoadedExecutableInstance::unwrap(args->executable)->isDeleted();

  return nullptr;
}

PJRT_Error *
onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");

  tt_pjrt_status status =
      LoadedExecutableInstance::unwrap(args->executable)->execute(args);

  return *ErrorInstance::makeError(status).release();
}

} // namespace internal

} // namespace tt::pjrt
