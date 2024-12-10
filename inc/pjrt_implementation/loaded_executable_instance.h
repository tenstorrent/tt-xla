// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/status.h"
#include "pjrt_implementation/device_instance.h"
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_LOADED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

class ExecutableImage;

struct ResidentExecutable {
  DeviceInstance *device_instance;
  size_t arg_count;
  size_t result_count;
};

class LoadedExecutableInstance {
public:
  LoadedExecutableInstance(
      ClientInstance &client, ExecutableImage *image,
      const std::vector<DeviceInstance *> &addressable_devices)
      : client_(client), image_(image),
        addressable_devices_(addressable_devices) {}
  ~LoadedExecutableInstance();

  operator PJRT_LoadedExecutable *() {
    return reinterpret_cast<PJRT_LoadedExecutable *>(this);
  }
  static void BindApi(PJRT_Api *api);
  static LoadedExecutableInstance *Unwrap(PJRT_LoadedExecutable *exe) {
    return reinterpret_cast<LoadedExecutableInstance *>(exe);
  }

  const std::vector<DeviceInstance *> &addressable_devices() {
    return addressable_devices_;
  }

  // Loads all executables to addressable devices.
  tt_pjrt_status LoadAll();

  tt_pjrt_status GetDefaultResidentExecutable(ResidentExecutable **out_loaded);
  tt_pjrt_status GetArgResultCount(size_t *out_arg_count,
                                   size_t *out_result_count);

  tt_pjrt_status Execute(PJRT_LoadedExecutable_Execute_Args *args);

private:
  ClientInstance &client_;
  ExecutableImage *image_; // Ref-counted semantics.
  std::vector<DeviceInstance *> addressable_devices_;
  std::vector<ResidentExecutable> resident_executables_;
};

} // namespace tt::pjrt

#endif
