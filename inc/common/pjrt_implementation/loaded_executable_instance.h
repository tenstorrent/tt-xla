// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "xla/pjrt/c/pjrt_c_api.h"

#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

struct ResidentExecutable {
  DeviceInstance *device_instance;
  size_t arg_count;
  size_t result_count;
};

class LoadedExecutableInstance {
public:
  LoadedExecutableInstance(
      ClientInstance &client, ExecutableImage *image,
      const std::vector<DeviceInstance *> &addressable_devices,
      size_t num_devices_to_utilize)
      : client_(client), image_(image),
        addressable_devices_(addressable_devices),
        num_devices_to_utilize_(num_devices_to_utilize) {}
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

  size_t get_num_devices_to_utilize() const { return num_devices_to_utilize_; }

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
  size_t num_devices_to_utilize_;
  std::vector<ResidentExecutable> resident_executables_;
};

} // namespace tt::pjrt

#endif
