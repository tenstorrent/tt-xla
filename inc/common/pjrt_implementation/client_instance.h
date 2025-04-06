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

// c++ standard library includes
#include <memory>
#include <vector>

// tt-xla includes
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/loaded_executable_instance.h"
#include "common/platform.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_CLIENT_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_CLIENT_INSTANCE_H_

namespace tt::pjrt {

class ModuleBuilder;

// Represents PJRT_Client structure and the functionality around it.
class ClientInstance {

public:
  ClientInstance(std::unique_ptr<Platform> platform);
  virtual ~ClientInstance();

  // Binds monomorphic entry-points for the client.
  static void bindApi(PJRT_Api *api);

  static ClientInstance *unwrap(PJRT_Client *client) {
    return reinterpret_cast<ClientInstance *>(client);
  }

  // Before the client is usable, it must be initialized.
  PJRT_Error *Initialize();

  Platform &platform() { return *platform_; }
  const std::string &cached_platform_name() { return cached_platform_name_; }
  const std::string &cached_platform_version() {
    return cached_platform_version_;
  }

  // Returns vector of raw pointers to all devices, including addressable and
  // non-addressable devices.
  const std::vector<DeviceInstance *> &getDevicesRaw() const {
    return m_devices_raw;
  }

  // Returns vector of raw pointers to addressable devices.
  const std::vector<DeviceInstance *> &getAddressableDevicesRaw() const {
    return m_addressable_devices_raw;
  }

  // Compiles.
  // See TODOs in PJRT_Client_Compile.
  PJRT_Error *
  Compile(const PJRT_Program *program, /*xla::CompileOptions options, */
          LoadedExecutableInstance **executable);

  // Advances the timeline, returning (current, next) time point values.
  std::tuple<uint64_t, uint64_t> AdvanceTimeline();

protected:
  std::string cached_platform_name_;
  std::string cached_platform_version_;

private:
  tt_pjrt_status PopulateDevices();

  std::unique_ptr<Platform> platform_;

  // Vector of all devices visible to the runtime, including addressable and
  // non-addressable devices.
  std::vector<std::unique_ptr<DeviceInstance>> m_devices;

  // Vector of raw pointers to all devices, owned by `m_devices`. Necessary to
  // have to be able to return it in `PJRT_Client_Devices` API call.
  std::vector<DeviceInstance *> m_devices_raw;

  // Vector of raw pointers to addressable devices, which are subset of and
  // owned by `m_devices`. Necessary to have to be able to return it in
  // `PJRT_Client_AddressableDevices` API call.
  std::vector<DeviceInstance *> m_addressable_devices_raw;

  std::unique_ptr<ModuleBuilder> module_builder_;

  // System descriptor (that TTIR to TTNN backend pipeline needs).
  tt::runtime::SystemDesc system_descriptor_;

  // TODO: Remove once tt-mlir supports passing the system descriptor object to
  // TTIR to TTNN backend pipeline.
  std::string cached_system_descriptor_path_;

  // Synchronization.
  // We keep one global execution timeline across all devices. The management
  // of this is currently somewhat primitive: we increment it by one for each
  // invocation. Batch invocations (i.e. across multiple devices), only
  // increment by one. In the future, additional parallelism could be plumbed
  // up to the framework to allow different kinds of timeline management.
  // Waiting on the current value of |execution_timeline_| will drain all
  // scheduled work to date.
  uint64_t execution_timeline_ = 0ull;
};

namespace internal {

// Implements PJRT_Client_Devices API function.
PJRT_Error *onClientDevices(PJRT_Client_Devices_Args *args);

// Implements PJRT_Client_AddressableDevices API function.
PJRT_Error *
onClientAddressableDevices(PJRT_Client_AddressableDevices_Args *args);

// Implements PJRT_Client_LookupDevice API function.
PJRT_Error *onClientLookupDevice(PJRT_Client_LookupDevice_Args *args);

// Implements PJRT_Client_LookupAddressableDevice API function.
PJRT_Error *
onClientLookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args *args);

// Implements PJRT_Client_BufferFromHostBuffer API function.
PJRT_Error *onBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_CLIENT_INSTANCE_H_
