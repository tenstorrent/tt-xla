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
#include <string>
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

  // Returns process index of this client.
  int getProcessIndex() const { return m_process_index; }

  // Returns vector of raw pointers to all devices, including addressable and
  // non-addressable devices.
  const std::vector<DeviceInstance *> &getDevicesRaw() const {
    return m_devices_raw;
  }

  // Returns vector of raw pointers to addressable devices.
  const std::vector<DeviceInstance *> &getAddressableDevicesRaw() const {
    return m_addressable_devices_raw;
  }

  // Compiles given mlir program.
  tt_pjrt_status compileMlirProgram(const PJRT_Program *mlir_program,
                                    LoadedExecutableInstance **out_executable);

protected:
  std::string cached_platform_name_;
  std::string cached_platform_version_;

private:
  tt_pjrt_status populateDevices();

  std::unique_ptr<Platform> platform_;

  // Process index of this client. Always 0 in single-process settings.
  int m_process_index;

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

  // Module builder that compiles program code.
  std::unique_ptr<ModuleBuilder> m_module_builder;

  // System descriptor (that TTIR to TTNN backend pipeline needs).
  tt::runtime::SystemDesc m_system_descriptor;

  // TODO: Remove once tt-mlir supports passing the system descriptor object to
  // TTIR to TTNN backend pipeline.
  std::string m_cached_system_descriptor_path;
};

namespace internal {

// Implements PJRT_Client_Destroy API function.
PJRT_Error *onClientDestroy(PJRT_Client_Destroy_Args *args);

// Implements PJRT_Client_PlatformName API function.
PJRT_Error *onClientPlatformName(PJRT_Client_PlatformName_Args *args);

// Implements PJRT_Client_ProcessIndex API function.
PJRT_Error *onClientProcessIndex(PJRT_Client_ProcessIndex_Args *args);

// Implements PJRT_Client_PlatformVersion API function.
PJRT_Error *onClientPlatformVersion(PJRT_Client_PlatformVersion_Args *args);

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

// Implements PJRT_Client_AddressableMemories API function.
PJRT_Error *
onClientAddressableMemories(PJRT_Client_AddressableMemories_Args *args);

// Implements PJRT_Client_Compile API function.
PJRT_Error *onClientCompile(PJRT_Client_Compile_Args *args);

// Implements PJRT_Client_DefaultDeviceAssignment API function.
PJRT_Error *
onClientDefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args *args);

// Implements PJRT_Client_BufferFromHostBuffer API function.
PJRT_Error *onBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_CLIENT_INSTANCE_H_
