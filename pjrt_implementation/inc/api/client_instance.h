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
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// tt-xla includes
#include "api/device_instance.h"
#include "api/loaded_executable_instance.h"
#include "api/memory_instance.h"
#include "utils/status.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_CLIENT_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_CLIENT_INSTANCE_H_

namespace tt::pjrt {

class BufferInstance;

namespace module_builder {
class ModuleBuilder;
}

// Singleton class that wraps the PJRT Client Instance.
// Ensures that we properly destroy the client instance on process termination.
//
// NOTE: This is needed since `torch_xla` implementation doesn't call
// `PJRT_Client_Destroy` API properly and `tt-metal` currently cannot recover
// (on n300 boards) if we do not properly close all previously opened devices -
// which is done on client destruction.
//
// NOTE: This serves only as a fallback option if `PJRT_Client_Destroy` is not
// called by the framework.
class GlobalClientInstanceSingleton {
public:
  static ClientInstance *getClientInstance();
  static PJRT_Error *initClient();
  static void destroyClient();

private:
  GlobalClientInstanceSingleton(std::unique_ptr<ClientInstance> client_instance)
      : m_client_instance(std::move(client_instance)) {}

  bool isInitialized() const { return m_client_instance != nullptr; }

  static GlobalClientInstanceSingleton &getInstance();
  std::unique_ptr<ClientInstance> m_client_instance;
};

// Represents PJRT_Client structure and the functionality around it.
class ClientInstance {

public:
  ClientInstance();

  // Non-default destructor required to:
  // - Close the associated mesh device if one was opened.
  // - Remove the cached system descriptor file from disk.
  ~ClientInstance();

  // Binds monomorphic entry-points for the client.
  static void bindApi(PJRT_Api *api);

  static ClientInstance *unwrap(PJRT_Client *client) {
    return reinterpret_cast<ClientInstance *>(client);
  }

  // Before the client is usable, it must be initialized.
  PJRT_Error *initialize();

  const std::string &platformName() { return m_platform_name; }
  const std::string &platformVersion() { return m_platform_version; }

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

  const std::vector<MemoryInstance *> &getAddressableMemoriesRaw() const {
    return m_addressable_memories_raw;
  }

  // Returns the mesh device of the provided shape. If there is already opened
  // mesh device within this client instance and its shape matches the provided
  // shape, it is returned. Otherwise, we close any previously opened mesh
  // device and open a new one with the provided shape.
  //
  // NOTE: this method is not thread-safe and we will need to revisit this when
  // adding support for parallel execution.
  tt::runtime::Device
  getOrCreateMeshDevice(const std::vector<uint32_t> &target_mesh_shape);

  // Registers a buffer with this client for tracking. This is called when a
  // buffer is created so we can materialize all buffers before mesh reshape.
  void registerBuffer(BufferInstance *buffer);

  // Unregisters a buffer from this client. This is called when a buffer is
  // destroyed.
  void unregisterBuffer(BufferInstance *buffer);

  // Materializes all tracked buffers that have device tensors but no host
  // tensors. This should be called before mesh reshape to prevent data loss.
  void materializeAllBuffersToHost();

  // Returns parent mesh.
  std::optional<tt::runtime::Device> &parentMesh() { return m_parent_mesh; };

  // Returns parent mesh.
  const std::optional<tt::runtime::Device> &parentMesh() const {
    return m_parent_mesh;
  };

  // Closes currently opened mesh device and submesh device, if any.
  void closeMeshDevice();

  // Returns the optimizer submesh device of the provided shape. If there is
  // already opened optimizer submesh and its shape matches the provided shape,
  // it is returned. Otherwise, we close any previously opened optimizer submesh
  // and create a new one with the provided shape.
  //
  // NOTE: this method is not thread-safe and we will need to revisit this when
  // adding support for parallel execution.
  tt::runtime::Device
  getOrCreateOptimizerSubmesh(const std::vector<uint32_t> &target_mesh_shape);

  // Closes currently opened parrent mesh device.
  void closeParentMesh();

  // Closes currently opened optimizer submesh device, if any.
  void closeOptimizerSubmesh();

  // Compiles given mlir program.
  tt_pjrt_status compileMlirProgram(
      const PJRT_Program *mlir_program,
      LoadedExecutableInstance **out_executable,
      const std::unordered_map<std::string, std::string> &compile_options,
      const std::optional<std::vector<int64_t>> &replica_device_ids);

private:
  tt_pjrt_status populateDevices();
  tt_pjrt_status populateMemories();

  // Wrapper method around `tt::runtime::openMeshDevice` that also handles
  // setting fabric config when needed.
  tt::runtime::Device openMeshDevice(const std::vector<uint32_t> &mesh_shape);

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

  // Vector of all device memories visible to the runtime.
  // The host memory is in the m_addressable_host_memory member.
  std::vector<std::unique_ptr<MemoryInstance>> m_addressable_device_memories;

  // MemoryInstance object representing host memory.
  std::unique_ptr<MemoryInstance> m_addressable_host_memory;

  // Vector of raw pointers to all addressable memories, owned by
  // `m_addressable_device_memories` and `m_addressable_host_memory`.
  // Necessary to have to be able to return it in
  // `PJRT_Client_AddressableMemories` API call.
  std::vector<MemoryInstance *> m_addressable_memories_raw;

  // Module builder that compiles program code.
  std::unique_ptr<module_builder::ModuleBuilder> m_module_builder;

  // System descriptor (that TTIR to TTNN backend pipeline needs).
  tt::runtime::SystemDesc m_system_descriptor;

  // TODO: Remove once tt-mlir supports passing the system descriptor object to
  // TTIR to TTNN backend pipeline.
  std::string m_cached_system_descriptor_path;

  // Currently in-use mesh device.
  std::optional<tt::runtime::Device> m_parent_mesh;

  // Optimizer submesh device (created from m_parent_mesh for optimizer passes).
  std::optional<tt::runtime::Device> m_optimizer_submesh;

  // Used to identify the platform.
  const std::string m_platform_name = "tt";

  // We don't have a versioning system for our software stack yet.
  const std::string m_platform_version = "0.0.0";

  // Set of all tracked buffers. Used to materialize all buffers before mesh
  // reshape to prevent data loss.
  std::unordered_set<BufferInstance *> m_tracked_buffers;

  // Mutex protecting m_tracked_buffers.
  mutable std::mutex m_tracked_buffers_mutex;
};

namespace internal {

// Implements PJRT_Client_Create API function.
PJRT_Error *onClientCreate(PJRT_Client_Create_Args *args);

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

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_CLIENT_INSTANCE_H_
