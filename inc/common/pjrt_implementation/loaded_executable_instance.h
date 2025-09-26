// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/executable_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

class ClientInstance;

// Represents `PJRT_LoadedExecutable` structure and the functionality around it.
// It is the in-memory loaded executable which is ready for input arguments to
// execute.
class LoadedExecutableInstance {
public:
  // Creates new loaded executable instance from the executable image.
  static std::unique_ptr<LoadedExecutableInstance>
  createInstance(std::shared_ptr<ExecutableImage> executable_image,
                 std::vector<DeviceInstance *> &&addressable_devices,
                 ClientInstance *client_instance);

  // Binds PJRT API functions implementation related to PJRT_LoadedExecutable
  // structure.
  static void bindApi(PJRT_Api *api);

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
  tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args);

private:
  // Creates loaded executable instance from the executable image.
  LoadedExecutableInstance(
      std::shared_ptr<ExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance)
      : m_executable_image(std::move(executable_image)),
        m_addressable_devices(addressable_devices), m_deleted(false),
        m_client_instance(client_instance) {}

  // Gets the appropriate runtime device from the `ClientInstance`.
  std::optional<tt::runtime::Device>
  getOrCreateMeshDevice(PJRT_Buffer *const *const *argument_lists,
                        size_t num_args, size_t num_devices,
                        PJRT_Device *pjrt_device);

  // Collects device ids from the addressable devices.
  std::unordered_set<int>
  getDeviceIds(PJRT_Buffer *const *const *argument_lists, size_t num_args,
               size_t num_devices);

  // Gets input runtime tensors from the arguments' buffers and converts them to
  // desired layout determined from the compiled graph.
  tt_pjrt_status
  getInputRuntimeTensors(PJRT_Buffer *const *const *argument_lists,
                         size_t num_args, size_t num_devices,
                         const tt::runtime::Device &runtime_device,
                         std::uint32_t program_index,
                         std::vector<tt::runtime::Tensor> &input_tensors);

  // Fills strategy map from sharding configuration.
  // TODO: This function might be better suited living in the tt-mlir
  // repository. https://github.com/tenstorrent/tt-xla/issues/374
  static mlir::FailureOr<std::unordered_map<std::string, std::string>>
  fillStrategyMapFromSharding(
      const mlir::tt::sharding_utils::MeshSharding &meshSharding,
      size_t num_devices);

  // Either returns single tensor or creates multi-device host tensor from arg
  // tensors, depending on the strategy.
  tt::runtime::Tensor getTensorFromStrategy(
      const std::vector<tt::runtime::Tensor> &arg_tensors,
      const std::unordered_map<std::string, std::string> &strategy);

  // Converts input tensor to desired layout. This might move it on device.
  tt::runtime::Tensor
  convertTensorLayout(tt::runtime::Tensor input_tensor,
                      std::uint32_t program_index, size_t arg_index,
                      const tt::runtime::Device &runtime_device);

  // Untilizes output tensors and transfers them from device to host.
  tt_pjrt_status untilizeToHost(
      const std::vector<tt::runtime::Tensor> &output_tensors,
      size_t num_devices,
      std::vector<std::vector<tt::runtime::Tensor>> &untilized_output_tensors);

  // Fills the output lists of the PJRT API with the outputs of tt runtime
  // execution.
  void fillPJRTOutputLists(
      const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs,
      size_t num_devices, PJRT_Buffer **const *output_lists,
      const std::vector<PJRT_Buffer_Type> &expected_output_data_types);

  // Returns the shape of the output on the specified index.
  std::vector<std::uint32_t> getOutputShape(size_t output_index);

  // Helper function to get expected layout for a specific program and argument
  // index.
  tt::runtime::Layout getExpectedLayout(std::uint32_t program_index,
                                        size_t arg_index);

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

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
