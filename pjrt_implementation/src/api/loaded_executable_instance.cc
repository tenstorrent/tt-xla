// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/loaded_executable_instance.h"
#include "tt/runtime/types.h"

// c++ standard library includes
#include <cassert>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_set>

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/runtime.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/device_instance.h"
#include "api/error_instance.h"
#include "api/executable_image.h"
#include "api/executable_instance.h"
#include "utils/logging.h"

namespace tt::pjrt {

// Clears program cache on instance destroy.
LoadedExecutableInstance::~LoadedExecutableInstance() {
  using namespace tt::runtime;

  const std::optional<Device> &device = m_client_instance->parentMesh();
  if (device && getCurrentHostRuntime() == HostRuntime::Local &&
      isProgramCacheEnabled(*device)) {
    DLOG_F(LOG_DEBUG, "Clearing program cache.");
    clearProgramCache(*device);
  }
}

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

void LoadedExecutableInstance::dumpInputs(
    const std::vector<tt::runtime::Tensor> &input_tensors) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::dumpInputs");

  assert(m_executable_image->getCompileOptions().export_path.has_value() &&
         "Export path must be set when dumping inputs");

  std::filesystem::path dump_dir =
      std::filesystem::path(
          m_executable_image->getCompileOptions().export_path.value()) /
      "tensors";
  std::filesystem::create_directories(dump_dir);

  for (int i = 0; i < input_tensors.size(); ++i) {
    std::string filename = "arg" + std::to_string(i) + ".tensorbin";
    std::filesystem::path filepath = dump_dir / filename;

    tt::runtime::dumpTensor(input_tensors[i], filepath.string());
  }
}

std::optional<tt::runtime::Device>
LoadedExecutableInstance::getOrCreateMeshDevice(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices, PJRT_Device *pjrt_device) {
  std::unordered_set<int> device_ids =
      getDeviceIds(argument_lists, num_args, num_devices);

  const std::vector<std::uint32_t> &devices_mesh_shape =
      m_executable_image->getDevicesMeshShape();
  size_t mesh_shape_num_devices = static_cast<size_t>(
      std::accumulate(devices_mesh_shape.begin(), devices_mesh_shape.end(), 1,
                      std::multiplies<std::uint32_t>{}));

  if (device_ids.size() != mesh_shape_num_devices) {
    DLOG_F(ERROR,
           "Input buffers are placed on a different number of devices (%zu) "
           "than in the mesh shape estimated by the compiler (%zu)",
           device_ids.size(), mesh_shape_num_devices);
    return std::nullopt;
  }

  DeviceInstance *device_instance = DeviceInstance::unwrap(pjrt_device);
  if (device_instance &&
      !(device_ids.size() == 1 &&
        *device_ids.begin() == device_instance->getGlobalDeviceId())) {
    DLOG_F(ERROR, "Input buffers are placed on a different device than the one "
                  "specified in the execute_device argument");
    return std::nullopt;
  }

  // TODO(mrakita): Currently runtime doesn't allow us to open specific devices
  // subset, we can only open contiguous subset of devices starting from some
  // offset. We need to keep track of opened devices in Client and map the
  // buffers devices to these devices.
  // https://github.com/tenstorrent/tt-xla/issues/502

  return m_client_instance->getOrCreateMeshDevice(devices_mesh_shape);
}

std::unordered_set<int> LoadedExecutableInstance::getDeviceIds(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices) {
  std::unordered_set<int> device_ids;

  for (size_t device_index = 0; num_args && device_index < num_devices;
       device_index++) {
    const BufferInstance *buffer =
        BufferInstance::unwrap(argument_lists[device_index][0]);
    int64_t buffer_device_id = buffer->getDevice()->getGlobalDeviceId();
    device_ids.emplace(buffer_device_id);
  }

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit. Maybe use `execute_device` from the args?
  if (device_ids.size() == 0) {
    assert(!m_addressable_devices.empty());
    device_ids.emplace(m_addressable_devices.front()->getGlobalDeviceId());
  }

  return device_ids;
}

tt_pjrt_status LoadedExecutableInstance::getInputRuntimeTensors(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices, const tt::runtime::Device &runtime_device,
    std::uint32_t program_index,
    std::vector<tt::runtime::Tensor> &input_tensors) {
  for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
    std::vector<BufferInstance *> arg_buffers;
    arg_buffers.reserve(num_devices);

    for (size_t device_index = 0; device_index < num_devices; ++device_index) {
      BufferInstance *buffer =
          BufferInstance::unwrap(argument_lists[device_index][arg_index]);
      arg_buffers.push_back(buffer);
    }

    std::optional<tt::runtime::Tensor> prepared_tensor = prepareInputTensor(
        arg_buffers, runtime_device, num_devices, program_index, arg_index);

    if (!prepared_tensor.has_value()) {
      // Error is reported in `prepareInputTensor`.
      return tt_pjrt_status::kInternal;
    }

    input_tensors.push_back(*prepared_tensor);

    // Safety check to ensure no input tensor can be accidentally
    //  deallocated during execution, as it may be reused in a future graph.
    if (!tt::runtime::getTensorRetain(*prepared_tensor)) {
      DLOG_F(ERROR, "Prepared input tensor should have retain=true or it may "
                    "be deallocated during execution.");
      return tt_pjrt_status::kInternal;
    }
    if (!tt::runtime::isTensorAllocated(*prepared_tensor)) {
      DLOG_F(ERROR, "Prepared input tensor is not allocated on device. This "
                    "means it was deallocated by a previous operation.");
      return tt_pjrt_status::kInternal;
    }
  }
  return tt_pjrt_status::kSuccess;
}

mlir::FailureOr<std::unordered_map<std::string, std::string>>
LoadedExecutableInstance::fillStrategyMapFromSharding(
    const mlir::tt::sharding_utils::MeshSharding &meshSharding,
    size_t num_devices) {
  std::unordered_map<std::string, std::string> strategy;
  mlir::tt::ttcore::MeshShardType meshType = meshSharding.getShardType();
  if (meshType == mlir::tt::ttcore::MeshShardType::Replicate) {
    // If there is only one device, the output will be replicated, but there is
    // no need to replicate.
    if (num_devices == 1) {
      strategy["strategy"] = "identity";
    } else {
      strategy["strategy"] = "replicate";
      strategy["replication_factor"] = std::to_string(num_devices);
    }
  } else if (meshType == mlir::tt::ttcore::MeshShardType::Devices) {
    llvm::SmallVector<int64_t> mesh_shape_data = meshSharding.getMeshShape();
    assert(mesh_shape_data.size() <= 2 && mesh_shape_data.size() >= 1);
    if (mesh_shape_data.size() == 1) {
      strategy["strategy"] = "shard";
      strategy["shard_dim"] = std::to_string(mesh_shape_data[0]);
    }
    if (mesh_shape_data.size() == 2) {
      strategy["strategy"] = "shard_2d";
      strategy["mesh_shape_y"] = std::to_string(mesh_shape_data[0]);
      strategy["mesh_shape_x"] = std::to_string(mesh_shape_data[1]);
    }
  } else if (meshType == mlir::tt::ttcore::MeshShardType::Identity) {
    strategy["strategy"] = "identity";
  } else {
    return mlir::failure();
  }
  return strategy;
}

// TODO: We are using std::maps with strings as that is the way it is defined in
// the tt::runtime, instead of a more structured approach with structs and/or
// enums. See issue: https://github.com/tenstorrent/tt-mlir/issues/2513
tt::runtime::Tensor LoadedExecutableInstance::getTensorFromStrategy(
    const std::vector<BufferInstance *> &arg_buffers,
    const std::unordered_map<std::string, std::string> &strategy) {
  if (strategy.at("strategy") == "identity") {
    std::optional<tt::runtime::Tensor> host_runtime_tensor =
        arg_buffers.front()->getHostRuntimeTensor();
    assert(
        host_runtime_tensor.has_value() &&
        "Host tensor should be available in the buffer instance at this point");
    return *host_runtime_tensor;
  }

  std::vector<tt::runtime::Tensor> runtime_tensor_shards;
  runtime_tensor_shards.reserve(arg_buffers.size());
  for (const BufferInstance *buffer : arg_buffers) {
    std::optional<tt::runtime::Tensor> host_runtime_tensor =
        buffer->getHostRuntimeTensor();
    assert(
        host_runtime_tensor.has_value() &&
        "Host tensor should be available in the buffer instance at this point");
    runtime_tensor_shards.push_back(*host_runtime_tensor);
  }

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceHostTensor(
      runtime_tensor_shards, strategy,
      m_executable_image->getDevicesMeshShape());
  tt::runtime::setTensorRetain(tensor, /*retain=*/true);

  return tensor;
}

namespace internal {

PJRT_Error *
onLoadedExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Destroy");

  delete LoadedExecutableInstance::unwrap(args->executable);

  return nullptr;
}

PJRT_Error *onLoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args *args) {
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
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");

  LoadedExecutableInstance::unwrap(args->executable)->releaseResources();

  return nullptr;
}

PJRT_Error *
onLoadedExecutableIsDeleted(PJRT_LoadedExecutable_IsDeleted_Args *args) {
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted");

  args->is_deleted =
      LoadedExecutableInstance::unwrap(args->executable)->isDeleted();

  return nullptr;
}

PJRT_Error *
onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");

  tt_pjrt_status status =
      LoadedExecutableInstance::unwrap(args->executable)->execute(args);

  return *ErrorInstance::makeError(status).release();
}

} // namespace internal

} // namespace tt::pjrt
