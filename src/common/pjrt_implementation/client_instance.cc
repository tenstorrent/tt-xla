// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/client_instance.h"

// c++ standard library includes
#include <cstddef>
#include <filesystem>
#include <string>

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/module_builder.h"
#include "common/pjrt_implementation/buffer_instance.h"

namespace tt::pjrt {

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)), system_descriptor_(nullptr) {
  DLOG_F(LOG_DEBUG, "ClientInstance::ClientInstance");
  module_builder_ = std::make_unique<ModuleBuilder>();
  // TODO: Ensure this name is unique to prevent clashes between multiple
  // clients. Since we plan to remove the need for storing the descriptor on
  // disk soon, we’re keeping it simple for now.
  cached_system_descriptor_path_ =
      std::filesystem::temp_directory_path().concat(
          "/tt_pjrt_system_descriptor");
}

ClientInstance::~ClientInstance() {
  DLOG_F(LOG_DEBUG, "ClientInstance::~ClientInstance");
  std::remove(cached_system_descriptor_path_.data());
}

PJRT_Error *ClientInstance::Initialize() {
  DLOG_F(LOG_DEBUG, "ClientInstance::Initialize");

  return ErrorInstance::MakeError(PopulateDevices());
}

void ClientInstance::bindApi(PJRT_Api *api) {
  // PJRT_Client_Create is polymorphic
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Destroy");
    delete ClientInstance::unwrap(args->client);
    return nullptr;
  };
  api->PJRT_Client_PlatformName =
      +[](PJRT_Client_PlatformName_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformName");
    ClientInstance *client = ClientInstance::unwrap(args->client);
    args->platform_name = client->cached_platform_name().data();
    args->platform_name_size = client->cached_platform_name().size();
    return nullptr;
  };
  api->PJRT_Client_ProcessIndex =
      +[](PJRT_Client_ProcessIndex_Args *args) -> PJRT_Error * {
    args->process_index = 0;
    return nullptr;
  };
  api->PJRT_Client_PlatformVersion =
      +[](PJRT_Client_PlatformVersion_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformVersion");
    ClientInstance *client = ClientInstance::unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices = internal::onClientDevices;
  api->PJRT_Client_AddressableDevices = internal::onClientAddressableDevices;
  api->PJRT_Client_LookupDevice = internal::onClientLookupDevice;
  api->PJRT_Client_LookupAddressableDevice =
      internal::onClientLookupAddressableDevice;
  api->PJRT_Client_AddressableMemories =
      +[](PJRT_Client_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableMemories");
    // return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
    args->num_addressable_memories =
        0; // ClientInstance::unwrap(args->client)->addressable_memories.size();
    args->addressable_memories =
        nullptr; // ClientInstance::unwrap(args->client)->addressable_memories.data();
    return nullptr;
  };
  api->PJRT_Client_Compile =
      +[](PJRT_Client_Compile_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Compile");
    // TODO: It is not great that we only get a client here vs a list of
    // devices to consider (or something). The issue is that systems often
    // have unrelated devices that will not actually be scheduled and those
    // will very naturally have different tuning flags. We therefore have to
    // guess... which is an accident waiting to happen.
    // Looks like what I need is buried in the compile options... need to
    // work on that.
    ClientInstance *client = ClientInstance::unwrap(args->client);
    LoadedExecutableInstance *executable;

    PJRT_Error *error = client->Compile(args->program, &executable);
    if (error)
      return error;
    args->executable = *executable;
    return nullptr;
  };
  api->PJRT_Client_DefaultDeviceAssignment =
      +[](PJRT_Client_DefaultDeviceAssignment_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_DefaultDeviceAssignment");
    // TODO: Something sensible.
    for (size_t i = 0; i < args->default_assignment_size; ++i) {
      args->default_assignment[i] = 0;
    }
    return nullptr;
  };
  api->PJRT_Client_BufferFromHostBuffer = internal::onBufferFromHostBuffer;
}

tt_pjrt_status ClientInstance::PopulateDevices() {
  DLOG_F(LOG_DEBUG, "ClientInstance::PopulateDevices");

  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();

  system_descriptor_ = system_desc;
  system_descriptor_.store(cached_system_descriptor_path_.data());
  if (std::filesystem::exists(cached_system_descriptor_path_) == false) {
    DLOG_F(ERROR,
           "Failed to store the system descriptor to the disk using path: %s",
           cached_system_descriptor_path_.c_str());
    return tt_pjrt_status::kInternal;
  }

  size_t devices_count = chip_ids.size();
  m_devices.reserve(devices_count);
  m_devices_raw.reserve(devices_count);
  m_addressable_devices_raw.reserve(devices_count);

  for (size_t i = 0; i < devices_count; ++i) {
    int global_device_id = chip_ids[i];
    int local_device_id = i;

    // For now, just make all devices addressable.
    bool is_addressable = true;

    std::unique_ptr<DeviceInstance> device_instance =
        DeviceInstance::createInstance(
            global_device_id, is_addressable, local_device_id,
            system_desc->chip_descs()->Get(i)->arch());

    m_devices_raw.push_back(device_instance.get());
    if (is_addressable) {
      m_addressable_devices_raw.push_back(device_instance.get());
    }

    m_devices.emplace_back(std::move(device_instance));
  }

  if (m_addressable_devices_raw.empty()) {
    DLOG_F(ERROR, "Found no addressable devices in the system");
    return tt_pjrt_status::kInternal;
  }

  return tt_pjrt_status::kSuccess;
}

PJRT_Error *ClientInstance::Compile(const PJRT_Program *program,
                                    LoadedExecutableInstance **out_executable) {
  DLOG_F(LOG_DEBUG, "ClientInstance::Compile");

  std::string_view code(program->code, program->code_size);
  std::string_view format(program->format, program->format_size);

  tt_pjrt_status status = module_builder_->buildModule(
      code, format, cached_system_descriptor_path_);
  if (!tt_pjrt_status_is_ok(status)) {
    return ErrorInstance::MakeError(status);
  }

  auto executable = std::make_unique<LoadedExecutableInstance>(
      new ExecutableImage(module_builder_->getBinary(),
                          std::string(program->code, program->code_size),
                          module_builder_->getInputShardings(),
                          module_builder_->getOutputShardings(),
                          module_builder_->getMeshShape(),
                          module_builder_->getIsOutputScalar()),
      m_addressable_devices_raw, module_builder_->getNumDevicesToUtilize());
  *out_executable = executable.release();
  return nullptr;
}

std::tuple<uint64_t, uint64_t> ClientInstance::AdvanceTimeline() {
  uint64_t current = execution_timeline_;
  uint64_t next = current + 1;
  execution_timeline_ = next;
  return std::make_tuple(current, next);
}

namespace internal {

PJRT_Error *onClientDevices(PJRT_Client_Devices_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Devices");

  const std::vector<DeviceInstance *> &devices_raw =
      ClientInstance::unwrap(args->client)->getDevicesRaw();

  args->devices = reinterpret_cast<PJRT_Device *const *>(devices_raw.data());
  args->num_devices = devices_raw.size();

  return nullptr;
}

PJRT_Error *
onClientAddressableDevices(PJRT_Client_AddressableDevices_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableDevices");

  const std::vector<DeviceInstance *> &addressable_devices_raw =
      ClientInstance::unwrap(args->client)->getAddressableDevicesRaw();

  args->addressable_devices =
      reinterpret_cast<PJRT_Device *const *>(addressable_devices_raw.data());
  args->num_addressable_devices = addressable_devices_raw.size();

  return nullptr;
}

PJRT_Error *onClientLookupDevice(PJRT_Client_LookupDevice_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_LookupDevice");

  ClientInstance *client_instance = ClientInstance::unwrap(args->client);
  for (DeviceInstance *device_instance : client_instance->getDevicesRaw()) {
    if (device_instance->getGlobalDeviceId() == args->id) {
      args->device = *device_instance;
      return nullptr;
    }
  }

  DLOG_F(ERROR, "Client device lookup failed for device with ID: %d", args->id);

  return ErrorInstance::MakeError(tt_pjrt_status::kInvalidArgument);
}

PJRT_Error *onClientLookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_LookupAddressableDevice");

  ClientInstance *client_instance = ClientInstance::unwrap(args->client);
  for (DeviceInstance *device_instance :
       client_instance->getAddressableDevicesRaw()) {
    if (device_instance->getLocalDeviceId() == args->local_hardware_id) {
      args->addressable_device = *device_instance;
      return nullptr;
    }
  }

  DLOG_F(ERROR,
         "Client addressable device lookup failed for device with local ID: %d",
         args->id);

  return ErrorInstance::MakeError(tt_pjrt_status::kInvalidArgument);
}

PJRT_Error *
onBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_BufferFromHostBuffer");

  if (args->memory) {
    DLOG_F(ERROR, "Copying to custom memory is not supported");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  }

  if (args->device_layout &&
      args->device_layout->type != PJRT_Buffer_MemoryLayout_Type_Strides) {
    DLOG_F(ERROR, "Only strided memory layout is supported");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  }

  if (args->num_byte_strides != 0 && args->num_byte_strides != args->num_dims) {
    DLOG_F(ERROR, "Invalid `num_byte_strides` argument");
    return ErrorInstance::MakeError(tt_pjrt_status::kInvalidArgument);
  }

  std::unique_ptr<BufferInstance> buffer =
      BufferInstance::createInputBufferInstance(
          args->type, args->dims, args->num_dims,
          DeviceInstance::unwrap(args->device));

  buffer->copyFromHost(
      args->data, args->type, args->dims, args->num_dims, args->byte_strides,
      args->num_byte_strides, args->host_buffer_semantics,
      reinterpret_cast<EventInstance **>(&args->done_with_host_buffer));

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling PJRT_Buffer_Destroy on the buffer.
  args->buffer = *buffer.release();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
