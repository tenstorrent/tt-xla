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

#include <string>

#include "common/pjrt_implementation/utils.h"

namespace tt::pjrt {

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)) {
  DLOG_F(LOG_DEBUG, "ClientInstance::ClientInstance");
  module_builder_ = std::make_unique<ModuleBuilder>();
}

ClientInstance::~ClientInstance() {
  DLOG_F(LOG_DEBUG, "ClientInstance::~ClientInstance");
}

PJRT_Error *ClientInstance::Initialize() {
  DLOG_F(LOG_DEBUG, "ClientInstance::Initialize");

  tt_pjrt_status status = PopulateDevices();
  if (!tt_pjrt_status_is_ok(status)) {
    return ErrorInstance::MakeError(status);
  }

  return nullptr;
}

void ClientInstance::BindApi(PJRT_Api *api) {
  // PJRT_Client_Create is polymorphic
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Destroy");
    delete ClientInstance::Unwrap(args->client);
    return nullptr;
  };
  api->PJRT_Client_PlatformName =
      +[](PJRT_Client_PlatformName_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformName");
    ClientInstance *client = ClientInstance::Unwrap(args->client);
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
    ClientInstance *client = ClientInstance::Unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices =
      +[](PJRT_Client_Devices_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Devices");
    const std::vector<DeviceInstance *> &devices =
        ClientInstance::Unwrap(args->client)->devices();
    args->devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableDevices_Args");
    const std::vector<DeviceInstance *> &devices =
        ClientInstance::Unwrap(args->client)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_LookupDevice =
      +[](PJRT_Client_LookupDevice_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_LookupDevice_Args");
    const std::vector<DeviceInstance *> &devices =
        ClientInstance::Unwrap(args->client)->devices();
    size_t id_as_size = args->id;
    if (id_as_size >= devices.size()) {
      return ErrorInstance::MakeError(tt_pjrt_status::kOutOfRange);
    }
    args->device = *devices[id_as_size];
    return nullptr;
  };
  api->PJRT_Client_AddressableMemories =
      +[](PJRT_Client_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableMemories");
    // return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
    args->num_addressable_memories =
        0; // ClientInstance::Unwrap(args->client)->addressable_memories.size();
    args->addressable_memories =
        nullptr; // ClientInstance::Unwrap(args->client)->addressable_memories.data();
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
    ClientInstance *client = ClientInstance::Unwrap(args->client);
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
  api->PJRT_Client_BufferFromHostBuffer =
      +[](PJRT_Client_BufferFromHostBuffer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_BufferFromHostBuffer");
    tt_pjrt_status status =
        DeviceInstance::Unwrap(args->device)
            ->HostBufferToDevice(
                args->data, args->type, args->dims, args->num_dims,
                args->byte_strides, args->num_byte_strides,
                args->host_buffer_semantics,
                reinterpret_cast<EventInstance **>(
                    &args->done_with_host_buffer),
                reinterpret_cast<BufferInstance **>(&args->buffer));
    return ErrorInstance::MakeError(status);
  };
  api->PJRT_LoadedExecutable_Fingerprint =
      +[](PJRT_LoadedExecutable_Fingerprint_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_LoadedExecutable_Fingerprint");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
}

tt_pjrt_status ClientInstance::PopulateDevices() {
  DLOG_F(LOG_DEBUG, "ClientInstance::PopulateDevices");
  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
  int device_info_count_ =
      1; // TODO: revert to chip_ids.size(); once
         // https://github.com/tenstorrent/tt-xla/issues/9 is fixed

  devices_.resize(device_info_count_);
  for (size_t i = 0; i < device_info_count_; ++i) {
    devices_[i] = new DeviceInstance(i, *this);
  }

  // For now, just make all devices addressable.
  addressable_devices_.reserve(devices_.size());
  for (DeviceInstance *device : devices_) {
    addressable_devices_.push_back(device);
  }
  return tt_pjrt_status::kSuccess;
}

PJRT_Error *ClientInstance::Compile(const PJRT_Program *program,
                                    LoadedExecutableInstance **out_executable) {
  DLOG_F(LOG_DEBUG, "ClientInstance::Compile");

  std::string_view code(program->code, program->code_size);
  std::string_view format(program->format, program->format_size);

  tt_pjrt_status status = module_builder_->buildModule(code, format);
  if (!tt_pjrt_status_is_ok(status)) {
    return ErrorInstance::MakeError(status);
  }

  auto executable = std::make_unique<LoadedExecutableInstance>(
      *this,
      new ExecutableImage(module_builder_->getBinary(),
                          std::string(program->code, program->code_size),
                          module_builder_->getNumInputs(),
                          module_builder_->getNumOutputs()),
      addressable_devices_);
  *out_executable = executable.release();

  return nullptr;
}

std::tuple<uint64_t, uint64_t> ClientInstance::AdvanceTimeline() {
  uint64_t current = execution_timeline_;
  uint64_t next = current + 1;
  execution_timeline_ = next;
  return std::make_tuple(current, next);
}

} // namespace tt::pjrt