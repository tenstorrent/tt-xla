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

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/module_builder.h"
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/event_instance.h"

namespace tt::pjrt {

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)), m_system_descriptor(nullptr),
      m_module_builder(std::make_unique<ModuleBuilder>()) {
  DLOG_F(LOG_DEBUG, "ClientInstance::ClientInstance");

  // TODO(mrakita): Add support for multi-process environment. Process index is
  // always 0 in single-process settings.
  m_process_index = 0;

  // TODO: Ensure this name is unique to prevent clashes between multiple
  // clients. Since we plan to remove the need for storing the descriptor on
  // disk soon, we’re keeping it simple for now.
  m_cached_system_descriptor_path =
      std::filesystem::temp_directory_path().concat(
          "/tt_pjrt_system_descriptor");
}

ClientInstance::~ClientInstance() {
  DLOG_F(LOG_DEBUG, "ClientInstance::~ClientInstance");

  std::remove(m_cached_system_descriptor_path.data());
}

PJRT_Error *ClientInstance::Initialize() {
  DLOG_F(LOG_DEBUG, "ClientInstance::Initialize");

  return ErrorInstance::makeError(populateDevices());
}

void ClientInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Client_ProcessIndex = internal::onClientProcessIndex;
  api->PJRT_Client_Devices = internal::onClientDevices;
  api->PJRT_Client_AddressableDevices = internal::onClientAddressableDevices;
  api->PJRT_Client_LookupDevice = internal::onClientLookupDevice;
  api->PJRT_Client_LookupAddressableDevice =
      internal::onClientLookupAddressableDevice;
  api->PJRT_Client_Compile = internal::onClientCompile;
  api->PJRT_Client_BufferFromHostBuffer = internal::onBufferFromHostBuffer;

  // TODO(mrakita): Move these below to internal too and revisit implementation.

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
  api->PJRT_Client_PlatformVersion =
      +[](PJRT_Client_PlatformVersion_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformVersion");
    ClientInstance *client = ClientInstance::unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_AddressableMemories =
      +[](PJRT_Client_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableMemories");
    // return ErrorInstance::makeError(tt_pjrt_status::kUnimplemented);
    args->num_addressable_memories =
        0; // ClientInstance::unwrap(args->client)->addressable_memories.size();
    args->addressable_memories =
        nullptr; // ClientInstance::unwrap(args->client)->addressable_memories.data();
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
}

tt_pjrt_status ClientInstance::populateDevices() {
  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();

  m_system_descriptor = system_desc;
  m_system_descriptor.store(m_cached_system_descriptor_path.data());
  if (std::filesystem::exists(m_cached_system_descriptor_path) == false) {
    DLOG_F(ERROR,
           "Failed to store the system descriptor to the disk using path: %s",
           m_cached_system_descriptor_path.c_str());
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
      device_instance->setProcessIndex(m_process_index);
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

tt_pjrt_status
ClientInstance::compileMlirProgram(const PJRT_Program *mlir_program,
                                   LoadedExecutableInstance **out_executable) {

  std::string_view mlir_code(mlir_program->code, mlir_program->code_size);

  tt_pjrt_status compile_status =
      m_module_builder->buildModule(mlir_code, m_cached_system_descriptor_path);
  if (!tt_pjrt_status_is_ok(compile_status)) {
    return compile_status;
  }

  // TODO(mrakita): Decide which module to pass here. If this is going to be
  // used only for debugging then we might want TTIR or TTNN module, but if it
  // is going to be used for the `PJRT_Executable_DeserializeAndLoad` to
  // recompile the flatbuffer then we need either original program code or
  // VHLO/SHLO module. Passing original program code for now.
  std::string optimized_mlir_code(mlir_code);

  // TODO(mrakita): Use the VHLO module name from the module builder, if it has
  // a name, otherwise some default string like the current one.
  std::string executable_name = "tt_executable";

  std::shared_ptr<ExecutableImage> executable_image =
      ExecutableImage::createInstance(
          m_module_builder->getFlatbufferBinary(),
          std::move(optimized_mlir_code), std::move(executable_name),
          m_module_builder->getNumPartitions(),
          m_module_builder->getNumReplicas(),
          m_module_builder->getNumDevicesToUtilize(),
          m_module_builder->getDevicesMeshShape(),
          m_module_builder->getInputShardings(),
          m_module_builder->getOutputShardings(),
          m_module_builder->getIsOutputScalar());

  // TODO(mrakita): Currently there is no way to determine addressable devices
  // from the mlir code. XLA parses device assignment from the `compile_options`
  // arg, but that field is a serialized protobuf of `xla::CompileOptions` which
  // we cannot deserialize easily without linking whole XLA. Passing a subset of
  // first `num_devices_to_utilize` client's addressable devices for now, but
  // this will lead to errors if buffers are put on different devices than
  // those. https://github.com/openxla/xla/issues/24990
  std::vector<DeviceInstance *> addressable_devices(
      m_addressable_devices_raw.begin(),
      m_addressable_devices_raw.begin() +
          m_module_builder->getNumDevicesToUtilize());

  std::unique_ptr<LoadedExecutableInstance> executable =
      LoadedExecutableInstance::createInstance(executable_image,
                                               std::move(addressable_devices));

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_LoadedExecutable_Destroy` on the executable.
  *out_executable = executable.release();

  return tt_pjrt_status::kSuccess;
}

namespace internal {

PJRT_Error *onClientProcessIndex(PJRT_Client_ProcessIndex_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_ProcessIndex");

  args->process_index = ClientInstance::unwrap(args->client)->getProcessIndex();

  return nullptr;
}

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

  return ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument);
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
         args->local_hardware_id);

  return ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument);
}

PJRT_Error *onClientCompile(PJRT_Client_Compile_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Compile");

  std::string_view program_format(args->program->format,
                                  args->program->format_size);
  if (program_format != ModuleBuilder::c_mlir_format_name) {
    DLOG_F(ERROR,
           "Program code format \"%s\" is not supported, only MLIR format is "
           "currently supported",
           args->program->format);
    return ErrorInstance::makeError(tt_pjrt_status::kUnimplemented);
  }

  ClientInstance *client_instance = ClientInstance::unwrap(args->client);

  tt_pjrt_status compile_status = client_instance->compileMlirProgram(
      args->program,
      reinterpret_cast<LoadedExecutableInstance **>(&args->executable));

  return ErrorInstance::makeError(compile_status);
}

PJRT_Error *
onBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_BufferFromHostBuffer");

  if (args->memory) {
    DLOG_F(ERROR, "Copying to custom memory is not supported");
    return ErrorInstance::makeError(tt_pjrt_status::kUnimplemented);
  }

  if (args->device_layout &&
      args->device_layout->type != PJRT_Buffer_MemoryLayout_Type_Strides) {
    DLOG_F(ERROR,
           "Copying from host is supported only with strided memory layout");
    return ErrorInstance::makeError(tt_pjrt_status::kUnimplemented);
  }

  if (args->num_byte_strides != 0 && args->num_byte_strides != args->num_dims) {
    DLOG_F(ERROR, "Invalid `num_byte_strides` argument");
    return ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument);
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
  // responsible for calling `PJRT_Buffer_Destroy` on the buffer.
  args->buffer = *buffer.release();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
