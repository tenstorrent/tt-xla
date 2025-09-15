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

// third-party includes
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/unknown_field_set.h>

// tt-xla includes
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/event_instance.h"
#include "common/pjrt_implementation/memory_instance.h"
#include "common/pjrt_implementation/module_builder/module_builder.h"

namespace tt::pjrt {

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)), m_system_descriptor(nullptr),
      m_module_builder(std::make_unique<module_builder::ModuleBuilder>()) {
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
  tt_pjrt_status device_status = populateDevices();
  if (!tt_pjrt_status_is_ok(device_status)) {
    return *ErrorInstance::makeError(device_status).release();
  }

  tt_pjrt_status memory_status = populateMemories();
  if (!tt_pjrt_status_is_ok(memory_status)) {
    return *ErrorInstance::makeError(memory_status).release();
  }

  return nullptr;
}

void ClientInstance::bindApi(PJRT_Api *api) {
  // TODO(mrakita): Add `PJRT_Client_Create` here too, currently it is
  // polymorphic and defined in `api_bindings.h`.
  api->PJRT_Client_Destroy = internal::onClientDestroy;
  api->PJRT_Client_PlatformName = internal::onClientPlatformName;
  api->PJRT_Client_ProcessIndex = internal::onClientProcessIndex;
  api->PJRT_Client_PlatformVersion = internal::onClientPlatformVersion;
  api->PJRT_Client_Devices = internal::onClientDevices;
  api->PJRT_Client_AddressableDevices = internal::onClientAddressableDevices;
  api->PJRT_Client_LookupDevice = internal::onClientLookupDevice;
  api->PJRT_Client_LookupAddressableDevice =
      internal::onClientLookupAddressableDevice;
  api->PJRT_Client_AddressableMemories = internal::onClientAddressableMemories;
  api->PJRT_Client_Compile = internal::onClientCompile;
  api->PJRT_Client_DefaultDeviceAssignment =
      internal::onClientDefaultDeviceAssignment;
  api->PJRT_Client_BufferFromHostBuffer = internal::onBufferFromHostBuffer;
}

tt_pjrt_status ClientInstance::populateDevices() {
  m_system_descriptor = tt::runtime::getCurrentSystemDesc();
  m_system_descriptor.store(m_cached_system_descriptor_path.data());
  if (std::filesystem::exists(m_cached_system_descriptor_path) == false) {
    DLOG_F(ERROR,
           "Failed to store the system descriptor to the disk using path: %s",
           m_cached_system_descriptor_path.c_str());
    return tt_pjrt_status::kInternal;
  }

  size_t devices_count = tt::runtime::getNumAvailableDevices();
  m_devices.reserve(devices_count);
  m_devices_raw.reserve(devices_count);
  m_addressable_devices_raw.reserve(devices_count);

  for (size_t i = 0; i < devices_count; ++i) {
    int global_device_id = m_system_descriptor->chip_desc_indices()->Get(i);
    int local_device_id = i;

    // For now, just make all devices addressable.
    bool is_addressable = true;

    std::unique_ptr<DeviceInstance> device_instance =
        DeviceInstance::createInstance(
            global_device_id, is_addressable, local_device_id,
            m_system_descriptor->chip_descs()->Get(i)->arch());

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

tt_pjrt_status ClientInstance::populateMemories() {
  m_addressable_host_memory =
      MemoryInstance::createInstance(m_addressable_devices_raw, /*id=*/0,
                                     /*is_host_memory=*/true);
  m_addressable_memories_raw.push_back(m_addressable_host_memory.get());

  for (size_t i = 0; i < m_devices.size(); ++i) {
    // Adding host memory to device addressable memories.
    m_devices[i]->addAddressableMemory(m_addressable_host_memory.get());

    // Adding device memory to device addressable memories.
    std::vector<DeviceInstance *> single_addressable_device = {
        m_addressable_devices_raw[i]};
    std::unique_ptr<MemoryInstance> device_memory =
        MemoryInstance::createInstance(single_addressable_device, /*id=*/i + 1,
                                       /*is_host_memory=*/false);
    m_addressable_memories_raw.push_back(device_memory.get());
    m_devices[i]->addAddressableMemory(device_memory.get());
    m_devices[i]->setDefaultMemory(device_memory.get());
    m_addressable_device_memories.push_back(std::move(device_memory));
  }

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ClientInstance::compileMlirProgram(
    const PJRT_Program *mlir_program, LoadedExecutableInstance **out_executable,
    const std::unordered_map<std::string, std::string> &compile_options) {

  std::string_view mlir_code(mlir_program->code, mlir_program->code_size);

  tt_pjrt_status compile_status = m_module_builder->buildModule(
      mlir_code, m_cached_system_descriptor_path, compile_options);
  if (!tt_pjrt_status_is_ok(compile_status)) {
    return compile_status;
  }

  // TODO(mrakita): Use the VHLO module name from the module builder, if it has
  // a name, otherwise some default string like the current one.
  std::string executable_name = "tt_executable";

  // Get parsed compile options from module builder.
  const module_builder::CompileOptions &parsed_compile_options =
      m_module_builder->getParsedCompileOptions();

  std::shared_ptr<ExecutableImage> executable_image =
      ExecutableImage::createInstance(
          m_module_builder->getFlatbufferBinary(), std::string(mlir_code),
          m_module_builder->getTTIRMlirCode(),
          m_module_builder->getTTNNMlirCode(), std::move(executable_name),
          m_module_builder->getNumPartitions(),
          m_module_builder->getNumReplicas(),
          m_module_builder->getNumDevicesToUtilize(),
          m_module_builder->getDevicesMeshShape(),
          m_module_builder->getInputShardings(),
          m_module_builder->getOutputShardings(),
          m_module_builder->getIsOutputScalar(),
          m_module_builder->getOutputDataTypes(), parsed_compile_options);

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

std::unordered_map<std::string, std::string>
ClientInstance::getCompileOptions(const char *compile_options_data,
                                  size_t compile_options_size) {

  google::protobuf::io::CodedInputStream cis(
      reinterpret_cast<const uint8_t *>(compile_options_data),
      compile_options_size);
  google::protobuf::UnknownFieldSet unknown_fields;

  if (!unknown_fields.MergeFromCodedStream(&cis)) {
    return {};
  }
  std::unordered_map<std::string, std::string> compile_options_map =
      ClientInstance::extractCustomProtobufFields(unknown_fields);

  return compile_options_map;
}

std::unordered_map<std::string, std::string>
ClientInstance::extractCustomProtobufFields(
    const google::protobuf::UnknownFieldSet &unknown_fields) {
  std::unordered_map<std::string, std::string> result;

  // The custom compiler options that are defined in through the jax.jit()
  // function are stored in the field number 7 in the UnknownFieldSet.
  constexpr int kCustomCompilerOptionsFieldNumber = 7;

  for (int i = 0; i < unknown_fields.field_count(); ++i) {
    const google::protobuf::UnknownField &field = unknown_fields.field(i);
    // Currently, we only support the custom compiler options field that are in
    // the form of a dictionary, which is represented as a length_delimited
    // field.
    // TODO: See if we can support other types of custom fields in the future.
    if (field.number() != kCustomCompilerOptionsFieldNumber ||
        field.type() != google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
      continue;
    }

    google::protobuf::UnknownFieldSet custom_field_set;
    google::protobuf::io::CodedInputStream input(
        reinterpret_cast<const uint8_t *>(field.length_delimited().data()),
        field.length_delimited().size());
    custom_field_set.ParseFromCodedStream(&input);

    std::string key;
    std::string value;

    for (int j = 0; j < custom_field_set.field_count(); ++j) {
      const google::protobuf::UnknownField &inner_field =
          custom_field_set.field(j);
      // In the inner field set, first field is the key and second field is the
      // value. We expect both to be length-delimited fields (coming from a
      // dictionary).
      if (inner_field.number() == 1 &&
          inner_field.type() ==
              google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
        key = inner_field.length_delimited();
      } else if (inner_field.number() == 2 &&
                 inner_field.type() ==
                     google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
        google::protobuf::UnknownFieldSet custom_nested_set;
        google::protobuf::io::CodedInputStream nested_input(
            reinterpret_cast<const uint8_t *>(
                inner_field.length_delimited().data()),
            inner_field.length_delimited().size());
        custom_nested_set.ParseFromCodedStream(&nested_input);
        if (custom_nested_set.field_count() == 0 ||
            custom_nested_set.field_count() > 1) {
          // If the nested set has more than one field or is empty, it is not a
          // simple key-value pair of strings, so we skip it for now.
          continue;
        }
        const google::protobuf::UnknownField &value_field =
            custom_nested_set.field(0);
        if (value_field.type() ==
            google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
          value = value_field.length_delimited();
        }
      }
    }

    if (!key.empty() && !value.empty()) {
      result[key] = value;
    }
  }

  return result;
}

namespace internal {

PJRT_Error *onClientDestroy(PJRT_Client_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Destroy");

  delete ClientInstance::unwrap(args->client);

  return nullptr;
}

PJRT_Error *onClientPlatformName(PJRT_Client_PlatformName_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformName");

  ClientInstance *client = ClientInstance::unwrap(args->client);

  // TODO(mrakita): Revisit this implementation.
  args->platform_name = client->cached_platform_name().data();
  args->platform_name_size = client->cached_platform_name().size();

  return nullptr;
}

PJRT_Error *onClientProcessIndex(PJRT_Client_ProcessIndex_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_ProcessIndex");

  args->process_index = ClientInstance::unwrap(args->client)->getProcessIndex();

  return nullptr;
}

PJRT_Error *onClientPlatformVersion(PJRT_Client_PlatformVersion_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformVersion");

  ClientInstance *client = ClientInstance::unwrap(args->client);

  // TODO(mrakita): Revisit this implementation.
  args->platform_version = client->cached_platform_version().data();
  args->platform_version_size = client->cached_platform_version().size();

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

  return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument).release();
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

  return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument).release();
}

PJRT_Error *
onClientAddressableMemories(PJRT_Client_AddressableMemories_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableMemories");

  ClientInstance *client_instance = ClientInstance::unwrap(args->client);
  args->addressable_memories =
      (PJRT_Memory *const *)(client_instance->getAddressableMemoriesRaw()
                                 .data());
  args->num_addressable_memories =
      client_instance->getAddressableMemoriesRaw().size();

  return nullptr;
}

PJRT_Error *onClientCompile(PJRT_Client_Compile_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Compile");
  std::unordered_map<std::string, std::string> compile_options_map =
      ClientInstance::getCompileOptions(args->compile_options,
                                        args->compile_options_size);
  std::string_view program_format(args->program->format,
                                  args->program->format_size);
  if (program_format != module_builder::c_mlir_format_name) {
    DLOG_F(ERROR,
           "Program code format \"%s\" is not supported, only MLIR format is "
           "currently supported",
           args->program->format);
    return *ErrorInstance::makeError(tt_pjrt_status::kUnimplemented).release();
  }

  ClientInstance *client_instance = ClientInstance::unwrap(args->client);

  tt_pjrt_status compile_status = client_instance->compileMlirProgram(
      args->program,
      reinterpret_cast<LoadedExecutableInstance **>(&args->executable),
      compile_options_map);

  return *ErrorInstance::makeError(compile_status).release();
}

PJRT_Error *onClientDefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_DefaultDeviceAssignment");

  // TODO(mrakita): Revisit this implementation.
  for (size_t i = 0; i < args->default_assignment_size; ++i) {
    args->default_assignment[i] = 0;
  }

  return nullptr;
}

PJRT_Error *
onBufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_BufferFromHostBuffer");

  if (args->device_layout &&
      args->device_layout->type != PJRT_Buffer_MemoryLayout_Type_Strides) {
    DLOG_F(ERROR,
           "Copying from host is supported only with strided memory layout");
    return *ErrorInstance::makeError(tt_pjrt_status::kUnimplemented).release();
  }

  if (args->num_byte_strides != 0 && args->num_byte_strides != args->num_dims) {
    DLOG_F(ERROR, "Invalid `num_byte_strides` argument");
    return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument)
                .release();
  }

  MemoryInstance *memory_instance = MemoryInstance::unwrap(args->memory);
  DeviceInstance *device_instance = DeviceInstance::unwrap(args->device);

  // From PJRT specification: "If nullptr, host data will be copied to `device`,
  // otherwise we copy data to `memory`."
  if (memory_instance) {
    if (device_instance && device_instance != memory_instance->getDevice()) {
      DLOG_F(ERROR, "Device set in `device` arg is different from the memory "
                    "space device set in `memory` arg");
      return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument)
                  .release();
    }
    device_instance = memory_instance->getDevice();
  } else {
    memory_instance = device_instance->getDefaultMemory();
  }

  if (!memory_instance) {
    DLOG_F(ERROR, "Memory space is not set either in `memory` arg nor in "
                  "device from `device` arg");
    return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument)
                .release();
  }
  if (memory_instance->isHostMemory()) {
    DLOG_F(ERROR, "We only support creating buffers on device memory");
    return *ErrorInstance::makeError(tt_pjrt_status::kUnimplemented).release();
  }
  if (!device_instance) {
    DLOG_F(ERROR, "Device is not set either in `device` arg nor in device from "
                  "`memory` arg");
    return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument)
                .release();
  }

  std::unique_ptr<BufferInstance> buffer =
      BufferInstance::createInputBufferInstance(args->type, args->dims,
                                                args->num_dims, device_instance,
                                                memory_instance);

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
