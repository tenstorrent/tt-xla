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
#include <optional>
#include <sstream>
#include <unordered_map>

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

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
#include "common/pjrt_implementation/utils.h"

namespace tt::pjrt {

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)), m_system_descriptor(nullptr),
      m_module_builder(std::make_unique<module_builder::ModuleBuilder>()),
      m_parent_mesh(std::nullopt) {
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

  if (m_parent_mesh.has_value()) {
    tt::runtime::closeMeshDevice(*m_parent_mesh);
  }

  clearTensorCache();
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

  m_parent_mesh =
      getOrCreateMeshDevice({1, static_cast<uint32_t>(m_devices.size())});

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
      mlir_code, m_cached_system_descriptor_path, compile_options, this);

  if (!tt_pjrt_status_is_ok(compile_status)) {
    return compile_status;
  }

  // TODO(mrakita): Use the VHLO module name from the module builder, if it has
  // a name, otherwise some default string like the current one.
  std::string executable_name = "tt_executable";

  // Parse compile options for fingerprint generation
  module_builder::CompileOptions parsed_compile_options =
      module_builder::CompileOptions::parse(compile_options);

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
          m_module_builder->getOutputDataTypes(),
          std::move(parsed_compile_options));

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
      LoadedExecutableInstance::createInstance(
          executable_image, std::move(addressable_devices), this);

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_LoadedExecutable_Destroy` on the executable.
  *out_executable = executable.release();

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status ClientInstance::getCompileOptions(
    const char *compile_options_data, size_t compile_options_size,
    std::unordered_map<std::string, std::string> &out_compile_options) {

  google::protobuf::io::CodedInputStream cis(
      reinterpret_cast<const uint8_t *>(compile_options_data),
      compile_options_size);
  google::protobuf::UnknownFieldSet unknown_fields;

  if (!unknown_fields.MergeFromCodedStream(&cis)) {
    DLOG_F(ERROR, "Failed to parse the unknown fields set from the compile "
                  "options protobuf data");
    return tt_pjrt_status::kInternal;
  }

  return ClientInstance::extractCustomProtobufFields(unknown_fields,
                                                     out_compile_options);
}

tt_pjrt_status ClientInstance::extractCustomProtobufFields(
    const google::protobuf::UnknownFieldSet &unknown_fields,
    std::unordered_map<std::string, std::string> &out_compile_options) {

  // The custom compiler options that are passed in through the `jax.jit()`
  // or `torch_xla.set_custom_compile_options()` are stored in the field
  // number 7 in the UnknownFieldSet, which is defined as:
  // `env_option_overrides (map<string, OptionOverrideProto>)`.
  // Each map entry is a nested message with key/value inside.
  constexpr int kCustomCompilerOptionsFieldNumber = 7;

  // Field number corresponding to the string key of the compile options map
  // entry.
  constexpr int kMapKeyFieldNumber = 1;

  // Field number corresponding to the OptionOverrideProto value of the compile
  // options map entry.
  constexpr int kMapValueFieldNumber = 2;

  for (int i = 0; i < unknown_fields.field_count(); ++i) {
    const google::protobuf::UnknownField &field = unknown_fields.field(i);

    // Currently, we only support custom compiler options serialized in the
    // `kCustomCompilerOptionsFieldNumber` field. In case we encounter
    // options being serialized into some other field we will need to update
    // this to support them.
    if (field.number() != kCustomCompilerOptionsFieldNumber ||
        field.type() != google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
      continue;
    }

    const std::string &bytes = field.length_delimited();
    google::protobuf::io::CodedInputStream cis(
        reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());

    google::protobuf::UnknownFieldSet map_entry_fields;
    if (!map_entry_fields.MergeFromCodedStream(&cis)) {
      DLOG_F(ERROR, "Failed to parse the map entry fields from the custom "
                    "compile options protobuf data");
      return tt_pjrt_status::kInternal;
    }

    std::string key;
    std::string value;

    for (int j = 0; j < map_entry_fields.field_count(); ++j) {
      const google::protobuf::UnknownField &entry_field =
          map_entry_fields.field(j);
      // In the inner field set, first field is the key and second field is the
      // value. We expect both to be length-delimited fields (coming from a
      // dictionary).
      if (entry_field.number() == kMapKeyFieldNumber &&
          entry_field.type() ==
              google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
        key = entry_field.length_delimited();
      } else if (entry_field.number() == kMapValueFieldNumber &&
                 entry_field.type() ==
                     google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
        const std::string &override_bytes = entry_field.length_delimited();
        google::protobuf::io::CodedInputStream override_stream(
            reinterpret_cast<const uint8_t *>(override_bytes.data()),
            override_bytes.size());

        google::protobuf::UnknownFieldSet value_fields;
        if (!value_fields.MergeFromCodedStream(&override_stream)) {
          DLOG_F(ERROR, "Failed to parse the map entry field value from the "
                        "custom compile options protobuf data");
          return tt_pjrt_status::kInternal;
        }

        // https://github.com/openxla/xla/blob/main/xla/pjrt/proto/compile_options.proto#L151C1-L158C2
        // Field numbers and types for OptionOverrideProto
        // message OptionOverrideProto {
        //   oneof value {
        //     string string_field = 1;
        //     bool bool_field = 2;
        //     int64 int_field = 3;
        //     double double_field = 4;
        //   }
        // }
        if (value_fields.field_count() != 1) {
          DLOG_F(
              ERROR,
              "Expected exactly one field in OptionOverrideProto, but got %d",
              value_fields.field_count());
          return tt_pjrt_status::kInternal;
        }

        const google::protobuf::UnknownField &value_field =
            value_fields.field(0);
        switch (value_field.number()) {
        case 1: {
          value = value_field.length_delimited();
          break;
        }
        case 2: {
          value = value_field.varint() ? "true" : "false";
          break;
        }
        case 3: {
          value = std::to_string(value_field.varint());
          break;
        }
        case 4: {
          value = std::to_string(value_field.fixed64());
          break;
        }
        default: {
          DLOG_F(ERROR, "Unknown field number in OptionOverrideProto: %d",
                 value_field.number());
          return tt_pjrt_status::kInternal;
        }
        }
      }
    }

    if (!key.empty()) {
      out_compile_options[key] = value;
    }
  }

  return tt_pjrt_status::kSuccess;
}

tt::runtime::Tensor *ClientInstance::getCachedTensor(
    const std::vector<BufferInstance *> &buffer_instances) {
  auto it = m_tensor_cache.find(buffer_instances);

  std::ostringstream oss;
  oss << "(";
  for (BufferInstance *keyfrag : buffer_instances) {
    oss << keyfrag << ", ";
  }
  oss << ")";

  std::string keys = oss.str();

  if (it != m_tensor_cache.end()) {
    DLOG_F(LOG_DEBUG, "Tensor cache HIT for %zu buffer instances %s",
           buffer_instances.size(), keys.c_str());
    return &(it->second);
  }
  DLOG_F(LOG_DEBUG, "Tensor cache MISS for %zu buffer instances %s",
         buffer_instances.size(), keys.c_str());
  return nullptr;
}

void ClientInstance::setCachedTensor(
    const std::vector<BufferInstance *> &buffer_instances,
    const tt::runtime::Tensor &tensor) {
  m_tensor_cache[buffer_instances] = tensor;
  DLOG_F(LOG_DEBUG,
         "Cached tensor for %zu buffer instances (cache size now: %zu)",
         buffer_instances.size(), m_tensor_cache.size());
}

void ClientInstance::clearTensorCache() {
  DLOG_F(LOG_DEBUG, "Clearing tensor cache with %zu entries",
         m_tensor_cache.size());
  m_tensor_cache.clear();
}

tt::runtime::Device ClientInstance::getOrCreateMeshDevice(
    const std::vector<uint32_t> &target_mesh_shape) {

  if (!m_parent_mesh.has_value()) {
    m_parent_mesh = openMeshDevice(target_mesh_shape);
    return *m_parent_mesh;
  }

  std::vector<uint32_t> parent_mesh_shape =
      tt::runtime::getMeshShape(*m_parent_mesh);

  if (parent_mesh_shape == target_mesh_shape) {
    DLOG_F(LOG_DEBUG,
           "ClientInstance::getOrCreateMeshDevice - reusing "
           "already opened mesh device %s",
           utils::to_string(parent_mesh_shape).c_str());
    return *m_parent_mesh;
  }

  DLOG_F(LOG_DEBUG,
         "ClientInstance::getOrCreateMeshDevice - "
         "reshaping mesh device - %s -> %s",
         utils::to_string(parent_mesh_shape).c_str(),
         utils::to_string(target_mesh_shape).c_str());

  // NOTE: Due to some issues hit when testing, instead of using the reshape
  // mesh API, we are closing and re-opening the device with the wanted mesh
  // shape. This should be revisited in the future (#1436).
  //
  // Additionally, we are supposed to utilize sub-meshes if the target mesh
  // shape is contained within the already opened parent mesh. Also, in case
  // we are running multiple models on different parts of the mesh (pipeline
  // parallel). However, similar as to the case with reshape API, there were
  // some issues when testing sub-meshes, so for now we are always closing and
  // re-opening the whole mesh.
  tt::runtime::closeMeshDevice(*m_parent_mesh);
  m_parent_mesh = openMeshDevice(target_mesh_shape);

  return *m_parent_mesh;
}

tt::runtime::Device
ClientInstance::openMeshDevice(const std::vector<uint32_t> &mesh_shape) {
  size_t num_devices =
      static_cast<size_t>(std::accumulate(mesh_shape.begin(), mesh_shape.end(),
                                          1, std::multiplies<std::uint32_t>{}));

  // NOTES:
  // - this should probably be set automatically by the mlir runtime.
  // - it looks like metal context is being reinitialized each time we
  // open/close the device, so we need to set the fabric config each time
  // (even if we always set it to the same value).
  if (num_devices > 1) {
    tt::runtime::setFabricConfig(tt::runtime::FabricConfig::FABRIC_1D);
  } else {
    tt::runtime::setFabricConfig(tt::runtime::FabricConfig::DISABLED);
  }

  tt::runtime::MeshDeviceOptions options = tt::runtime::MeshDeviceOptions{
      .meshShape = mesh_shape,
  };

  return tt::runtime::openMeshDevice(options);
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
  std::unordered_map<std::string, std::string> compile_options_map;

  tt_pjrt_status compile_option_status = ClientInstance::getCompileOptions(
      args->compile_options, args->compile_options_size, compile_options_map);
  if (!tt_pjrt_status_is_ok(compile_option_status)) {
    return *ErrorInstance::makeError(compile_option_status).release();
  }

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
