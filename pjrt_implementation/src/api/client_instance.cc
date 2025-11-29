// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/client_instance.h"

// c++ standard library includes
#include <cstddef>
#include <filesystem>
#include <optional>

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

// third-party includes
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/unknown_field_set.h>

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/error_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "api/memory_instance.h"
#include "api/module_builder/module_builder.h"
#include "utils/logging.h"
#include "utils/utils.h"

namespace tt::pjrt {

static std::string getRankBindingPath(const std::string &metal_home) {
  static std::unordered_map<std::string, std::string> rank_binding_paths = {
      {"2x4_multiprocess",
       "tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml"},
  };

  const char *rank_binding = std::getenv("TT_DISTRIBUTED_RANK_BINDING");
  if (!rank_binding) {
    DLOG_F(ERROR,
           "TT_DISTRIBUTED_RANK_BINDING environment variable is not set");
    return "";
  }
  if (rank_binding_paths.find(rank_binding) == rank_binding_paths.end()) {
    DLOG_F(ERROR, "Invalid rank binding: %s", rank_binding);
    return "";
  }

  std::filesystem::path rank_binding_path =
      std::filesystem::path(metal_home) / rank_binding_paths.at(rank_binding);

  if (std::filesystem::exists(rank_binding_path) == false) {
    DLOG_F(ERROR, "Rank binding file does not exist at path: %s",
           rank_binding_path.c_str());
    return "";
  }

  return rank_binding_path.string();
}

static std::string getDistributedWorkerPath() {
  const char *distributed_worker_path =
      std::getenv("TT_DISTRIBUTED_WORKER_PATH");
  if (!distributed_worker_path) {
    DLOG_F(ERROR, "TT_DISTRIBUTED_WORKER_PATH environment variable is not set");
    return "";
  }

  if (std::filesystem::exists(distributed_worker_path) == false) {
    DLOG_F(ERROR, "Distributed worker file does not exist at path: %s",
           distributed_worker_path);
    return "";
  }

  return distributed_worker_path;
}

static tt_pjrt_status launchDistributedRuntime() {
  const char *metal_home = std::getenv("TT_METAL_RUNTIME_ROOT");
  if (!metal_home) {
    DLOG_F(ERROR, "TT_METAL_RUNTIME_ROOT environment variable is not set");
    return tt_pjrt_status::kInternal;
  }
  tt::runtime::setMetalHome(metal_home);

  std::string rank_binding_path = getRankBindingPath(metal_home);
  if (rank_binding_path.empty()) {
    return tt_pjrt_status::kInternal;
  }

  std::string distributed_worker_path = getDistributedWorkerPath();
  if (distributed_worker_path.empty()) {
    return tt_pjrt_status::kInternal;
  }

  tt::runtime::DistributedOptions distributed_options;
  distributed_options.mode = tt::runtime::DistributedMode::MultiProcess;
  distributed_options.workerPath = distributed_worker_path;
  distributed_options.multiProcessArgs =
      tt::runtime::MultiProcessArgs::create(rank_binding_path)
          .withAllowRunAsRoot(true);

  tt::runtime::setCurrentHostRuntime(tt::runtime::HostRuntime::Distributed);
  tt::runtime::launchDistributedRuntime(distributed_options);

  return tt_pjrt_status::kSuccess;
}

static tt_pjrt_status setMemoryLogLevel() {
  const char *memory_log_level_env = std::getenv("TT_RUNTIME_MEMORY_LOG_LEVEL");
  if (!memory_log_level_env) {
    return tt_pjrt_status::kSuccess;
  }

  std::string memory_log_level_str(memory_log_level_env);
  std::transform(memory_log_level_str.begin(), memory_log_level_str.end(),
                 memory_log_level_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  tt::runtime::MemoryLogLevel log_level;
  if (memory_log_level_str == "none") {
    log_level = tt::runtime::MemoryLogLevel::NONE;
  } else if (memory_log_level_str == "program") {
    log_level = tt::runtime::MemoryLogLevel::Program;
  } else if (memory_log_level_str == "operation") {
    log_level = tt::runtime::MemoryLogLevel::Operation;
  } else if (memory_log_level_str == "any" || memory_log_level_str == "all") {
    log_level = tt::runtime::MemoryLogLevel::ANY;
  } else {
    DLOG_F(ERROR, "Invalid memory logging level: %s", memory_log_level_env);
    return tt_pjrt_status::kInternal;
  }

  tt::runtime::setMemoryLogLevel(log_level);

  return tt_pjrt_status::kSuccess;
}

PJRT_Error *GlobalClientInstanceSingleton::initClient() {
  std::unique_ptr<ClientInstance> client = std::make_unique<ClientInstance>();
  PJRT_Error *error = client->initialize();
  if (error) {
    return error;
  }

  GlobalClientInstanceSingleton &singleton_instance = getInstance();
  singleton_instance.m_client_instance = std::move(client);

  return nullptr;
}

void GlobalClientInstanceSingleton::destroyClient() {
  GlobalClientInstanceSingleton &singleton_instance = getInstance();
  if (singleton_instance.isInitialized()) {
    singleton_instance.m_client_instance.reset();
  }
}

GlobalClientInstanceSingleton &GlobalClientInstanceSingleton::getInstance() {
  static GlobalClientInstanceSingleton singleton =
      GlobalClientInstanceSingleton(nullptr);
  return singleton;
}

ClientInstance *GlobalClientInstanceSingleton::getClientInstance() {
  auto &singleton = GlobalClientInstanceSingleton::getInstance();
  return singleton.m_client_instance.get();
}

ClientInstance::ClientInstance()
    : m_system_descriptor(nullptr),
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
  closeMeshDevice();
  std::remove(m_cached_system_descriptor_path.data());
}

PJRT_Error *ClientInstance::initialize() {
  DLOG_F(LOG_DEBUG, "ClientInstance::Initialize");

  bool distributed_runtime =
      std::getenv("TT_RUNTIME_ENABLE_DISTRIBUTED") != nullptr &&
      std::string(std::getenv("TT_RUNTIME_ENABLE_DISTRIBUTED")) != "0";

  if (distributed_runtime) {
    tt_pjrt_status launch_result = launchDistributedRuntime();
    if (!tt_pjrt_status_is_ok(launch_result)) {
      return *ErrorInstance::makeError(launch_result).release();
    }
  }

  tt_pjrt_status memory_log_level_status = setMemoryLogLevel();
  if (!tt_pjrt_status_is_ok(memory_log_level_status)) {
    return *ErrorInstance::makeError(memory_log_level_status).release();
  }

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
  api->PJRT_Client_Create = internal::onClientCreate;
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

  std::tuple<tt_pjrt_status, std::shared_ptr<ExecutableImage>> compile_result =
      m_module_builder->buildModule(mlir_code, m_cached_system_descriptor_path,
                                    compile_options, this);
  tt_pjrt_status status = std::get<tt_pjrt_status>(compile_result);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }

  auto executable_image =
      std::get<std::shared_ptr<ExecutableImage>>(compile_result);

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
          executable_image->getNumDevicesToUtilize());

  std::unique_ptr<LoadedExecutableInstance> executable =
      executable_image->toExecutableInstance(std::move(addressable_devices),
                                             this);

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
  closeMeshDevice();
  m_parent_mesh = openMeshDevice(target_mesh_shape);

  return *m_parent_mesh;
}

void ClientInstance::closeMeshDevice() {
  closeOptimizerSubmesh();
  closeParentMesh();
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

  // TODO(odjuricicTT, #1485): This is a temporary way to disable program cache
  // now that it's enabled by default here,
  // until we have a proper way for a user to pass device options. After that
  // this should be removed. Issue for device options:
  // https://github.com/tenstorrent/tt-xla/issues/1480
  bool enableProgramCache =
      std::getenv("TT_RUNTIME_ENABLE_PROGRAM_CACHE") == nullptr ||
      std::string(std::getenv("TT_RUNTIME_ENABLE_PROGRAM_CACHE")) != "0";

  // TODO(jnie-TT, #1485): This is a temporary way to set trace region size
  // until we have a proper way for a user to pass device options. After that
  // this should be removed. Issue for device options:
  // https://github.com/tenstorrent/tt-xla/issues/1480
  std::optional<size_t> traceRegionSize = std::nullopt;
  if (std::getenv("TT_RUNTIME_TRACE_REGION_SIZE") != nullptr) {
    traceRegionSize = std::stoull(std::getenv("TT_RUNTIME_TRACE_REGION_SIZE"));
  }

  tt::runtime::MeshDeviceOptions options = tt::runtime::MeshDeviceOptions{
      .enableProgramCache = enableProgramCache,
      .meshShape = mesh_shape,
      .traceRegionSize = traceRegionSize,
  };

  return tt::runtime::openMeshDevice(options);
}

void ClientInstance::closeParentMesh() {
  if (m_parent_mesh.has_value()) {
    DLOG_F(LOG_DEBUG, "Closing parent mesh.");
    tt::runtime::closeMeshDevice(*m_parent_mesh);
    m_parent_mesh.reset();
  }
}

void ClientInstance::closeOptimizerSubmesh() {
  if (m_optimizer_submesh.has_value()) {
    DLOG_F(LOG_DEBUG, "Closing optimizer submesh.");
    tt::runtime::releaseSubMeshDevice(*m_optimizer_submesh);
    m_optimizer_submesh.reset();
  }
}

tt::runtime::Device ClientInstance::getOrCreateOptimizerSubmesh(
    const std::vector<uint32_t> &target_mesh_shape) {

  // Ensure parent mesh exists with the correct shape
  tt::runtime::Device parent_mesh = getOrCreateMeshDevice(target_mesh_shape);

  if (m_optimizer_submesh.has_value()) {
    std::vector<uint32_t> optimizer_submesh_shape =
        tt::runtime::getMeshShape(*m_optimizer_submesh);

    if (optimizer_submesh_shape == target_mesh_shape) {
      DLOG_F(LOG_DEBUG, "ClientInstance::getOrCreateOptimizerSubmesh - reusing "
                        "already created optimizer submesh");
      return *m_optimizer_submesh;
    }

    // If shape changed, parent mesh was closed and reopened in
    // getOrCreateMeshDevice, which automatically closed the submesh.
    // Clear the stale reference.
    m_optimizer_submesh.reset();
  }

  DLOG_F(LOG_DEBUG, "ClientInstance::getOrCreateOptimizerSubmesh - "
                    "creating optimizer submesh");
  m_optimizer_submesh =
      tt::runtime::createSubMeshDevice(parent_mesh, target_mesh_shape);

  return *m_optimizer_submesh;
}

namespace internal {

PJRT_Error *onClientCreate(PJRT_Client_Create_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Create");

  // We currently don't utilize any of the PJRT Client create options.
  for (size_t i = 0; i < args->num_options; ++i) {
    DLOG_F(WARNING, "Unused PJRT Client create option: %s",
           args->create_options[i].name);
  }

  PJRT_Error *error = GlobalClientInstanceSingleton::initClient();

  if (error) {
    DLOG_F(ERROR, "Failed to initialize PJRT client instance");
    return error;
  }

  ClientInstance *client_instance =
      GlobalClientInstanceSingleton::getClientInstance();
  assert(client_instance != nullptr);
  args->client = reinterpret_cast<PJRT_Client *>(client_instance);

  return nullptr;
}

PJRT_Error *onClientDestroy(PJRT_Client_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Destroy");

  ClientInstance *client_instance = ClientInstance::unwrap(args->client);
  ClientInstance *global_client_instance =
      GlobalClientInstanceSingleton::getClientInstance();
  assert(client_instance == global_client_instance);
  GlobalClientInstanceSingleton::destroyClient();
  return nullptr;
}

PJRT_Error *onClientPlatformName(PJRT_Client_PlatformName_Args *args) {
  DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformName");

  ClientInstance *client = ClientInstance::unwrap(args->client);

  args->platform_name = client->platformName().data();
  args->platform_name_size = client->platformName().size();

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

  args->platform_version = client->platformVersion().data();
  args->platform_version_size = client->platformVersion().size();

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
