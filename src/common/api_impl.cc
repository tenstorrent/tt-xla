// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/api_impl.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

#include "common/module_builder.h"
#include "common/status.h"

namespace tt::pjrt {

const std::string_view kMlirFormat = "mlir";

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

  auto status = PopulateDevices();
  if (!tt_pjrt_status_is_ok(status)) {
    return MakeError(status);
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
    auto *client = ClientInstance::Unwrap(args->client);
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
    auto *client = ClientInstance::Unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices =
      +[](PJRT_Client_Devices_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Devices");
    auto &devices = ClientInstance::Unwrap(args->client)->devices();
    args->devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableDevices_Args");
    auto &devices = ClientInstance::Unwrap(args->client)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_LookupDevice =
      +[](PJRT_Client_LookupDevice_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_LookupDevice_Args");
    auto &devices = ClientInstance::Unwrap(args->client)->devices();
    size_t id_as_size = args->id;
    if (id_as_size >= devices.size()) {
      return MakeError(tt_pjrt_status::kOutOfRange);
    }
    args->device = *devices[id_as_size];
    return nullptr;
  };
  api->PJRT_Client_AddressableMemories =
      +[](PJRT_Client_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableMemories");
    // return MakeError(tt_pjrt_status::kUnimplemented);
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
    auto *client = ClientInstance::Unwrap(args->client);
    LoadedExecutableInstance *executable;

    auto *error = client->Compile(args->program, &executable);
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
    auto status = DeviceInstance::Unwrap(args->device)
                      ->HostBufferToDevice(
                          args->data, args->type, args->dims, args->num_dims,
                          args->byte_strides, args->num_byte_strides,
                          args->host_buffer_semantics,
                          reinterpret_cast<EventInstance **>(
                              &args->done_with_host_buffer),
                          reinterpret_cast<BufferInstance **>(&args->buffer));
    return MakeError(status);
  };
  api->PJRT_LoadedExecutable_Fingerprint =
      +[](PJRT_LoadedExecutable_Fingerprint_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_LoadedExecutable_Fingerprint");
    return MakeError(tt_pjrt_status::kUnimplemented);
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
  for (auto *device : devices_) {
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
    return MakeError(status);
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

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

EventInstance::EventInstance() {
  bool fence = false;
  // TODO: fence and wait
  if (!fence) {
    is_ready_ = true;
    return;
  }

  // {
  //   std::lock_guard<std::mutex> guard(lock_);
  //   // Create a thread that waits on the fence and executes the callbacks
  //   when
  //   // the fence is ready.
  //   signal_thread_ = std::make_unique<std::thread>(
  //       [](EventInstance* event_instance,
  //          iree::vm::ref<iree_hal_fence_t> fence) {
  //         iree_status_t wait_status =
  //             iree_hal_fence_wait(fence.get(), iree_infinite_timeout());
  //         event_instance->SignalReady(wait_status);
  //       },
  //       this, std::move(fence));
  // }
}

EventInstance::~EventInstance() {
  std::lock_guard<std::mutex> guard(lock_);
  if (signal_thread_) {
    if (std::this_thread::get_id() != signal_thread_->get_id()) {
      signal_thread_->join();
    } else {
      // An `EventInstance` is allowed to delete itself in one of its callbacks,
      // resulting in `signal_thread_` being the thread calling the destructor.
      // In such cases, we must let the thread continue running independent of
      // the destructor to avoid a deadlock.
      signal_thread_->detach();
      signal_thread_.release();
    }
  }
}

void EventInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "EventInstance::BindApi");
  api->PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Destroy");
    auto instance = EventInstance::Unwrap(args->event);
    auto delete_event = [](PJRT_Error *error, void *user_data) {
      EventInstance *event = static_cast<EventInstance *>(user_data);
      delete event;
      if (error) {
        delete ErrorInstance::FromError(error);
      }
    };

    instance->OnReady(delete_event, args->event);
    return nullptr;
  };
  api->PJRT_Event_IsReady = +[](PJRT_Event_IsReady_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_IsReady");
    args->is_ready = EventInstance::Unwrap(args->event)->is_ready();
    return nullptr;
  };
  api->PJRT_Event_Error = +[](PJRT_Event_Error_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Error");
    return (PJRT_Error *)EventInstance::Unwrap(args->event)->error();
  };
  api->PJRT_Event_Await = +[](PJRT_Event_Await_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Await");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_OnReady");
    return MakeError(EventInstance::Unwrap(args->event)
                         ->OnReady(args->callback, args->user_arg));
  };
}

ErrorInstance *EventInstance::error() {
  std::lock_guard<std::mutex> guard(lock_);
  if (!tt_pjrt_status_is_ok(status_))
    return new ErrorInstance(status_);
  return nullptr;
}
bool EventInstance::is_ready() {
  DLOG_F(LOG_DEBUG, "EventInstance::is_ready");
  std::lock_guard<std::mutex> guard(lock_);
  return is_ready_;
}

tt_pjrt_status EventInstance::OnReady(PJRT_Event_OnReadyCallback callback,
                                      void *user_arg) {
  DLOG_F(LOG_DEBUG, "EventInstance::OnReady");
  tt_pjrt_status local_status;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (!is_ready_) {
      pending_callbacks_.push_back({callback, user_arg});
      return tt_pjrt_status::kSuccess;
    }
    local_status = status_;
  }

  // Already signalled. Callback out of lock scope.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  callback(tt_pjrt_status_is_ok(local_status)
               ? nullptr
               : (PJRT_Error *)new ErrorInstance(local_status),
           user_arg);
  return tt_pjrt_status::kSuccess;
}

void EventInstance::SignalReady(tt_pjrt_status status) {
  DLOG_F(LOG_DEBUG, "EventInstance::SignalReady");
  tt_pjrt_status local_status;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void *>> local_callbacks;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (is_ready_) {
      return;
    }
    local_callbacks.swap(pending_callbacks_);
    is_ready_ = true;
    status_ = status;
    local_status = status_;
  }

  // Trigger callbacks outside of the lock.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  for (auto &cb : local_callbacks) {
    cb.first(tt_pjrt_status_is_ok(local_status)
                 ? nullptr
                 : (PJRT_Error *)new ErrorInstance(local_status),
             cb.second);
  }
}

//===----------------------------------------------------------------------===//
// LoadedExecutableInstance
//===----------------------------------------------------------------------===//

void ExecutableImage::BindApi(PJRT_Api *api) {
  api->PJRT_Executable_Destroy =
      +[](PJRT_Executable_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Destroy");
    ExecutableImage::Unwrap(args->executable)->DecRef();
    return nullptr;
  };
  api->PJRT_Executable_Name =
      +[](PJRT_Executable_Name_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Name");
    const char *dummy_name = "tt_pjrt_exe";
    args->executable_name = dummy_name;
    args->executable_name_size = std::strlen(dummy_name);
    return nullptr;
  };
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      +[](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args)
      -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_SizeOfGeneratedCodeInBytes");
    args->size_in_bytes =
        0; // TODO:
           // ExecutableImage::Unwrap(args->executable)->binary->GetDataSize();
    return nullptr;
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_NumOutputs");
    auto *exec = ExecutableImage::Unwrap(args->executable);
    args->num_outputs = exec->result_count;
    return nullptr;
  };
  api->PJRT_Executable_NumPartitions =
      +[](PJRT_Executable_NumPartitions_Args *args) -> PJRT_Error * {
    // This should be updated once iree supports partitioning.
    args->num_partitions = 1;
    return nullptr;
  };
  api->PJRT_Executable_NumReplicas =
      +[](PJRT_Executable_NumReplicas_Args *args) -> PJRT_Error * {
    // This should be updated once iree supports replicas.
    args->num_replicas = 1;
    return nullptr;
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_DeserializeAndLoad =
      +[](PJRT_Executable_DeserializeAndLoad_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_DeserializeAndLoad_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OptimizedProgram");
    ExecutableImage *executable = ExecutableImage::Unwrap(args->executable);
    PJRT_Program *program = args->program;
    program->format = kMlirFormat.data();
    program->format_size = kMlirFormat.size();
    size_t code_size = executable->code.size();
    if (program->code == nullptr) {
      program->code_size = code_size;
    } else {
      if (program->code_size < code_size) {
        return MakeError(tt_pjrt_status::kInvalidArgument);
      }
      std::memcpy(program->code, executable->code.c_str(),
                  executable->code.size());
    }
    return nullptr;
  };
  api->PJRT_Executable_GetCostAnalysis =
      +[](PJRT_Executable_GetCostAnalysis_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_GetCostAnalysis_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputElementTypes =
      +[](PJRT_Executable_OutputElementTypes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_OutputElementTypes_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputDimensions =
      +[](PJRT_Executable_OutputDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputDimensions_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputMemoryKinds =
      +[](PJRT_Executable_OutputMemoryKinds_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputMemoryKinds");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
}

void LoadedExecutableInstance::BindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy =
      +[](PJRT_LoadedExecutable_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Destroy");
    delete LoadedExecutableInstance::Unwrap(args->executable);
    return nullptr;
  };
  api->PJRT_LoadedExecutable_AddressableDevices =
      +[](PJRT_LoadedExecutable_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(
        LOG_DEBUG,
        "LoadedExecutableInstance::PJRT_LoadedExecutable_AddressableDevices");
    auto &devices = LoadedExecutableInstance::Unwrap(args->executable)
                        ->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Delete =
      +[](PJRT_LoadedExecutable_Delete_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_LoadedExecutable_IsDeleted =
      +[](PJRT_LoadedExecutable_IsDeleted_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_LoadedExecutable_Execute =
      +[](PJRT_LoadedExecutable_Execute_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");
    return MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->Execute(args));
  };
  api->PJRT_LoadedExecutable_GetExecutable =
      +[](PJRT_LoadedExecutable_GetExecutable_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_GetExecutable");
    auto *loaded_exe =
        LoadedExecutableInstance::Unwrap(args->loaded_executable);
    ExecutableImage *image = loaded_exe->image_;

    image->AddRef();
    args->executable = *image;
    return nullptr;
  };
}

tt_pjrt_status
LoadedExecutableInstance::Execute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::Execute");

  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
  int dev_0 = chip_ids[0];
  auto device = tt::runtime::openDevice({dev_0});

  assert(args->num_devices == 1);
  int dev_index = 0;
  tt::runtime::Binary binary(image_->binary);

  std::vector<tt::runtime::Tensor> rt_inputs;
  rt_inputs.reserve(args->num_args);

  for (size_t i = 0; i < args->num_args; ++i) {
    auto *buffer = BufferInstance::Unwrap(args->argument_lists[dev_index][i]);
    rt_inputs.emplace_back(buffer->tensor());
    DLOG_F(INFO, "Runtime input id: %d", buffer->unique_id());
  }

  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, 0, rt_inputs);
  std::vector<tt::runtime::TensorDesc> output_specs =
      binary.getProgramOutputs(0);

  assert(rt_outputs.size() == output_specs.size());

  for (size_t i = 0; i < output_specs.size(); ++i) {
    auto result_buffer = std::make_unique<BufferInstance>(
        *this->addressable_devices_[dev_index], rt_outputs[i],
        output_specs[i].shape, output_specs[i].stride);
    result_buffer->setType(
        convertElementTypeToBufferType(output_specs[i].dataType));
    DLOG_F(INFO, "Runtime output id: %d", result_buffer->unique_id());
    args->output_lists[dev_index][i] = *(result_buffer.release());
  }

  if (args->device_complete_events) {
    args->device_complete_events[dev_index] = *(new EventInstance());
  }

  tt::runtime::closeDevice(device);

  return tt_pjrt_status::kSuccess;
}

static void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(LOG_DEBUG, "STUB: " #API);                                          \
    return (decltype(api->API(args)))MakeError(                                \
        tt_pjrt_status::kUnimplemented);                                       \
  }

#include "stubs.inc"
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api *api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->extension_start = nullptr;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  // This is a bare implementation throwing UNDEFINED errors. This way new
  // functions will not segmentation fault on invocation.
  BindUndefineds(api);
  ErrorInstance::BindApi(api);

  api->PJRT_Plugin_Initialize =
      +[](PJRT_Plugin_Initialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Initialize");
    return nullptr;
  };

  api->PJRT_Plugin_Attributes =
      +[](PJRT_Plugin_Attributes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");
    args->num_attributes = 0;
    return nullptr;
  };

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceDescription::BindApi(api);
  DeviceInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
}

} // namespace tt::pjrt
