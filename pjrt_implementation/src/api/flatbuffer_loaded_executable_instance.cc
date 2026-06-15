// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/flatbuffer_loaded_executable_instance.h"

// c++ standard library includes
#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// tracy includes
#include "tracy/Tracy.hpp"
#include "tt/runtime/runtime.h"

// tt-mlir includes
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "api/tensor.h"
#include "utils/logging.h"
#include "utils/utils.h"

namespace tt::pjrt {

namespace {

std::string
strategyToString(const std::unordered_map<std::string, std::string> &strategy) {
  std::vector<std::string> keys;
  keys.reserve(strategy.size());
  for (const auto &[key, _] : strategy) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  std::stringstream stream;
  stream << "{";
  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string &key = keys[i];
    stream << key << ": " << strategy.at(key)
           << (i + 1 < keys.size() ? ", " : "");
  }
  stream << "}";
  return stream.str();
}

} // namespace

std::unique_ptr<FlatbufferLoadedExecutableInstance>
FlatbufferLoadedExecutableInstance::createInstance(
    std::shared_ptr<FlatbufferExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices,
    ClientInstance *client_instance) {
  struct make_unique_enabler : public FlatbufferLoadedExecutableInstance {
    make_unique_enabler(
        std::shared_ptr<FlatbufferExecutableImage> executable_image,
        std::vector<DeviceInstance *> &&addressable_devices,
        ClientInstance *client_instance)
        : FlatbufferLoadedExecutableInstance(std::move(executable_image),
                                             std::move(addressable_devices),
                                             client_instance) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image),
                                               std::move(addressable_devices),
                                               client_instance);
}

FlatbufferLoadedExecutableInstance::FlatbufferLoadedExecutableInstance(
    std::shared_ptr<FlatbufferExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices,
    ClientInstance *client_instance)
    : LoadedExecutableInstance(std::move(executable_image), addressable_devices,
                               client_instance) {}

std::optional<tt::runtime::Tensor>
FlatbufferLoadedExecutableInstance::prepareInputTensor(
    const std::vector<BufferInstance *> &arg_buffers,
    tt::runtime::Device runtime_device, size_t num_devices,
    std::uint32_t program_index, size_t arg_index) {
  ZoneScoped;

  FlatbufferExecutableImage *executable_image =
      static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

  tt::runtime::Layout expected_layout = tt::runtime::getLayout(
      executable_image->getFlatbufferBinary(), program_index, arg_index);

  mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
      fillStrategyMapFromSharding(
          m_executable_image->getInputSharding(arg_index), num_devices);

  if (mlir::failed(strategy)) {
    LOG_F(ERROR, "Failed to fill strategy map from sharding");
    return std::nullopt;
  }

  bool is_distributed = ::tt::runtime::getCurrentHostRuntime() ==
                        tt::runtime::HostRuntime::Distributed;
  auto runtime_mesh_shape = utils::invoke_noexcept(
      [&] { return tt::runtime::getMeshShape(runtime_device); });
  std::string runtime_mesh_shape_str =
      runtime_mesh_shape.has_value() ? utils::to_string(*runtime_mesh_shape)
                                     : "<unavailable>";

  LOG_F(INFO,
        "prepareInputTensor arg=%zu program_index=%u num_devices=%zu "
        "distributed=%d executable_mesh_shape=%s runtime_mesh_shape=%s "
        "strategy=%s",
        arg_index, program_index, num_devices, is_distributed,
        utils::to_string(m_executable_image->getDevicesMeshShape()).c_str(),
        runtime_mesh_shape_str.c_str(), strategyToString(*strategy).c_str());

  if (is_distributed) {
    LOG_F(INFO, "Materializing shell tensors for arg=%zu", arg_index);
    materializeShellTensors(arg_buffers);
    LOG_F(INFO, "Done materializing shell tensors for arg=%zu", arg_index);
  }

  PjrtTensor &tensor = PjrtTensor::from_pjrt_buffers(
      arg_buffers, m_executable_image->getDevicesMeshShape(), *strategy);

  LOG_F(INFO,
        "Calling ensure_layout for arg=%zu tensor_uid=%lu shard_count=%zu "
        "has_runtime_tensor=%d",
        arg_index, tensor.uid(), tensor.shards().size(),
        tensor.has_runtime_tensor());
  loguru::flush();
  tensor.ensure_layout(runtime_device, expected_layout);
  LOG_F(INFO, "Done ensure_layout for arg=%zu tensor_uid=%lu", arg_index,
        tensor.uid());

  return tensor.runtime_tensor();
}

void FlatbufferLoadedExecutableInstance::materializeShellTensors(
    const std::vector<BufferInstance *> &arg_buffers) {
  // Group arg buffers by their borrowed host base pointer for grouped transfer
  std::unordered_map<const void *, std::vector<BufferInstance *>>
      borrowed_host_base_ptr_to_buffers;

  for (BufferInstance *buf : arg_buffers) {
    // Buffers without a shell have already been materialized
    const auto &shell = buf->getPjrtTensor()->host_tensor_shell();
    if (shell.has_value()) {
      borrowed_host_base_ptr_to_buffers[shell->host_buffer].push_back(buf);
    }
  }

  for (const auto &[host_base, buffers] : borrowed_host_base_ptr_to_buffers) {
    const auto &shell = buffers.front()->getPjrtTensor()->host_tensor_shell();

    TT_FATAL(shell.has_value(),
             "Missing host tensor shell metadata for buffer uid=%lu ptr=%p",
             buffers.front()->getUID(),
             static_cast<const void *>(buffers.front()));

    for (size_t i = 1; i < buffers.size(); ++i) {
      const auto &other = buffers[i]->getPjrtTensor()->host_tensor_shell();
      TT_FATAL(other.has_value() && *other == *shell,
               "Shell metadata mismatch for buffers sharing host_buffer %p: "
               "buffer uid=%lu vs uid=%lu",
               host_base, buffers[i]->getUID(), buffers.front()->getUID());
    }

    LOG_F(INFO,
          "materializeShellTensors host_base=%p group_size=%zu shape=%s "
          "strides=%s element_size=%u dtype=%s first_buffer_uid=%lu",
          host_base, buffers.size(), utils::to_string(shell->shape).c_str(),
          utils::to_string(shell->strides).c_str(), shell->element_size,
          ::tt::target::EnumNameDataType(shell->runtime_data_type),
          buffers.front()->getUID());

    // First buffer gets a newly-allocated owned host tensor that actually
    // copies the bytes from the client's host_base pointer. Subsequent
    // buffers in the group get unsafe-borrowed tensors aliasing that owned
    // tensor, so we only hold one copy of the data on the worker side.
    tt::runtime::Tensor owned_tensor = tt::runtime::createOwnedHostTensor(
        const_cast<void *>(shell->host_buffer), shell->shape, shell->strides,
        shell->element_size, shell->runtime_data_type);

    for (size_t i = 0; i < buffers.size(); ++i) {
      BufferInstance *buffer = buffers[i];

      const bool is_first = (i == 0);
      tt::runtime::Tensor worker_runtime_tensor =
          is_first ? owned_tensor
                   : tt::runtime::createUnsafeBorrowedHostTensor(owned_tensor);

      LOG_F(INFO,
            "materializeShellTensors assigning worker tensor "
            "buffer_uid=%lu group_index=%zu is_first=%d global_device_id=%d",
            buffer->getUID(), i, is_first,
            buffer->getDevice()->getGlobalDeviceId());

      // Inplace replacement: rebuilds the BufferInstance's PjrtTensor around
      // the new runtime tensor and updates m_pjrt_tensor via setPjrtTensor.
      // This releases the old (borrowed-from-client) runtime tensor.
      PjrtTensor::from_runtime_tensor({buffer},
                                      std::move(worker_runtime_tensor));
    }
  }
}

std::shared_ptr<FlatbufferExecutableImage>
FlatbufferLoadedExecutableInstance::getSharedExecutableImage() const {
  return std::static_pointer_cast<FlatbufferExecutableImage>(
      m_executable_image);
}

void FlatbufferLoadedExecutableInstance::releaseResources() {
  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  if (m_deleted) {
    return;
  }

  // Here we should drop executable's reference to the internal runtime object
  // and associated resources, but we currently store no runtime objects so
  // releasing only resources.
  m_executable_image.reset();

  m_deleted = true;
}

// TODO(mrakita): Make this method work in asynchronous fashion.
tt_pjrt_status FlatbufferLoadedExecutableInstance::execute(
    PJRT_LoadedExecutable_Execute_Args *args) {
  ZoneScoped;
  LOG_F(INFO, "FlatbufferLoadedExecutableInstance::Execute");
  LOG_BRINGUP_STAGE("RUNTIME_EXECUTION_START");

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    LOG_F(ERROR, "Device count mismatch: %zu vs %zu", args->num_devices,
          m_executable_image->getNumDevicesToUtilize());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    LOG_F(ERROR, "Argument count mismatch: %zu vs %zu", args->num_args,
          m_executable_image->getNumInputs());
    return tt_pjrt_status::kInternal;
  }

  std::optional<tt::runtime::Device> runtime_device =
      getOrCreateMeshDevice(args->argument_lists, args->num_args,
                            args->num_devices, args->execute_device);

  if (!runtime_device) {
    // Logging is done inside `getOrCreateMeshDevice`.
    return tt_pjrt_status::kInternal;
  }

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;

  LOG_F(INFO, "Invoking getInputRuntimeTensors");

  if (!m_executable_image->getCompileOptions().dry_run) {
    FlatbufferExecutableImage *executable_image = static_cast<FlatbufferExecutableImage *>(m_executable_image.get());
    tt::runtime::Binary binary = executable_image->getFlatbufferBinary();
    tt::runtime::seedProgramBinary(binary);
  }

  std::vector<tt::runtime::Tensor> input_tensors;
  input_tensors.reserve(args->num_args);
  tt_pjrt_status status = getInputRuntimeTensors(
      args->argument_lists, args->num_args, args->num_devices, *runtime_device,
      program_index, input_tensors);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }

  if (m_executable_image->getCompileOptions().export_tensors) {
    dumpInputs(input_tensors);
  }

  if (m_executable_image->getCompileOptions().dry_run) {
    status = createDefaultOutputBuffers(args->output_lists, args->num_devices);
    if (!tt_pjrt_status_is_ok(status)) {
      return status;
    }
  } else {
    FlatbufferExecutableImage *executable_image =
        static_cast<FlatbufferExecutableImage *>(m_executable_image.get());

    LOG_F(INFO, "Invoking tt::runtime::submit");
    auto r = utils::invoke_noexcept(tt::runtime::submit, *runtime_device,
                                    executable_image->getFlatbufferBinary(),
                                    program_index, input_tensors);

    LOG_F(INFO, "Done tt::runtime::submit");

    if (!r) {
      m_client_instance->closeMeshDevice();
      return tt_pjrt_status::kInternal;
    }

    std::vector<tt::runtime::Tensor> &output_tensors = *r;

    if (output_tensors.size() != m_executable_image->getNumOutputs()) {
      LOG_F(ERROR,
            "Runtime produced different number of output tensors (%zu) than "
            "the compiler estimated number of outputs (%zu)",
            output_tensors.size(), m_executable_image->getNumOutputs());
      return tt_pjrt_status::kInternal;
    }

    status = fillPJRTOutputLists(output_tensors, args->num_devices,
                                 args->output_lists,
                                 m_executable_image->getOutputTypes());
    LOG_F(INFO, "Done fillPJRTOutputLists");

    if (!tt_pjrt_status_is_ok(status)) {
      return status;
    }
  }

  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      EventInstance::markAsReadyAndCallback(device_complete_event.get(),
                                            tt_pjrt_status::kSuccess);

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Event_Destroy` on the event.
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt
