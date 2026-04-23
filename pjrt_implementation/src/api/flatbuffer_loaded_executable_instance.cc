// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/flatbuffer_loaded_executable_instance.h"

// tracy includes
#include "tracy/Tracy.hpp"

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

  PjrtTensor &tensor = PjrtTensor::from_pjrt_buffers(
      arg_buffers, m_executable_image->getDevicesMeshShape(), *strategy);

  tensor.ensure_layout(runtime_device, expected_layout);

  return tensor.runtime_tensor();
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
  DLOG_F(LOG_DEBUG, "FlatbufferLoadedExecutableInstance::Execute");
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

    auto r = utils::invoke_noexcept(tt::runtime::submit, *runtime_device,
                                    executable_image->getFlatbufferBinary(),
                                    program_index, input_tensors);

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
