// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#include "api/so_loaded_executable_instance.h"

// c++ standard library includes
#include <mutex>
#include <numeric>

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/client_instance.h"
#include "api/device_instance.h"
#include "api/event_instance.h"
#include "api/executable_image.h"
#include "utils/data_type_utils.h"
#include "utils/logging.h"
#include "utils/status.h"

namespace tt::pjrt {

std::unique_ptr<SOLoadedExecutableInstance>
SOLoadedExecutableInstance::createInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices,
    ClientInstance *client_instance) {
  struct make_unique_enabler : public SOLoadedExecutableInstance {
    make_unique_enabler(std::shared_ptr<SOExecutableImage> executable_image,
                        std::vector<DeviceInstance *> &&addressable_devices,
                        ClientInstance *client_instance)
        : SOLoadedExecutableInstance(std::move(executable_image),
                                     std::move(addressable_devices),
                                     client_instance) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image),
                                               std::move(addressable_devices),
                                               client_instance);
}

SOLoadedExecutableInstance::SOLoadedExecutableInstance(
    std::shared_ptr<SOExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices,
    ClientInstance *client_instance)
    : LoadedExecutableInstance(std::move(executable_image), addressable_devices,
                               client_instance) {}

std::shared_ptr<SOExecutableImage>
SOLoadedExecutableInstance::getSharedExecutableImage() const {
  return std::static_pointer_cast<SOExecutableImage>(m_executable_image);
}

void SOLoadedExecutableInstance::releaseResources() {
  if (m_deleted) {
    return;
  }

  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  if (m_deleted) {
    return;
  }

  // Release SO-specific resources
  m_executable_image.reset();

  m_deleted = true;
}

tt_pjrt_status
SOLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "SOLoadedExecutableInstance::Execute");

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    DLOG_F(ERROR, "Device count mismatch: %zu vs %zu", args->num_devices,
           m_executable_image->getNumDevicesToUtilize());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    DLOG_F(ERROR, "Argument count mismatch: %zu vs %zu", args->num_args,
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

  // Assuming only one program per SO for now.
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

  CompileOptions options = m_executable_image->getCompileOptions();
  std::string lang =
      options.backend == BackendRuntime::TTNNCodegenPy ? "Python" : "C++";
  std::cout << lang << " codegen successful. Check "
            << options.export_path.value() << " for the results." << std::endl;
  // TODO: Implement SO execution. For now, we create default output buffers.
  // https://github.com/tenstorrent/tt-xla/issues/2038
  createDefaultOutputBuffers(args->output_lists, args->num_devices);

  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      device_complete_event->markAsReady(tt_pjrt_status::kSuccess);

      // Releasing ownership to the PJRT API caller
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }

  return tt_pjrt_status::kSuccess;
}

void SOLoadedExecutableInstance::createDefaultOutputBuffers(
    PJRT_Buffer **const *output_lists, size_t num_devices) {
  size_t num_outputs = m_executable_image->getNumOutputs();

  for (size_t device_index = 0; device_index < num_devices; ++device_index) {
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      std::vector<std::uint32_t> output_shape =
          m_executable_image->getOutputShape(output_index);
      PJRT_Buffer_Type output_type =
          m_executable_image->getOutputTypes()[output_index];
      ::tt::target::DataType runtime_data_type =
          data_type_utils::convertPJRTToRuntimeDataType(output_type);
      std::uint32_t element_size =
          tt::runtime::utils::dataTypeElementSize(runtime_data_type);

      // We create a row-major tensor. Last stride is 1, one before is the last
      // dimension size, etc. That means the right algorithm is the exclusive
      // right scan.
      std::vector<std::uint32_t> strides(output_shape.size());
      std::exclusive_scan(output_shape.rbegin(), output_shape.rend(),
                          strides.rbegin(), std::uint32_t(1),
                          std::multiplies<>());

      tt::runtime::Tensor host_tensor = tt::runtime::createOwnedHostTensor(
          nullptr, output_shape, strides, element_size, runtime_data_type);

      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createOutputBufferInstance(
              host_tensor, std::move(output_shape),
              m_addressable_devices[device_index],
              m_addressable_devices[device_index]->getDefaultMemory(),
              output_type);

      output_buffer->markAsDataReady();

      // Release ownership to the PJRT API caller
      output_lists[device_index][output_index] = *output_buffer.release();
    }
  }
}

std::optional<tt::runtime::Tensor>
SOLoadedExecutableInstance::prepareInputTensor(
    const std::vector<BufferInstance *> &arg_buffers,
    tt::runtime::Device runtime_device, size_t num_devices,
    std::uint32_t program_index, size_t arg_index) {
  // Assert that all buffer instances have the same prepared tensor.
  // NOTE: In case of sharded tensor we have multiple buffer instances on the
  // PJRT side, but on our side (tt-mlir runtime) we prepare a single
  // multi-device tensor.
  assert(!arg_buffers.empty());
  std::optional<tt::runtime::Tensor> prepared_tensor =
      arg_buffers[0]->getPreparedTensor();
  for (size_t i = 1; i < arg_buffers.size(); ++i) {
    assert(arg_buffers[i]->getPreparedTensor().has_value() ==
           prepared_tensor.has_value());
    if (prepared_tensor.has_value()) {
      assert(arg_buffers[i]->getPreparedTensor()->handle ==
             prepared_tensor->handle);
    }
  }

  // For SO path, we don't have layout information from flatbuffer,
  // so we can't check if the prepared tensor has the correct layout.
  // As a hack, we'll always use whatever we have prepared.
  if (prepared_tensor.has_value()) {
    DLOG_F(LOG_DEBUG,
           "Reusing already prepared input tensor for argument index %zu",
           arg_index);
    return *prepared_tensor;
  }

  // We don't have an already prepared tensor so we need to prepare it now.
  mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
      fillStrategyMapFromSharding(
          m_executable_image->getInputSharding(arg_index), num_devices);
  if (mlir::failed(strategy)) {
    DLOG_F(ERROR, "Failed to fill strategy map from sharding");
    return std::nullopt;
  }

  tt::runtime::Tensor input_tensor =
      getTensorFromStrategy(arg_buffers, *strategy);

  tt::runtime::Tensor laid_out_tensor = convertTensorLayout(
      input_tensor, program_index, arg_index, runtime_device);

  // Right now we don't actually lay out tensors, so no need to save it.

  return laid_out_tensor;
}

tt::runtime::Tensor SOLoadedExecutableInstance::convertTensorLayout(
    tt::runtime::Tensor input_tensor, std::uint32_t program_index,
    size_t arg_index, const tt::runtime::Device &runtime_device) {
  return input_tensor;
}

} // namespace tt::pjrt
