// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/loaded_executable_instance.h"

// c++ standard library includes
#include <numeric>

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

// tt-xla includes
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/error_instance.h"

namespace tt::pjrt {

std::unique_ptr<LoadedExecutableInstance>
LoadedExecutableInstance::createInstance(
    std::shared_ptr<ExecutableImage> executable_image,
    std::vector<DeviceInstance *> &&addressable_devices) {
  struct make_unique_enabler : public LoadedExecutableInstance {
    make_unique_enabler(std::shared_ptr<ExecutableImage> executable_image,
                        std::vector<DeviceInstance *> &&addressable_devices)
        : LoadedExecutableInstance(std::move(executable_image),
                                   std::move(addressable_devices)) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image),
                                               std::move(addressable_devices));
}

void LoadedExecutableInstance::bindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy = internal::onLoadedExecutableDestroy;
  api->PJRT_LoadedExecutable_GetExecutable =
      internal::onLoadedExecutableGetExecutable;
  api->PJRT_LoadedExecutable_AddressableDevices =
      internal::onLoadedExecutableAddressableDevices;
  api->PJRT_LoadedExecutable_Delete = internal::onLoadedExecutableDelete;
  api->PJRT_LoadedExecutable_IsDeleted = internal::onLoadedExecutableIsDeleted;
  api->PJRT_LoadedExecutable_Execute = internal::onLoadedExecutableExecute;
}

bool LoadedExecutableInstance::isDeleted() {
  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  return m_deleted;
}

void LoadedExecutableInstance::releaseResources() {
  if (m_runtime_device_opened) {
    tt::runtime::closeMeshDevice(*m_runtime_device);
  }

  if (m_deleted) {
    return;
  }

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
tt_pjrt_status
LoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args *args) {
  // Profiling: record start time
  auto start = std::chrono::high_resolution_clock::now();

  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::Execute");

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    DLOG_F(ERROR,
           "Requested number of devices to run the executable on (%zu) doesn't "
           "match the compiler estimated number of devices (%zu)",
           args->num_devices, m_executable_image->getNumDevicesToUtilize());
    // Profiling: record end time and print
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    DLOG_F(LOG_DEBUG,
           "[PROFILE] execute: early exit (device count mismatch) after %f s",
           elapsed.count());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    DLOG_F(ERROR,
           "Requested number of arguments to provide to the executable (%zu) "
           "doesn't match the compiler estimated number of inputs (%zu)",
           args->num_args, m_executable_image->getNumInputs());
    // Profiling: record end time and print
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    DLOG_F(LOG_DEBUG,
           "[PROFILE] execute: early exit (arg count mismatch) after %f s",
           elapsed.count());
    return tt_pjrt_status::kInternal;
  }

  if (!m_runtime_device_opened) {
    auto t0 = std::chrono::high_resolution_clock::now();
    m_runtime_device = openDevices(args->argument_lists, args->num_args,
                                   args->num_devices, args->execute_device);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> open_dev_time = t1 - t0;
    DLOG_F(LOG_DEBUG, "[PROFILE] execute: openDevices took %f s",
           open_dev_time.count());
    if (!m_runtime_device) {
      // Logging is done inside `openDevices`.
      // Profiling: record end time and print
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      DLOG_F(LOG_DEBUG,
             "[PROFILE] execute: early exit (openDevices failed) after %f s",
             elapsed.count());
      return tt_pjrt_status::kInternal;
    }
    m_runtime_device_opened = true;
  }
  std::optional<tt::runtime::Device> runtime_device = m_runtime_device;

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;

  std::vector<tt::runtime::Tensor> input_tensors;
  input_tensors.reserve(args->num_args);

  auto t_get_inputs_start = std::chrono::high_resolution_clock::now();
  tt_pjrt_status status = getInputRuntimeTensors(
      args->argument_lists, args->num_args, args->num_devices, *runtime_device,
      program_index, input_tensors);
  auto t_get_inputs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> get_inputs_time =
      t_get_inputs_end - t_get_inputs_start;
  DLOG_F(LOG_DEBUG, "[PROFILE] execute: getInputRuntimeTensors took %f s",
         get_inputs_time.count());

  if (!tt_pjrt_status_is_ok(status)) {
    // Profiling: record end time and print
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    DLOG_F(LOG_DEBUG,
           "[PROFILE] execute: early exit (getInputRuntimeTensors failed) "
           "after %f s",
           elapsed.count());
    return status;
  }

  auto t_submit_start = std::chrono::high_resolution_clock::now();
  std::vector<tt::runtime::Tensor> output_tensors = tt::runtime::submit(
      *runtime_device, m_executable_image->getFlatbufferBinary(), program_index,
      input_tensors);
  auto t_submit_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> submit_time = t_submit_end - t_submit_start;
  DLOG_F(LOG_DEBUG, "[PROFILE] execute: tt::runtime::submit took %f s",
         submit_time.count());

  if (output_tensors.size() != m_executable_image->getNumOutputs()) {
    DLOG_F(ERROR,
           "Runtime produced different number of output tensors (%zu) than the "
           "compiler estimated number of outputs (%zu)",
           output_tensors.size(), m_executable_image->getNumOutputs());
    // Profiling: record end time and print
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    DLOG_F(LOG_DEBUG,
           "[PROFILE] execute: early exit (output count mismatch) after %f s",
           elapsed.count());
    return tt_pjrt_status::kInternal;
  }

  auto t_untilize_start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<tt::runtime::Tensor>> untilized_output_tensors;
  untilized_output_tensors.reserve(output_tensors.size());
  status = untilizeToHost(output_tensors, args->num_devices,
                          untilized_output_tensors);
  auto t_untilize_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> untilize_time =
      t_untilize_end - t_untilize_start;
  DLOG_F(LOG_DEBUG, "[PROFILE] execute: untilizeToHost took %f s",
         untilize_time.count());

  if (!tt_pjrt_status_is_ok(status)) {
    // Profiling: record end time and print
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    DLOG_F(LOG_DEBUG,
           "[PROFILE] execute: early exit (untilizeToHost failed) after %f s",
           elapsed.count());
    return status;
  }

  auto t_fill_outputs_start = std::chrono::high_resolution_clock::now();
  fillPJRTOutputLists(untilized_output_tensors, args->num_devices,
                      args->output_lists, m_executable_image->getOutputTypes());
  auto t_fill_outputs_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fill_outputs_time =
      t_fill_outputs_end - t_fill_outputs_start;
  DLOG_F(LOG_DEBUG, "[PROFILE] execute: fillPJRTOutputLists took %f s",
         fill_outputs_time.count());

  auto t_dealloc_start = std::chrono::high_resolution_clock::now();
  for (size_t output_index = 0; output_index < output_tensors.size();
       ++output_index) {
    tt::runtime::deallocateTensor(output_tensors[output_index], /*force=*/true);
  }
  auto t_dealloc_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dealloc_time = t_dealloc_end - t_dealloc_start;
  DLOG_F(LOG_DEBUG,
         "[PROFILE] execute: deallocateTensor (all outputs) took %f s",
         dealloc_time.count());

  auto t_events_start = std::chrono::high_resolution_clock::now();
  if (args->device_complete_events) {
    for (int device_num = 0; device_num < args->num_devices; ++device_num) {
      std::unique_ptr<EventInstance> device_complete_event =
          EventInstance::createInstance();
      device_complete_event->markAsReady(tt_pjrt_status::kSuccess);

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Event_Destroy` on the event.
      args->device_complete_events[device_num] =
          *device_complete_event.release();
    }
  }
  auto t_events_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> events_time = t_events_end - t_events_start;
  DLOG_F(LOG_DEBUG, "[PROFILE] execute: device_complete_events took %f s",
         events_time.count());

  // Profiling: record end time and print
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  DLOG_F(LOG_DEBUG, "[PROFILE] execute: total time %f s", elapsed.count());

  // flush the output buffer (no longer needed with DLOG_F)

  return tt_pjrt_status::kSuccess;
}

std::optional<tt::runtime::Device>
LoadedExecutableInstance::openDevices(PJRT_Buffer *const *const *argument_lists,
                                      size_t num_args, size_t num_devices,
                                      PJRT_Device *pjrt_device) {
  std::unordered_set<int> device_ids =
      getDeviceIds(argument_lists, num_args, num_devices);

  const std::vector<std::uint32_t> &devices_mesh_shape =
      m_executable_image->getDevicesMeshShape();
  size_t mesh_shape_num_devices = static_cast<size_t>(
      std::accumulate(devices_mesh_shape.begin(), devices_mesh_shape.end(), 1,
                      std::multiplies<std::uint32_t>{}));

  if (device_ids.size() != mesh_shape_num_devices) {
    DLOG_F(ERROR,
           "Input buffers are placed on a different number of devices (%zu) "
           "than in the mesh shape estimated by the compiler (%zu)",
           device_ids.size(), mesh_shape_num_devices);
    return std::nullopt;
  }

  DeviceInstance *device_instance = DeviceInstance::unwrap(pjrt_device);
  if (device_instance &&
      !(device_ids.size() == 1 &&
        *device_ids.begin() == device_instance->getGlobalDeviceId())) {
    DLOG_F(ERROR, "Input buffers are placed on a different device than the one "
                  "specified in the execute_device argument");
    return std::nullopt;
  }

  // TODO(mrakita): Currently runtime doesn't allow us to open specific devices
  // subset, we can only open contiguous subset of devices starting from some
  // offset. We need to keep track of opened devices in Client and map the
  // buffers devices to these devices.
  // https://github.com/tenstorrent/tt-xla/issues/502

  return tt::runtime::openMeshDevice(devices_mesh_shape);
}

std::unordered_set<int> LoadedExecutableInstance::getDeviceIds(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices) {
  std::unordered_set<int> device_ids;

  for (size_t device_index = 0; num_args && device_index < num_devices;
       device_index++) {
    const BufferInstance *buffer =
        BufferInstance::unwrap(argument_lists[device_index][0]);
    int64_t buffer_device_id = buffer->getDevice()->getGlobalDeviceId();
    device_ids.emplace(buffer_device_id);
  }

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit. Maybe use `execute_device` from the args?
  if (device_ids.size() == 0) {
    assert(!m_addressable_devices.empty());
    device_ids.emplace(m_addressable_devices.front()->getGlobalDeviceId());
  }

  return device_ids;
}

tt_pjrt_status LoadedExecutableInstance::getInputRuntimeTensors(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices, const tt::runtime::Device &runtime_device,
    std::uint32_t program_index,
    std::vector<tt::runtime::Tensor> &input_tensors) {
  for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
    std::vector<tt::runtime::Tensor> arg_tensors;
    std::map<void *, BufferInstance *> buffer_map;
    arg_tensors.reserve(num_devices);

    for (size_t device_index = 0; device_index < num_devices; ++device_index) {
      BufferInstance *buffer =
          BufferInstance::unwrap(argument_lists[device_index][arg_index]);
      arg_tensors.push_back(buffer->getRuntimeTensor());
      buffer_map[buffer->getRuntimeTensor().handle.get()] = buffer;
    }

    mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
        mlir::tt::sharding_utils::MeshSharding::fillStrategyMapFromSharding(
            m_executable_image->getInputSharding(arg_index), num_devices);
    if (mlir::failed(strategy)) {
      DLOG_F(ERROR, "Failed to fill strategy map from sharding");
      return tt_pjrt_status::kInternal;
    }

    tt::runtime::Tensor input_tensor =
        getTensorFromStrategy(arg_tensors, *strategy);

    tt::runtime::Tensor laid_out_tensor = input_tensor;
    if (buffer_map[input_tensor.handle.get()]->needsLayoutConversion) {
      laid_out_tensor = convertTensorLayout(input_tensor, program_index,
                                            arg_index, runtime_device);
      buffer_map[input_tensor.handle.get()]->needsLayoutConversion = false;
      buffer_map[input_tensor.handle.get()]->setRuntimeTensor(laid_out_tensor);
    }

    // In case when new tensor was created, we want it to be automatically
    // deallocated during runtime.
    if (laid_out_tensor.data != input_tensor.data) {
      tt::runtime::setTensorRetain(laid_out_tensor, /*retain=*/false);
    }

    input_tensors.push_back(laid_out_tensor);
  }

  return tt_pjrt_status::kSuccess;
}

// TODO: We are using std::maps with strings as that is the way it is defined in
// the tt::runtime, instead of a more structured approach with structs and/or
// enums. See issue: https://github.com/tenstorrent/tt-mlir/issues/2513
tt::runtime::Tensor LoadedExecutableInstance::getTensorFromStrategy(
    const std::vector<tt::runtime::Tensor> &arg_tensors,
    const std::unordered_map<std::string, std::string> &strategy) {
  if (strategy.at("strategy") == "identity") {
    return arg_tensors.front();
  }

  tt::runtime::Tensor tensor = tt::runtime::createMultiDeviceHostTensor(
      arg_tensors, strategy, m_executable_image->getDevicesMeshShape());
  tt::runtime::setTensorRetain(tensor, /*retain=*/false);

  return tensor;
}

tt::runtime::Tensor LoadedExecutableInstance::convertTensorLayout(
    tt::runtime::Tensor input_tensor, std::uint32_t program_index,
    size_t arg_index, const tt::runtime::Device &runtime_device) {
  tt::runtime::Layout layout = tt::runtime::getLayout(
      m_executable_image->getFlatbufferBinary(), program_index, arg_index);

  return tt::runtime::toLayout(input_tensor, runtime_device, layout,
                               tt::runtime::getTensorRetain(input_tensor));
}

tt_pjrt_status LoadedExecutableInstance::untilizeToHost(
    const std::vector<tt::runtime::Tensor> &output_tensors, size_t num_devices,
    std::vector<std::vector<tt::runtime::Tensor>> &untilized_output_tensors) {
  for (size_t output_index = 0; output_index < output_tensors.size();
       ++output_index) {
    std::vector<tt::runtime::Tensor> untilized_output =
        tt::runtime::toHost(output_tensors[output_index], /* untilize */ true);

    // If the output is a replicated scalar, we expect only one tensor on
    // output, so we need to fill the rest of the output tensors with the same
    // tensors, to match the number of devices.
    if (untilized_output.size() != num_devices) {
      // If the output is not a scalar throw an error.
      if (getOutputShape(output_index).size() > 0 ||
          untilized_output.size() > 1) {
        DLOG_F(ERROR,
               "Untilize to host produced invalid number of output tensors: "
               "expected %zu, got %zu",
               num_devices, untilized_output.size());
        return tt_pjrt_status::kInternal;
      }
      for (size_t device_index = 1; device_index < num_devices;
           ++device_index) {
        untilized_output.emplace_back(untilized_output[0]);
      }
    }

    untilized_output_tensors.emplace_back(std::move(untilized_output));
  }

  return tt_pjrt_status::kSuccess;
}

void LoadedExecutableInstance::fillPJRTOutputLists(
    const std::vector<std::vector<tt::runtime::Tensor>>
        &untilized_output_tensors,
    size_t num_devices, PJRT_Buffer **const *output_lists,
    const std::vector<PJRT_Buffer_Type> &expected_output_data_types) {
  size_t num_outputs = untilized_output_tensors.size();

  for (int device_index = 0; device_index < num_devices; ++device_index) {
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      tt::runtime::Tensor output_tensor =
          untilized_output_tensors[output_index][device_index];
      std::vector<std::uint32_t> output_shape = getOutputShape(output_index);

      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createOutputBufferInstance(
              output_tensor, std::move(output_shape),
              m_addressable_devices[device_index],
              m_addressable_devices[device_index]->getDefaultMemory(),
              expected_output_data_types[output_index]);

      output_buffer->markAsDataReady();

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Buffer_Destroy` on the buffer.
      output_lists[device_index][output_index] = *output_buffer.release();
    }
  }
}

std::vector<std::uint32_t>
LoadedExecutableInstance::getOutputShape(size_t output_index) {
  std::vector<std::uint32_t> outputShape =
      m_executable_image->getOutputShape(output_index);
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      m_executable_image->getOutputSharding(output_index);

  if (outputSharding.getShardType() ==
          mlir::tt::ttcore::MeshShardType::Identity ||
      outputSharding.getShardType() ==
          mlir::tt::ttcore::MeshShardType::Replicate) {
    return outputShape;
  }

  llvm::ArrayRef<int64_t> shard_shape = outputSharding.getShardShape();
  assert(shard_shape.size() == outputShape.size() &&
         "Output sharding shape doesn't match the output shape");

  for (size_t i = 0; i < outputShape.size(); ++i) {
    assert(outputShape[i] % shard_shape[i] == 0 &&
           "Output shape is not divisible by the sharding shape");
    outputShape[i] /= shard_shape[i];
  }

  return outputShape;
}

namespace internal {

PJRT_Error *
onLoadedExecutableDestroy(PJRT_LoadedExecutable_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Destroy");

  delete LoadedExecutableInstance::unwrap(args->executable);

  return nullptr;
}

PJRT_Error *onLoadedExecutableGetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args *args) {
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_GetExecutable");

  LoadedExecutableInstance *loaded_executable =
      LoadedExecutableInstance::unwrap(args->loaded_executable);

  std::unique_ptr<ExecutableInstance> executable_instance =
      ExecutableInstance::createInstance(
          loaded_executable->getSharedExecutableImage());

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Executable_Destroy` on the executable.
  args->executable = *executable_instance.release();

  return nullptr;
}

PJRT_Error *onLoadedExecutableAddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args *args) {
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_AddressableDevices");

  LoadedExecutableInstance *loaded_executable =
      LoadedExecutableInstance::unwrap(args->executable);

  const std::vector<DeviceInstance *> &addressable_devices =
      loaded_executable->getAddressableDevices();

  args->addressable_devices =
      reinterpret_cast<PJRT_Device *const *>(addressable_devices.data());
  args->num_addressable_devices = addressable_devices.size();

  return nullptr;
}

PJRT_Error *onLoadedExecutableDelete(PJRT_LoadedExecutable_Delete_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");

  LoadedExecutableInstance::unwrap(args->executable)->releaseResources();

  return nullptr;
}

PJRT_Error *
onLoadedExecutableIsDeleted(PJRT_LoadedExecutable_IsDeleted_Args *args) {
  DLOG_F(LOG_DEBUG,
         "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted");

  args->is_deleted =
      LoadedExecutableInstance::unwrap(args->executable)->isDeleted();

  return nullptr;
}

PJRT_Error *
onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");

  tt_pjrt_status status =
      LoadedExecutableInstance::unwrap(args->executable)->execute(args);

  return *ErrorInstance::makeError(status).release();
}

} // namespace internal

} // namespace tt::pjrt
