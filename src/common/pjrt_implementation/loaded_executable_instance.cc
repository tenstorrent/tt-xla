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

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

// tt-xla includes
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/error_instance.h"

namespace tt::pjrt {

std::unique_ptr<LoadedExecutableInstance>
LoadedExecutableInstance::createInstance(
    std::shared_ptr<ExecutableImage> executable_image,
    const std::vector<DeviceInstance *> &addressable_devices) {
  return std::make_unique<LoadedExecutableInstance>(std::move(executable_image),
                                                    addressable_devices);
}

void LoadedExecutableInstance::bindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy = internal::onLoadedExecutableDestroy;
  api->PJRT_LoadedExecutable_GetExecutable =
      internal::onLoadedExecutableGetExecutable;
  api->PJRT_LoadedExecutable_AddressableDevices =
      internal::onLoadedExecutableAddressableDevices;
  api->PJRT_LoadedExecutable_Delete = internal::onLoadedExecutableDelete;
  api->PJRT_LoadedExecutable_IsDeleted = internal::onLoadedExecutableIsDeleted;
  api->PJRT_LoadedExecutable_Execute = internal::;
}

bool LoadedExecutableInstance::isDeleted() {
  std::lock_guard<std::mutex> deleted_lock(m_deleted_mutex);
  return m_deleted;
}

void LoadedExecutableInstance::releaseResources() {
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
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::Execute");

  if (args->num_devices != m_executable_image->getNumDevicesToUtilize()) {
    DLOG_F(ERROR,
           "Requested number of devices to run the executable on (%zu) doesn't "
           "match the compiler estimated number of devices (%zu)",
           args->num_devices, m_executable_image->getNumDevicesToUtilize());
    return tt_pjrt_status::kInternal;
  }

  if (args->num_args != m_executable_image->getNumInputs()) {
    DLOG_F(ERROR,
           "Requested number of arguments to provide to the executable (%zu) "
           "doesn't match the compiler estimated number of inputs (%zu)",
           args->num_args, m_executable_image->getNumInputs());
    return tt_pjrt_status::kInternal;
  }

  tt::runtime::Device runtime_device =
      openDevices(args->argument_lists, args->num_args, args->num_devices);

  // Assuming only one program per flatbuffer for now.
  std::uint32_t program_index = 0;

  std::vector<tt::runtime::Tensor> input_tensors;
  input_tensors.reserve(num_args);
  tt_pjrt_status status = getInputRuntimeTensors(
      args->argument_lists, args->num_args, args->num_devices, runtime_device,
      program_index, input_tensors);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }

  // Multichip support is only enabled if the toLayoutAPIAssumeSingleChip
  // workaround flag is turned off, which the line below does.
  // See issue: https://github.com/tenstorrent/tt-xla/issues/373
  tt::runtime::workaround::Env::get(true, true, false);

  std::vector<tt::runtime::Tensor> output_tensors = tt::runtime::submit(
      runtime_device, m_executable_image->getFlatbufferBinary(), program_index,
      input_tensors);

  if (output_tensors.size() != m_executable_image->getNumOutputs()) {
    DLOG_F(ERROR,
           "Runtime produced different number of output tensors (%zu) than the "
           "compiler estimated number of outputs (%zu)",
           output_tensors.size(), m_executable_image->getNumOutputs());
    return tt_pjrt_status::kInternal;
  }

  std::vector<std::vector<tt::runtime::Tensor>> untilized_output_tensors;
  untilized_output_tensors.reserve(output_tensors.size());
  status = untilizeToHost(output_tensors, untilized_output_tensors);
  if (!tt_pjrt_status_is_ok(status)) {
    return status;
  }

  fillPJRTOutputLists(untilized_output_tensors, args->num_devices,
                      args->output_lists);

  for (size_t output_index = 0; output_index < output_tensors.size();
       ++output_index) {
    tt::runtime::deallocateTensor(output_tensors[output_index], /*force=*/true);
  }

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

  tt::runtime::closeMeshDevice(runtime_device);

  return tt_pjrt_status::kSuccess;
}

tt::runtime::Device
LoadedExecutableInstance::openDevices(PJRT_Buffer *const *const *argument_lists,
                                      size_t num_args, size_t num_devices) {
  std::vector<int> device_ids =
      getDeviceIds(argument_lists, num_args, num_devices);

  tt::runtime::MeshDeviceOptions options;
  const std::vector<uint32_t> mesh_shape = {
      1, static_cast<uint32_t>(device_ids.size())};
  options.deviceIds = device_ids;

  return tt::runtime::openMeshDevice(mesh_shape, options);
}

std::vector<int> LoadedExecutableInstance::getDeviceIds(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices) {
  std::vector<int> device_ids;

  for (size_t device_index = 0; num_args && device_index < num_devices;
       device_index++) {
    const BufferInstance *buffer =
        BufferInstance::unwrap(argument_lists[device_index][0]);
    int64_t buffer_device_id = buffer->device().getGlobalDeviceId();
    device_ids.push_back(buffer_device_id);
  }

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit. Maybe use `execute_device` from the args?
  if (device_ids.size() == 0) {
    assert(!m_addressable_devices.empty());
    device_ids.push_back(m_addressable_devices.front()->getGlobalDeviceId());
  }

  return device_ids;
}

tt_pjrt_status LoadedExecutableInstance::getInputRuntimeTensors(
    PJRT_Buffer *const *const *argument_lists, size_t num_args,
    size_t num_devices, tt::runtime::Device runtime_device,
    std::uint32_t program_index,
    std::vector<tt::runtime::Tensor> &input_tensors) {
  for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
    std::vector<tt::runtime::Tensor> arg_tensors;
    arg_tensors.reserve(num_devices);

    for (size_t device_index = 0; device_index < num_devices; ++device_index) {
      BufferInstance *buffer =
          BufferInstance::unwrap(argument_lists[device_index][arg_index]);
      arg_tensors.push_back(buffer->getRuntimeTensor());
    }

    mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
        mlir::tt::sharding_utils::MeshSharding::fillStrategyMapFromSharding(
            m_executable_image->getInputSharding(arg_num), num_devices);
    if (mlir::failed(strategy)) {
      DLOG_F(ERROR, "Failed to fill strategy map from sharding");
      return tt_pjrt_status::kInternal;
    }

    tt::runtime::Tensor input_tensor =
        getTensorFromStrategy(arg_tensors, strategy);

    // Converting input tensor to desired layout, this might move it on device.
    tt::runtime::Layout layout =
        tt::runtime::getLayout(binary, program_index, arg_index);
    tt::runtime::Tensor laid_out_tensor =
        tt::runtime::toLayout(input_tensor, runtime_device, layout,
                              tt::runtime::getTensorRetain(input_tensor));

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

  tt::runtime::Tensor tensor =
      tt::runtime::createMultiDeviceHostTensor(arg_tensors, strategy);
  tt::runtime::setTensorRetain(tensor, /*retain=*/false);

  return tensor;
}

tt_pjrt_status LoadedExecutableInstance::untilizeToHost(
    const std::vector<tt::runtime::Tensor> &output_tensors,
    std::vector<std::vector<tt::runtime::Tensor>> &untilized_output_tensors) {
  for (size_t output_index = 0; output_index < output_tensors.size();
       ++output_index) {
    std::vector<tt::runtime::Tensor> untilized_output =
        tt::runtime::toHost(output_tensors[output_index], /* untilize */ true);

    size_t expected_num_outputs =
        isOutputReplicated(output_index) ? 1 : untilized_output_tensors.size();
    if (untilized_output.size() != expected_num_outputs) {
      DLOG_F(ERROR,
             "Untilize to host produced invalid number of output tensors: "
             "expected %zu, got %zu",
             expected_num_outputs, untilized_output.size());
      return tt_pjrt_status::kInternal;
    }

    untilized_output_tensors.emplace_back(std::move(untilized_output));
  }

  return tt_pjrt_status::kSuccess;
}

void LoadedExecutableInstance::fillPJRTOutputLists(
    const std::vector<std::vector<tt::runtime::Tensor>>
        &untilized_output_tensors,
    size_t num_devices, PJRT_Buffer **const *output_lists) {
  size_t num_outputs = untilized_output_tensors.size();

  for (int device_index = 0; device_index < num_devices; device_index++) {
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      tt::runtime::Tensor output_tensor =
          getOutputTensor(device_index, output_index, untilized_output_tensors);
      std::vector<std::uint32_t> output_shape =
          getOutputShape(output_index, num_devices);

      std::unique_ptr<BufferInstance> output_buffer =
          BufferInstance::createOutputBufferInstance(
              output_tensor, output_shape,
              this->m_addressable_devices[device_index]);

      output_buffer->markAsDataReady();

      // Releasing the ownership to the PJRT API caller since the caller is
      // responsible for calling `PJRT_Buffer_Destroy` on the buffer.
      output_lists[device_index][output_index] = *result_buffer.release();
    }
  }
}

tt::runtime::Tensor LoadedExecutableInstance::getOutputTensor(
    size_t device_index, size_t output_index,
    const std::vector<std::vector<tt::runtime::Tensor>>
        &untilized_output_tensors) const {
  // If the output is replicated, we just return the output tensor from the
  // first device, as this is what PJRT expects.
  return isOutputReplicated(output_index)
             ? untilized_output_tensors[output_index][0]
             : untilized_output_tensors[output_index][device_index];
}

bool LoadedExecutableInstance::isOutputReplicated(size_t output_index) const {
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      m_executable_image->getOutputSharding(output_index);
  return outputSharding.getShardType() == mlir::tt::MeshShardType::Replicate;
}

std::vector<std::uint32_t>
LoadedExecutableInstance::getOutputShape(size_t output_index,
                                         size_t num_devices) {
  std::vector<std::uint32_t> outputShape =
      m_executable_image->getOutputShape(output_index);
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      m_executable_image->getOutputSharding(output_index);

  mlir::FailureOr<std::unordered_map<std::string, std::string>>
      shardingStrategy =
          mlir::tt::sharding_utils::MeshSharding::fillStrategyMapFromSharding(
              outputSharding, num_devices);
  if (mlir::failed(shardingStrategy)) {
    DLOG_F(WARNING, "No valid output sharding, returning the original shape");
    return outputShape;
  }

  if (shardingStrategy->at("strategy") == "shard") {
    assert(!outputShape.empty());
    outputShape[0] /= num_devices;
  } else if (shardingStrategy->at("strategy") == "shard_2d") {
    assert(!outputShape.empty());
    assert(outputShape[0] % std::stoi(shardingStrategy->at("mesh_shape_y")) ==
           0);
    assert(outputShape[1] % std::stoi(shardingStrategy->at("mesh_shape_x")) ==
           0);
    outputShape[0] =
        outputShape[0] / std::stoi(shardingStrategy->at("mesh_shape_y"));
    outputShape[1] =
        outputShape[1] / std::stoi(shardingStrategy->at("mesh_shape_x"));
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
  args->executable = *loaded_executable->executable_instance.release();

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

  if (args->execute_device) {
    DLOG_F(ERROR, "Executing on specific single device is not supported");
    return ErrorInstance::makeError(tt_pjrt_status::kUnimplemented);
  }

  return ErrorInstance::makeError(
      LoadedExecutableInstance::unwrap(args->executable)->Execute(args));
}

} // namespace internal

} // namespace tt::pjrt
