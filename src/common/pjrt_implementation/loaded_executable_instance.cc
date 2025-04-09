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
#include <memory>

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

LoadedExecutableInstance::~LoadedExecutableInstance() {
  m_executable_image->DecRef();
}

void LoadedExecutableInstance::bindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy = internal::onLoadedExecutableDestroy;
  api->PJRT_LoadedExecutable_GetExecutable =
      internal::onLoadedExecutableGetExecutable;
  api->PJRT_LoadedExecutable_AddressableDevices =
      +[](PJRT_LoadedExecutable_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(
        LOG_DEBUG,
        "LoadedExecutableInstance::PJRT_LoadedExecutable_AddressableDevices");
    LoadedExecutableInstance *loaded_executable =
        LoadedExecutableInstance::unwrap(args->executable);
    // TODO: Set addressable devices in the loaded executable class to only the
    // devices being utilized, rather than all addressable devices. This way,
    // the number of devices will be determined by the list length instead of a
    // separate field in the class.
    const std::vector<DeviceInstance *> &addressable_devices =
        loaded_executable->addressable_devices();
    int num_addressable_devices =
        loaded_executable->get_num_devices_to_utilize();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(addressable_devices.data()));
    args->num_addressable_devices = num_addressable_devices;
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Execute =
      +[](PJRT_LoadedExecutable_Execute_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");

    // TODO_OOM: Check here if args->execute_device == nullptr

    return ErrorInstance::MakeError(
        LoadedExecutableInstance::unwrap(args->executable)->Execute(args));
  };
}

// TODO(mrakita): Make this method work in asynchronous fashion.
tt_pjrt_status
LoadedExecutableInstance::Execute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::Execute");

  // Check that the number of devices matches the number of devices counted
  // from the VHLO module.
  if (args->num_devices != m_num_devices_to_utilize) {
    DLOG_F(ERROR, "Number of devices in the executable does not match the "
                  "number of devices to utilize.");
    return tt_pjrt_status::kInternal;
  }

  // TODO_OOM: Check here if args->num_args matches num_args from image

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

  std::vector<tt::runtime::Tensor> output_tensors =
      tt::runtime::submit(runtime_device, m_executable_image->get_binary(),
                          program_index, input_tensors);

  // TODO_OOM: Check here if output_tensors.size() == m_image->num_outputs()

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
      // responsible for calling PJRT_Event_Destroy on event.
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
  std::vector<int> device_ids = getDeviceIds(
      argument_lists, num_args, num_devices, m_addressable_devices);

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
             "Untilize to host produced invalid number of output tensors, "
             "expected %zu got %zu",
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
      // responsible for calling PJRT_Buffer_Destroy on buffer.
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
      m_executable_image->get_output_shape(output_index);
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

  LoadedExecutableInstance *loaded_exe =
      LoadedExecutableInstance::unwrap(args->loaded_executable);
  ExecutableImage *image = loaded_exe->m_executable_image;

  image->AddRef();
  args->executable = *image;

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
