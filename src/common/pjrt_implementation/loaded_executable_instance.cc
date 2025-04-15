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
#include "common/status.h"

#include <string>
#include <unordered_map>

// tt-mlir includes
#define TTMLIR_ENABLE_STABLEHLO 1
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/utils.h"

namespace tt::pjrt {

LoadedExecutableInstance::~LoadedExecutableInstance() { image_->DecRef(); }

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
    LoadedExecutableInstance *loaded_executable =
        LoadedExecutableInstance::Unwrap(args->executable);
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
    return ErrorInstance::MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->Execute(args));
  };
  api->PJRT_LoadedExecutable_GetExecutable =
      +[](PJRT_LoadedExecutable_GetExecutable_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_GetExecutable");
    LoadedExecutableInstance *loaded_exe =
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

  // Check that the number of devices matches the number of devices counted
  // from the VHLO module.
  size_t num_devices = args->num_devices;
  if (num_devices != num_devices_to_utilize_) {
    DLOG_F(ERROR, "Number of devices in the executable does not match the "
                  "number of devices to utilize.");
    return tt_pjrt_status::kInternal;
  }

  const tt::runtime::Binary &binary = image_->get_binary();

  std::vector<tt::runtime::Tensor> rt_inputs;
  for (size_t arg_num = 0; arg_num < args->num_args; ++arg_num) {
    std::vector<const void *> data;
    BufferInstance *buffer;
    for (size_t device_index = 0; device_index < num_devices; ++device_index) {
      buffer =
          BufferInstance::Unwrap(args->argument_lists[device_index][arg_num]);
      data.push_back(buffer->get_host_buffer_ptr().get());
    }
    mlir::FailureOr<std::unordered_map<std::string, std::string>> strategy =
        mlir::tt::sharding_utils::MeshSharding::fillStrategyMapFromSharding(
            image_->getInputSharding(arg_num), num_devices);
    if (mlir::failed(strategy)) {
      DLOG_F(ERROR, "Failed to fill strategy map from sharding");
      return tt_pjrt_status::kInternal;
    }
    // As all the buffers that correspond to the same argument have the same
    // tensor descriptors (shape, stride, etc), we can just use the last one to
    // get the needed information.
    rt_inputs.push_back(getTensorFromStrategy(*strategy, buffer, data));
  }

  std::vector<int> device_ids = getDeviceIds(
      args->argument_lists, addressable_devices_, args->num_args, num_devices);

  tt::runtime::MeshDeviceOptions options;
  const std::vector<std::uint32_t> &mesh_shape = image_->get_mesh_shape();
  tt::runtime::Device device = tt::runtime::openMeshDevice(mesh_shape, options);
  std::vector<tt::runtime::Tensor> input_tensors;
  int size_inputs = rt_inputs.size();

  // Multichip support is only enabled if the toLayoutAPIAssumeSingleChip
  // workaround flag is turned off, which the line below does.
  // See issue: https://github.com/tenstorrent/tt-xla/issues/373
  tt::runtime::workaround::Env::get(true, true, false);

  std::vector<tt::runtime::Tensor> rt_inputs_with_layout;
  rt_inputs_with_layout.reserve(rt_inputs.size());
  std::transform(rt_inputs.begin(), rt_inputs.end(),
                 std::back_inserter(rt_inputs_with_layout),
                 [&](tt::runtime::Tensor &t) -> tt::runtime::Tensor {
                   tt::runtime::Layout layout =
                       tt::runtime::getLayout(binary, 0 /* program_index */,
                                              rt_inputs_with_layout.size());

                   tt::runtime::Tensor tensor =
                       tt::runtime::toLayout(t, device, layout, true);
                   return tensor;
                 });

  std::vector<tt::runtime::Tensor> rt_outputs = tt::runtime::submit(
      device, binary, 0 /* program_index */, rt_inputs_with_layout);
  std::vector<tt::runtime::TensorDesc> output_specs =
      binary.getProgramOutputs(0 /* program_index */);
  std::vector<std::vector<tt::runtime::Tensor>> rt_outputs_list(num_devices);

  for (size_t output_index = 0; output_index < output_specs.size();
       ++output_index) {
    std::vector<tt::runtime::Tensor> untilized_output =
        tt::runtime::toHost(rt_outputs[output_index], /* untilize */ true);
    for (size_t shard_index = 0; shard_index < untilized_output.size();
         shard_index++) {
      rt_outputs_list[shard_index].push_back(untilized_output[shard_index]);
    }
  }

  fillPJRTOutputLists(rt_outputs_list, output_specs, num_devices,
                      args->output_lists);

  for (size_t output_num = 0; output_num < rt_outputs.size(); ++output_num) {
    tt::runtime::deallocateTensor(rt_outputs[output_num], /*force=*/true);
  }

  if (args->device_complete_events) {
    for (int device_num = 0; device_num < num_devices; device_num++) {
      args->device_complete_events[device_num] = *(new EventInstance());
    }
  }

  tt::runtime::closeMeshDevice(device);

  return tt_pjrt_status::kSuccess;
}

// TODO: We are using std::maps with strings as that is the way it is defined in
// the tt::runtime, instead of a more structured approach with structs and/or
// enums. See issue: https://github.com/tenstorrent/tt-mlir/issues/2513
tt::runtime::Tensor LoadedExecutableInstance::getTensorFromStrategy(
    const std::unordered_map<std::string, std::string> &strategy,
    BufferInstance *buffer, std::vector<const void *> &data) {
  if (strategy.at("strategy") == "identity") {
    return buffer->getTensor();
  }
  std::pair<tt::target::DataType, size_t> tt_buffer_type =
      buffer->get_tt_buffer_type();
  tt::runtime::TensorDesc tensor_desc = {
      buffer->getDimensions(), buffer->get_stride(),
      static_cast<std::uint32_t>(tt_buffer_type.second), tt_buffer_type.first};
  return tt::runtime::createOwnedMultiDeviceHostTensor(data, tensor_desc,
                                                       strategy);
}

std::vector<std::uint32_t>
LoadedExecutableInstance::getOuputShape(size_t index, size_t num_devices) {
  std::vector<std::uint32_t> outputShape = image_->get_output_shape(index);
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      image_->getOutputSharding(index);
  mlir::FailureOr<std::unordered_map<std::string, std::string>>
      shardingStrategy =
          mlir::tt::sharding_utils::MeshSharding::fillStrategyMapFromSharding(
              outputSharding, num_devices);
  if (mlir::failed(shardingStrategy)) {
    DLOG_F(WARNING, "No valid output sharding, returning the original shape");
    return outputShape;
  }
  if (shardingStrategy->at("strategy") == "shard") {
    outputShape[0] /= num_devices;
  } else if (shardingStrategy->at("strategy") == "shard_2d") {
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

bool LoadedExecutableInstance::isOutputReplicated(size_t index) const {
  const mlir::tt::sharding_utils::MeshSharding &outputSharding =
      image_->getOutputSharding(index);
  return outputSharding.getShardType() == mlir::tt::MeshShardType::Replicate;
}

void LoadedExecutableInstance::fillPJRTOutputLists(
    const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs_list,
    const std::vector<tt::runtime::TensorDesc> &output_specs,
    size_t num_devices, PJRT_Buffer **const *output_lists) {
  size_t num_outputs = getNumberOfOutputs(rt_outputs_list);

  assert(num_outputs == output_specs.size());

  for (int device_index = 0; device_index < num_devices; device_index++) {
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      tt::runtime::Tensor output_tensor =
          getOuputTensor(device_index, output_index, rt_outputs_list);
      std::vector<std::uint32_t> output_shape =
          getOuputShape(output_index, num_devices);
      std::pair<tt::target::DataType, size_t> type_pair = {
          output_specs[output_index].dataType,
          output_specs[output_index].itemsize};
      auto result_buffer = std::make_unique<BufferInstance>(
          *this->addressable_devices_[device_index], output_tensor,
          output_shape, output_specs[output_index].stride, type_pair);
      result_buffer->setType(tt::pjrt::utils::convertElementTypeToBufferType(
          output_specs[output_index].dataType));
      output_lists[device_index][output_index] = *(result_buffer.release());
    }
  }
}

std::vector<int> LoadedExecutableInstance::getDeviceIds(
    PJRT_Buffer *const *const *argument_lists,
    const std::vector<DeviceInstance *> &addressable_devices, size_t num_args,
    size_t num_devices) {
  std::vector<int> device_ids;

  for (size_t device_index = 0; num_args && device_index < num_devices;
       device_index++) {
    const BufferInstance *buffer =
        BufferInstance::Unwrap(argument_lists[device_index][0]);
    int64_t buffer_device_id =
        buffer->device().device_description()->getDeviceId();
    device_ids.push_back(buffer_device_id);
  }

  // If there are no input buffers, we still want to run on a device.
  // TODO: Now we will run only on the first one, but this should be somehow
  // explicit.
  if (device_ids.size() == 0) {
    device_ids.push_back(
        addressable_devices_[0]->device_description()->getDeviceId());
  }

  return device_ids;
}

size_t LoadedExecutableInstance::getNumberOfOutputs(
    const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs_list)
    const {
  size_t num_outputs = rt_outputs_list[0].size();
  for (size_t i = 0; i < rt_outputs_list.size(); i++) {
    num_outputs = std::max(num_outputs, rt_outputs_list[i].size());
  }
  return num_outputs;
}

tt::runtime::Tensor LoadedExecutableInstance::getOuputTensor(
    size_t device_index, size_t output_index,
    const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs_list)
    const {
  // If the output is replicated, we just return the output tensor in the first
  // device devices, as this is what PJRT expects.
  return isOutputReplicated(output_index)
             ? rt_outputs_list[0][output_index]
             : rt_outputs_list[device_index][output_index];
}

} // namespace tt::pjrt
