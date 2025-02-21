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

#include <unordered_set>
#include <iostream>

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/utils.h"
#include "tt/runtime/runtime.h"

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
    auto &addressable_devices =
        LoadedExecutableInstance::Unwrap(args->executable)
            ->addressable_devices();
    int num_addressable_devices =
        LoadedExecutableInstance::Unwrap(args->executable)
            ->image_->get_num_addresible_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(addressable_devices.data()));
    args->num_addressable_devices = num_addressable_devices;
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Delete =
      +[](PJRT_LoadedExecutable_Delete_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_LoadedExecutable_IsDeleted =
      +[](PJRT_LoadedExecutable_IsDeleted_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted_Args");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
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

  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
  size_t num_devices = args->num_devices;

  tt::runtime::Binary binary(image_->get_binary());

  std::vector<tt::runtime::Tensor> rt_inputs;

  std::unordered_set<int> device_ids;

  for (size_t i = 0; i < num_devices; i++) {
    BufferInstance *buffer =
      BufferInstance::Unwrap(args->argument_lists[i][0]);
    device_ids.insert(
      chip_ids[buffer->device().device_description()->getDeviceId()]);
  }
  std::cerr << "num_args=" << args->num_args << std::endl;
  std::cerr << "num_devices=" << num_devices << std::endl;

  for (size_t i = 0; i < args->num_args; ++i)
  {
    std::cerr << "i=" << i << std::endl;
    std::vector<std::shared_ptr<void>> data;
    BufferInstance* buffer;
    for (size_t j = 0; j < num_devices; ++j) {
      std::cerr << "j=" << j << std::endl;
      buffer =
        BufferInstance::Unwrap(args->argument_lists[j][i]);
      data.push_back(buffer->get_host_buffer_ptr());
      std::cerr << "j=" << j << std::endl;
    }
    std::pair<tt::target::DataType, size_t> tt_buffer_type = buffer->get_tt_buffer_type();
    std::unordered_map<std::string, std::string> strategy;
    if (i==0)
    {
      strategy = {{"strategy", "shard_2d"}, {"mesh_shape_y", "1"}, {"mesh_shape_x" , "2"}};
    }
    else if (i==1)
    {
      strategy = {{"strategy", "shard_2d"}, {"mesh_shape_y", "1"}, {"mesh_shape_x" , "2"}};
    }
    tt::runtime::TensorDesc tensor_desc = { buffer->get_shape(), buffer->get_stride(), static_cast<std::uint32_t>(tt_buffer_type.second), tt_buffer_type.first};
    std::cerr << "before" << std::endl;
    tt::runtime::Tensor multichip_tensor = tt::runtime::createTensor(data, tensor_desc, strategy);
    std::cerr << "after" << std::endl;
    rt_inputs.push_back(multichip_tensor);
    std::cerr << "i=" << i << std::endl;
  }

  std::cerr << "I AM HERE" << std::endl;

  /*for (size_t i = 0; i < args->num_args; ++i) {
    BufferInstance *buffer =
        BufferInstance::Unwrap(args->argument_lists[0][i]);
    rt_inputs.push_back(buffer->getTensor());
    DLOG_F(INFO, "Runtime input id: %d", buffer->unique_id());
  }*/

  assert(device_ids.size() == num_devices);

  std::vector<int> device_ids_vector(device_ids.begin(), device_ids.end());

  tt::runtime::Device device = tt::runtime::openDevice(device_ids_vector);
  std::vector<tt::runtime::Tensor> input_tensors;
  int size_inputs = rt_inputs.size();
  std::vector<tt::runtime::Tensor> rt_outputs = tt::runtime::submit(device, binary, 0, rt_inputs);
  std::vector<tt::runtime::TensorDesc> output_specs =
      binary.getProgramOutputs(0);
  std::cerr << "OUTPUTS=" << rt_outputs.size() << std::endl;
  std::vector<std::vector<tt::runtime::Tensor>> rt_outputs_list(num_devices);

  for (size_t i = 0; i < output_specs.size(); ++i) {
    std::cerr << "i=" << i << std::endl;
    bool is_scalar = client_.isOutputScalar(i);
    // PJRT expects an empty shape for scalars.
    std::vector<std::uint32_t> output_shape =
        is_scalar ? std::vector<std::uint32_t>() : output_specs[i].shape;
    std::vector<tt::runtime::Tensor> untilized_multichip_output = tt::runtime::multiDeviceToHost(rt_outputs[i], true);
    std::cerr << "tralalala=" << untilized_multichip_output.size() << std::endl;
    for (size_t j=0;j<untilized_multichip_output.size();j++)
    {
      std::cerr << "j=" << j << std::endl;
      rt_outputs_list[j].push_back(untilized_multichip_output[j]);
    }
  }

  for (int i=0;i<rt_outputs_list.size();i++)
  {
    std::cerr << "ggg=" << rt_outputs_list[i].size() << std::endl;
  }

  for (int k=0;k<num_devices;k++)
  {
    std::cerr << "k=" << k << std::endl;
    for (size_t i = 0; i < rt_outputs_list[k].size(); ++i) {
      std::cerr << "i=" << i << std::endl;
      bool is_scalar = client_.isOutputScalar(i);
      // PJRT expects an empty shape for scalars.
      std::vector<std::uint32_t> output_shape =
          is_scalar ? std::vector<std::uint32_t>() : output_specs[i].shape;
      output_shape[1] = output_shape[1]/2;
      std::pair<tt::target::DataType, size_t> type_pair = {output_specs[i].dataType, output_specs[i].itemsize};
      std::cerr << "yyyy=" << output_specs[i].shape[0] << "  " << output_specs[i].shape[1] << std::endl;
      auto result_buffer = std::make_unique<BufferInstance>(
          *this->addressable_devices_[k], rt_outputs_list[k][i], output_shape,
          output_specs[i].stride, type_pair);
      std::cerr << "zzzz" << std::endl;
      result_buffer->setType(tt::pjrt::utils::convertElementTypeToBufferType(
          output_specs[i].dataType));
      DLOG_F(INFO, "Runtime output id: %d", result_buffer->unique_id());
      args->output_lists[k][i] = *(result_buffer.release());
    }
  }
  std::cerr << "WEEEE" << std::endl;
  for (size_t i = 0; i < rt_outputs.size(); ++i) {
    tt::runtime::deallocateTensor(rt_outputs[i], /*force=*/true);
  }

  if (args->device_complete_events) {
    for (int i=0;i<num_devices;i++) args->device_complete_events[i] = *(new EventInstance());
  }

  std::cerr << "tttt" << std::endl;

  tt::runtime::closeDevice(device);

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt
