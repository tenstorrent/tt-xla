// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <string>
#include <unordered_map>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

// TODO_OOM: Explain.
class LoadedExecutableInstance {
public:
  // TODO_OOM: Explain.
  LoadedExecutableInstance(
      ExecutableImage *executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      size_t num_devices_to_utilize)
      : m_executable_image(image), m_addressable_devices(addressable_devices),
        m_num_devices_to_utilize(num_devices_to_utilize) {}

  // TODO_OOM: Explain.
  ~LoadedExecutableInstance();

  // Binds PJRT API functions implementation related to PJRT_LoadedExecutable
  // structure.
  static void bindApi(PJRT_Api *api);

  // Casts this loaded executable instance to PJRT_LoadedExecutable and returns
  // pointer to it.
  operator PJRT_LoadedExecutable *() {
    return reinterpret_cast<PJRT_LoadedExecutable *>(this);
  }

  // Casts the PJRT_LoadedExecutable pointer to LoadedExecutableInstance
  // pointer.
  static LoadedExecutableInstance *unwrap(PJRT_LoadedExecutable *exe) {
    return reinterpret_cast<LoadedExecutableInstance *>(exe);
  }

  // TODO_OOM: Explain.
  const std::vector<DeviceInstance *> &addressable_devices() {
    return m_addressable_devices;
  }

  // TODO_OOM: Explain.
  size_t get_num_devices_to_utilize() const { return m_num_devices_to_utilize; }

  // TODO_OOM: Explain.
  tt_pjrt_status Execute(PJRT_LoadedExecutable_Execute_Args *args);

private:
  // Opens devices on which input arguments are placed, which we assume are the
  // the devices where computation will run.
  static tt::runtime::Device
  openDevices(PJRT_Buffer *const *const *argument_lists, size_t num_args,
              size_t num_devices,
              const std::vector<DeviceInstance *> &addressable_devices);

  // Collects device ids from the addressable devices.
  static std::vector<int>
  getDeviceIds(PJRT_Buffer *const *const *argument_lists, size_t num_args,
               size_t num_devices,
               const std::vector<DeviceInstance *> &addressable_devices);

  // Gets input runtime tensors from the arguments' buffers and converts them to
  // desired layout determined from the compiled graph.
  static std::vector<tt::runtime::Tensor> getInputRuntimeTensors(
      PJRT_Buffer *const *const *argument_lists, size_t num_args,
      size_t num_devices, tt::runtime::Device runtime_device,
      const tt::runtime::Binary &runtime_binary, std::uint32_t program_index,
      std::vector<tt::runtime::Tensor> &input_tensors);

  // Either returns single tensor or creates multi-device host tensor from arg
  // tensors, depending on the strategy.
  static tt::runtime::Tensor getTensorFromStrategy(
      const std::vector<tt::runtime::Tensor> &arg_tensors,
      const std::unordered_map<std::string, std::string> &strategy);

  // Untilizes output tensors and transfers them from device to host. Output
  // tensors are in [`num_devices`, `num_args`] format as PJRT expects.
  static tt_pjrt_status untilizeToHost(
      const std::vector<tt::runtime::Tensor> &output_tensors,
      std::vector<std::vector<tt::runtime::Tensor>> &untilized_output_tensors);

  // TODO_OOM: Check what can be made static

  // Fills the output lists of the PJRT API with the outputs of tt runtime
  // execution.
  void fillPJRTOutputLists(
      const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs,
      size_t num_devices, PJRT_Buffer **const *output_lists);

  // Given an output list, return the number of outputs of an executable.
  size_t getNumberOfOutputs(const std::vector<std::vector<tt::runtime::Tensor>>
                                &untilized_output_tensors) const;

  // Return a tensor representing an output on a particular device with a
  // particular index.
  tt::runtime::Tensor
  getOutputTensor(size_t device_index, size_t output_index,
                  const std::vector<std::vector<tt::runtime::Tensor>>
                      &untilized_output_tensors) const;

  // Returns is output on the specified index replicated.
  bool isOutputReplicated(size_t index) const;

  // Returns the shape of the output on the specified index.
  std::vector<std::uint32_t> getOutputShape(size_t index, size_t num_devices);

  // TODO_OOM: Explain.
  ExecutableImage *m_executable_image; // Ref-counted semantics.

  // TODO_OOM: Explain.
  const std::vector<DeviceInstance *> m_addressable_devices;

  // TODO_OOM: Explain.
  const size_t m_num_devices_to_utilize;
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
