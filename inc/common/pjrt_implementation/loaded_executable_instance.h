// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "xla/pjrt/c/pjrt_c_api.h"

#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/executable_image.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

struct ResidentExecutable {
  DeviceInstance *device_instance;
  size_t arg_count;
  size_t result_count;
};

class LoadedExecutableInstance {
public:
  LoadedExecutableInstance(
      ClientInstance &client, ExecutableImage *image,
      const std::vector<DeviceInstance *> &addressable_devices,
      size_t num_devices_to_utilize)
      : client_(client), image_(image),
        addressable_devices_(addressable_devices),
        num_devices_to_utilize_(num_devices_to_utilize) {}

  ~LoadedExecutableInstance();

  operator PJRT_LoadedExecutable *() {
    return reinterpret_cast<PJRT_LoadedExecutable *>(this);
  }
  static void BindApi(PJRT_Api *api);
  static LoadedExecutableInstance *Unwrap(PJRT_LoadedExecutable *exe) {
    return reinterpret_cast<LoadedExecutableInstance *>(exe);
  }

  const std::vector<DeviceInstance *> &addressable_devices() {
    return addressable_devices_;
  }

  size_t get_num_devices_to_utilize() const { return num_devices_to_utilize_; }

  // Loads all executables to addressable devices.
  tt_pjrt_status LoadAll();

  tt_pjrt_status GetDefaultResidentExecutable(ResidentExecutable **out_loaded);
  tt_pjrt_status GetArgResultCount(size_t *out_arg_count,
                                   size_t *out_result_count);

  tt_pjrt_status Execute(PJRT_LoadedExecutable_Execute_Args *args);

private:
  ClientInstance &client_;
  ExecutableImage *image_; // Ref-counted semantics.
  std::vector<DeviceInstance *> addressable_devices_;
  size_t num_devices_to_utilize_;
  std::vector<ResidentExecutable> resident_executables_;

  // Returns a runtime tensor given a creation strategy and a vestor of pointers
  // to data.
  tt::runtime::Tensor getTensorFromStrategy(
      const std::unordered_map<std::string, std::string> &strategy,
      BufferInstance *buffer, std::vector<std::shared_ptr<void>> &data);

  // Returns the shape of the output on the specified index.
  std::vector<std::uint32_t> getOuputShape(size_t index, size_t num_devices);

  // Returns is output on the specified index replicated.
  bool isOutputReplicated(size_t index) const;

  // Fills the output lists of the PJRT API with the outputs of tt runtime
  // execution.
  void fillPJRTOutputLists(
      const std::vector<std::vector<tt::runtime::Tensor>> &rt_outputs,
      const std::vector<tt::runtime::TensorDesc> &output_specs,
      size_t num_devices, PJRT_Buffer **const *output_lists);

  // Gets the device ids from the addressable devices.
  std::vector<int>
  getDeviceIds(PJRT_Buffer *const *const *argument_lists,
               const std::vector<DeviceInstance *> &addressable_devices,
               size_t num_args, size_t num_devices);

  // Given an output list, return the number of outputs of an executable.
  size_t getNumberOfOutputs(const std::vector<std::vector<tt::runtime::Tensor>>
                                &rt_outputs_list) const;

  // Return a tensor representing an output on a particular device with a
  // particular index.
  tt::runtime::Tensor
  getOuputTensor(size_t device_index, size_t output_index,
                 const std::vector<std::vector<tt::runtime::Tensor>>
                     &rt_outputs_list) const;
};

} // namespace tt::pjrt

#endif
