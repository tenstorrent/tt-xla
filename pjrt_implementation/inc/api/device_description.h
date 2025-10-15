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

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "ttmlir/Target/Common/types_generated.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_DEVICE_DESCRIPTION_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_DEVICE_DESCRIPTION_H_

namespace tt::pjrt {

// Represents PJRT_DeviceDescription structure and the functionality around it.
class DeviceDescription {
public:
  // Constructor.
  DeviceDescription(int32_t device_id, tt::target::Arch arch);

  // Binds PJRT API functions implementation related to PJRT_DeviceDescription
  // structure.
  static void bindApi(PJRT_Api *api);

  // Casts this device description instance to PJRT_DeviceDescription pointer.
  operator PJRT_DeviceDescription *() {
    return reinterpret_cast<PJRT_DeviceDescription *>(this);
  }

  // Casts the PJRT_DeviceDescription pointer to DeviceDescription pointer.
  static DeviceDescription *unwrap(PJRT_DeviceDescription *description) {
    return reinterpret_cast<DeviceDescription *>(description);
  }

  // Returns global device ID.
  int getDeviceId() const { return m_device_id; }

  // Returns a vendor-dependent string that uniquely identifies the kind of
  // device, e.g. `Wormhole_b0`.
  const std::string &getDeviceKind() const { return m_device_kind; }

  // Returns a debug string suitable for logging when errors occur. Should be
  // verbose enough to describe the current device unambiguously.
  const std::string &toDebugString() const { return m_user_string; }

  // Returns a device description string suitable for reading by the end users.
  const std::string &toString() const { return m_user_string; }

  // Returns process index from which this device is addressable.
  int getProcessIndex() const { return m_process_index; }

  // Sets process index from which this device is addressable.
  void setProcessIndex(int process_index) { m_process_index = process_index; }

private:
  // Global ID of this device unique between all devices.
  int m_device_id;

  // Process index from which this device is addressable.
  int m_process_index;

  // Vendor-dependent string that uniquely identifies the kind of device, e.g.
  // `Wormhole_b0`.
  std::string m_device_kind;

  // Device description string suitable for reading by the end users, should be
  // reasonably terse.
  std::string m_user_string;
};

namespace internal {

// Implements PJRT_DeviceDescription_Id API function.
PJRT_Error *onDeviceDescriptionId(PJRT_DeviceDescription_Id_Args *args);

// Implements PJRT_DeviceDescription_ProcessIndex API function.
PJRT_Error *
onDeviceDescriptionProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args *args);

// Implements PJRT_DeviceDescription_Attributes API function.
PJRT_Error *
onDeviceDescriptionAttributes(PJRT_DeviceDescription_Attributes_Args *args);

// Implements PJRT_DeviceDescription_Kind API function.
PJRT_Error *onDeviceDescriptionKind(PJRT_DeviceDescription_Kind_Args *args);

// Implements PJRT_DeviceDescription_DebugString API function.
PJRT_Error *
onDeviceDescriptionDebugString(PJRT_DeviceDescription_DebugString_Args *args);

// Implements PJRT_DeviceDescription_ToString API function.
PJRT_Error *
onDeviceDescriptionToString(PJRT_DeviceDescription_ToString_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_DEVICE_DESCRIPTION_H_
