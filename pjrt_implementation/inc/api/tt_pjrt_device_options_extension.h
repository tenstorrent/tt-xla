// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_TT_PJRT_DEVICE_OPTIONS_EXTENSION_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_TT_PJRT_DEVICE_OPTIONS_EXTENSION_H_

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// TT Device Options Extensions
// =============================================================================
// TT-specific PJRT extensions for device options management.
// Two mechanisms are provided:
//
// 1. Execute Options Extension (PJRT_TT_DeviceOptions_Extension):
//    Attached to PJRT_ExecuteOptions.extension_start to pass device options
//    with each execute call.
//
// 2. API Extension (PJRT_TT_SetDeviceOptions):
//    Attached to PJRT_Api.extension_start to allow setting device options
//    on a device before execution. This is useful when the execute path
//    doesn't support passing custom extensions.
// =============================================================================

// Extension type IDs for TT extensions.
// Using values in the vendor-specific range (>= 0x1000).
#define PJRT_Extension_Type_TT_DeviceOptions ((PJRT_Extension_Type)0x1000)
#define PJRT_Extension_Type_TT_SetDeviceOptions ((PJRT_Extension_Type)0x1001)

// =============================================================================
// Execute Options Extension (attached to PJRT_ExecuteOptions.extension_start)
// =============================================================================

// TT Device Options Extension structure for execute options.
struct PJRT_TT_DeviceOptions_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;  // Must be PJRT_Extension_Type_TT_DeviceOptions
  PJRT_Extension_Base *next;
  // Device options as key-value pairs.
  const PJRT_NamedValue *device_options;
  size_t num_device_options;
};
#define PJRT_TT_DeviceOptions_Extension_STRUCT_SIZE                            \
  PJRT_STRUCT_SIZE(PJRT_TT_DeviceOptions_Extension, num_device_options)

// =============================================================================
// API Extension (attached to PJRT_Api.extension_start)
// =============================================================================

// Arguments for PJRT_TT_SetDeviceOptions call.
struct PJRT_TT_SetDeviceOptions_Args {
  size_t struct_size;
  PJRT_Extension_Base *extension_start;
  // Device ID to set options for.
  int device_id;
  // Device options as key-value pairs.
  const PJRT_NamedValue *device_options;
  size_t num_device_options;
};
#define PJRT_TT_SetDeviceOptions_Args_STRUCT_SIZE                              \
  PJRT_STRUCT_SIZE(PJRT_TT_SetDeviceOptions_Args, num_device_options)

// Function pointer type for PJRT_TT_SetDeviceOptions.
typedef PJRT_Error *(*PJRT_TT_SetDeviceOptions)(
    PJRT_TT_SetDeviceOptions_Args *args);

// TT SetDeviceOptions API Extension structure.
// This extension is linked from PJRT_Api.extension_start.
struct PJRT_TT_SetDeviceOptions_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;  // Must be PJRT_Extension_Type_TT_SetDeviceOptions
  PJRT_Extension_Base *next;
  // Function to set device options.
  PJRT_TT_SetDeviceOptions set_device_options;
};
#define PJRT_TT_SetDeviceOptions_Extension_STRUCT_SIZE                         \
  PJRT_STRUCT_SIZE(PJRT_TT_SetDeviceOptions_Extension, set_device_options)

#ifdef __cplusplus
}
#endif

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_TT_PJRT_DEVICE_OPTIONS_EXTENSION_H_
