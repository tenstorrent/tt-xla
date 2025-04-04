// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

  // ---------------------------------------------------------------------------
  // PJRT_Buffer stubs:

  // Deprecated in v0.54
  _STUB(PJRT_Buffer_GetMemoryLayout);

  // Used to copy raw buffer from device to host. In order to support this we
  // need flexible memcpy from runtime in order to specify the offset and size.
  _STUB(PJRT_Buffer_OnDeviceSizeInBytes);
  _STUB(PJRT_Buffer_CopyRawToHost);
  // ---------------------------------------------------------------------------

  _STUB(PJRT_Error_Destroy);
  _STUB(PJRT_Error_Message);
  _STUB(PJRT_Error_GetCode);
  _STUB(PJRT_Plugin_Initialize);
  _STUB(PJRT_Plugin_Attributes);
  _STUB(PJRT_Client_Create);
  _STUB(PJRT_Client_Destroy);
  _STUB(PJRT_Client_PlatformName);
  _STUB(PJRT_Client_ProcessIndex);
  _STUB(PJRT_Client_PlatformVersion);
  _STUB(PJRT_Client_Devices);
  _STUB(PJRT_Client_AddressableDevices);
  _STUB(PJRT_Client_LookupDevice);
  _STUB(PJRT_Client_LookupAddressableDevice);
  _STUB(PJRT_Client_AddressableMemories);
  _STUB(PJRT_Client_Compile);
  _STUB(PJRT_Client_DefaultDeviceAssignment);
  _STUB(PJRT_Client_BufferFromHostBuffer);
  _STUB(PJRT_DeviceDescription_Id);
  _STUB(PJRT_DeviceDescription_ProcessIndex);
  _STUB(PJRT_DeviceDescription_Attributes);
  _STUB(PJRT_DeviceDescription_Kind);
  _STUB(PJRT_DeviceDescription_DebugString);
  _STUB(PJRT_DeviceDescription_ToString);
  _STUB(PJRT_Device_AddressableMemories);
  _STUB(PJRT_Device_DefaultMemory);
  _STUB(PJRT_Device_MemoryStats);
  _STUB(PJRT_Memory_Id);
  _STUB(PJRT_Memory_Kind);
  _STUB(PJRT_Memory_DebugString);
  _STUB(PJRT_Memory_ToString);
  _STUB(PJRT_Memory_AddressableByDevices);
  _STUB(PJRT_Executable_Destroy);
  _STUB(PJRT_Executable_Name);
  _STUB(PJRT_Executable_NumReplicas);
  _STUB(PJRT_Executable_NumPartitions);
  _STUB(PJRT_Executable_NumOutputs);
  _STUB(PJRT_Executable_SizeOfGeneratedCodeInBytes);
  _STUB(PJRT_Executable_GetCostAnalysis);
  _STUB(PJRT_Executable_OutputMemoryKinds);
  _STUB(PJRT_Executable_OptimizedProgram);
  _STUB(PJRT_Executable_Serialize);
  _STUB(PJRT_LoadedExecutable_Destroy);
  _STUB(PJRT_LoadedExecutable_GetExecutable);
  _STUB(PJRT_LoadedExecutable_AddressableDevices);
  _STUB(PJRT_LoadedExecutable_Delete);
  _STUB(PJRT_LoadedExecutable_IsDeleted);
  _STUB(PJRT_LoadedExecutable_Execute);
  _STUB(PJRT_Executable_DeserializeAndLoad);
  _STUB(PJRT_LoadedExecutable_Fingerprint);
  _STUB(PJRT_Buffer_Device);
  _STUB(PJRT_Buffer_Memory);
  _STUB(PJRT_Buffer_IsDeleted);
  _STUB(PJRT_Buffer_CopyToDevice);
  _STUB(PJRT_Buffer_IsOnCpu);
  _STUB(PJRT_Buffer_UnsafePointer);
  _STUB(PJRT_Buffer_IncreaseExternalReferenceCount);
  _STUB(PJRT_Buffer_DecreaseExternalReferenceCount);
  _STUB(PJRT_Buffer_OpaqueDeviceMemoryDataPointer);
  _STUB(PJRT_CopyToDeviceStream_Destroy);
  _STUB(PJRT_CopyToDeviceStream_AddChunk);
  _STUB(PJRT_CopyToDeviceStream_TotalBytes);
  _STUB(PJRT_CopyToDeviceStream_GranuleSize);
  _STUB(PJRT_CopyToDeviceStream_CurrentBytes);
  _STUB(PJRT_TopologyDescription_Create);
  _STUB(PJRT_TopologyDescription_Destroy);
  _STUB(PJRT_TopologyDescription_PlatformName);
  _STUB(PJRT_TopologyDescription_PlatformVersion);
  _STUB(PJRT_TopologyDescription_GetDeviceDescriptions);
  _STUB(PJRT_TopologyDescription_Serialize);
  _STUB(PJRT_TopologyDescription_Attributes);
  _STUB(PJRT_Compile);
  _STUB(PJRT_Executable_OutputElementTypes);
  _STUB(PJRT_Executable_OutputDimensions);
  _STUB(PJRT_Buffer_CopyToMemory);
  _STUB(PJRT_Client_CreateViewOfDeviceBuffer);
  _STUB(PJRT_Client_CreateBuffersForAsyncHostToDevice);
  _STUB(PJRT_Executable_Fingerprint);
  _STUB(PJRT_Client_TopologyDescription);
  _STUB(PJRT_Executable_GetCompiledMemoryStats);
  _STUB(PJRT_Memory_Kind_Id);
  _STUB(PJRT_ExecuteContext_Create);
  _STUB(PJRT_ExecuteContext_Destroy);
  _STUB(PJRT_AsyncHostToDeviceTransferManager_Destroy);
  _STUB(PJRT_AsyncHostToDeviceTransferManager_TransferData);
