// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt


  // TODO(mrakita): Catch-all todo to implement missing functionality listed in
  // these stubs.


  // ---------------------------------------------------------------------------
  // PJRT_Device stubs:

  // PJRT_Memory is not yet supported in our implementation.
  // https://github.com/tenstorrent/tt-xla/issues/468
  _STUB(PJRT_Device_AddressableMemories);
  _STUB(PJRT_Device_DefaultMemory);

  // This is used for diagnostic purposes and is allowed to return unimplemented
  // status code, but we could implement at least `bytes_in_use` calculation
  // which is the only non-optional stat that has to be set in this function.
  // https://github.com/tenstorrent/tt-xla/issues/470
  _STUB(PJRT_Device_MemoryStats);
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  // PJRT_Buffer stubs:

  // Deprecated in v0.54.
  _STUB(PJRT_Buffer_GetMemoryLayout);

  // Used to copy raw buffer from device to host. In order to support this we
  // need flexible memcpy from runtime in order to specify the offset and size.
  _STUB(PJRT_Buffer_OnDeviceSizeInBytes);
  _STUB(PJRT_Buffer_CopyRawToHost);

  // XLA client is not currently using this, but we should implement it, it
  // shouldn't be too hard. It should just create the output buffer by copying
  // the input buffer (its parameters and underlying tensor) and setting given
  // device.
  // https://github.com/tenstorrent/tt-xla/issues/467
  _STUB(PJRT_Buffer_CopyToDevice);

  // PJRT_Memory is not yet supported in our implementation.
  // https://github.com/tenstorrent/tt-xla/issues/468
  _STUB(PJRT_Buffer_CopyToMemory);
  _STUB(PJRT_Buffer_Memory);

  // Can be used in combination with `PJRT_Buffer_IsOnCpu` to enable certain
  // optimizations (avoids memory copy), but currently all our inputs are
  // transferred to device where computation runs and we can't return the
  // pointer to device memory.
  _STUB(PJRT_Buffer_UnsafePointer);
  _STUB(PJRT_Buffer_IncreaseExternalReferenceCount);
  _STUB(PJRT_Buffer_DecreaseExternalReferenceCount);
  _STUB(PJRT_Buffer_OpaqueDeviceMemoryDataPointer);
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  // PJRT_ExecuteContext stubs:

  // Not currently used by PJRT API. XLA uses it for creation of
  // `PJRT_ExecuteOptions` for `PJRT_LoadedExecutable_Execute` call, to store
  // some additional internal context.
  _STUB(PJRT_ExecuteContext_Create);
  _STUB(PJRT_ExecuteContext_Destroy);
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  // PJRT_LoadedExecutable stubs:

  // This is not required to be implemented by all platforms. Shouldn't be hard
  // to implement though, we can take hash of the optimized program, compile
  // options, compiler version, metal version, etc.
  // https://github.com/tenstorrent/tt-xla/issues/495
  _STUB(PJRT_Executable_Fingerprint);

  // Not required to be implemented and unclear how this is used by the clients
  // (if at all).
  _STUB(PJRT_Executable_GetCostAnalysis);

  // This is not required to be implemented by all platforms. It should return
  // memory stats that allow callers to estimate device memory usage when
  // running the executable.
  // https://github.com/tenstorrent/tt-xla/issues/496
  _STUB(PJRT_Executable_GetCompiledMemoryStats);

  // This should return list of lists of memory kind strings for outputs, in a
  // shape [num_programs, num_outputs]. Most implementations I've found just
  // populate it with default memory kind.
  // https://github.com/tenstorrent/tt-xla/issues/497
  _STUB(PJRT_Executable_OutputMemoryKinds);

  // Serialization/deserialization of our executable is not yet supported.
  // https://github.com/tenstorrent/tt-xla/issues/498
  _STUB(PJRT_Executable_Serialize);
  _STUB(PJRT_Executable_DeserializeAndLoad);
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  // PJRT_LoadedExecutable stubs:

  // Deprecated in v0.35, replaced by `PJRT_Executable_Fingerprint`.
  _STUB(PJRT_LoadedExecutable_Fingerprint);
  // ---------------------------------------------------------------------------


  _STUB(PJRT_Plugin_Initialize);
  _STUB(PJRT_Plugin_Attributes);
  _STUB(PJRT_Client_Create);
  _STUB(PJRT_Client_PlatformName);
  _STUB(PJRT_Client_PlatformVersion);
  _STUB(PJRT_Client_DefaultDeviceAssignment);
  _STUB(PJRT_Memory_Id);
  _STUB(PJRT_Memory_Kind);
  _STUB(PJRT_Memory_DebugString);
  _STUB(PJRT_Memory_ToString);
  _STUB(PJRT_Memory_AddressableByDevices);
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
  _STUB(PJRT_Client_CreateViewOfDeviceBuffer);
  _STUB(PJRT_Client_CreateBuffersForAsyncHostToDevice);
  _STUB(PJRT_Client_TopologyDescription);
  _STUB(PJRT_Memory_Kind_Id);
  _STUB(PJRT_AsyncHostToDeviceTransferManager_Destroy);
  _STUB(PJRT_AsyncHostToDeviceTransferManager_TransferData);
