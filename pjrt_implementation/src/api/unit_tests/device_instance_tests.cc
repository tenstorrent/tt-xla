// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// PJRT test headers
#include "unit_test_utils.h"

namespace tt::pjrt::tests {

// Specialization of PJRTComponentUnitTestsBase ensures that DeviceInstance
// unit tests will be scoped to their own test suite.
class DeviceInstanceUnitTests : public PJRTComponentUnitTests {};

// Tests successful creation of device instances with valid parameters.
// Verifies that all provided values are correctly persisted.
// As part of SetUp(), device_ has already been created.
TEST_F(DeviceInstanceUnitTests, createInstance_successCase) {
  ASSERT_NE(device_, nullptr);
  EXPECT_EQ(device_->getGlobalDeviceId(), TT_DEVICE_HOST);
  EXPECT_TRUE(device_->isAddressable());
  EXPECT_EQ(device_->getLocalDeviceId(), LOCAL_HARDWARE_ID_UNDEFINED);
  EXPECT_EQ(device_->getDefaultMemory(), nullptr);
  EXPECT_TRUE(device_->getAddressableMemories().empty());
}

// Tests casting DeviceInstance to raw PJRT_Device pointer.
TEST_F(DeviceInstanceUnitTests, castToPJRTDevice) {
  PJRT_Device *pjrt_dvc = *device_;
  EXPECT_NE(pjrt_dvc, nullptr);
  EXPECT_EQ(static_cast<void *>(device_.get()), static_cast<void *>(pjrt_dvc));
}

// Tests "unwrapping" raw PJRT_Device pointer back to DeviceInstance.
// Verifies the unwrapped instance matches the original.
TEST_F(DeviceInstanceUnitTests, unwrapPJRTDevice) {
  PJRT_Device *pjrt_dvc = *device_;
  const DeviceInstance *unwrapped = DeviceInstance::unwrap(pjrt_dvc);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, device_.get());
}

// Tests the assignment of default memory.
// TODO(acicovic): Should the test also validate getAddressableMemories()?
TEST_F(DeviceInstanceUnitTests, setDefaultMemory) {
  device_->setDefaultMemory(default_memory_.get());
  EXPECT_EQ(device_->getDefaultMemory(), default_memory_.get());
}

// Tests adding a memory to the addressable memories list.
// DEVNOTE: Uses default_memory_ for this because it is convenient.
TEST_F(DeviceInstanceUnitTests, addAddressableMemory) {
  device_->addAddressableMemory(default_memory_.get());
  const std::vector<MemoryInstance *> &v = device_->getAddressableMemories();
  ASSERT_EQ(v.size(), 1);
  EXPECT_EQ(v[0], default_memory_.get());
}

// Tests PJRT API for getting device description.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_GetDescription) {
  PJRT_Device_GetDescription_Args args;
  args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  args.device = *device_;
  args.device_description = nullptr;

  PJRT_Error *error = internal::onDeviceGetDescription(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.device_description, nullptr);
  DeviceDescription *desc = DeviceDescription::unwrap(args.device_description);
  EXPECT_EQ(desc->getDeviceId(), TT_DEVICE_HOST);
}

// Tests PJRT API for checking if device is addressable.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_IsAddressable) {
  PJRT_Device_IsAddressable_Args args;
  args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
  args.device = *device_;
  args.is_addressable = false; // intentionally different

  PJRT_Error *error = internal::onDeviceIsAddressable(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_addressable);
}

// Tests PJRT API for getting local hardware ID.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_LocalHardwareId) {
  PJRT_Device_LocalHardwareId_Args args;
  args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
  args.device = *device_;
  args.local_hardware_id = LOCAL_HARDWARE_ID_UNDEFINED + 1; // intentional

  PJRT_Error *error = internal::onDeviceLocalHardwareId(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.local_hardware_id, LOCAL_HARDWARE_ID_UNDEFINED);
}

// Tests PJRT API for getting addressable memories.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_AddressableMemories) {
  device_->addAddressableMemory(
      MemoryInstance::createInstance(device_as_vector_,
                                     /*id=*/10,
                                     /*is_host_memory=*/true)
          .get());

  device_->addAddressableMemory(
      MemoryInstance::createInstance(device_as_vector_,
                                     /*id=*/11,
                                     /*is_host_memory=*/true)
          .get());

  PJRT_Device_AddressableMemories_Args args;
  args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
  args.device = *device_;
  args.memories = nullptr;
  args.num_memories = 0;

  PJRT_Error *error = internal::onDeviceAddressableMemories(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_memories, 2);
  EXPECT_NE(args.memories, nullptr);
}

// Tests PJRT API for getting default memory.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_DefaultMemory) {
  device_->setDefaultMemory(default_memory_.get());

  PJRT_Device_DefaultMemory_Args args;
  args.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
  args.device = *device_;
  args.memory = nullptr;

  PJRT_Error *error = internal::onDeviceDefaultMemory(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.memory, nullptr);
  EXPECT_EQ(MemoryInstance::unwrap(args.memory), default_memory_.get());
}

} // namespace tt::pjrt::tests
