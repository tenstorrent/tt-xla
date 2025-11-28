// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "api/device_description.h"

// tt-mlir headers
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::pjrt::tests {

// Common literals.
static constexpr int TT_DEVICE_HOST = 0;
static constexpr tt::target::Arch TT_ARCH_WH = tt::target::Arch::Wormhole_b0;

// Tests successful creation of device description instances.
TEST(DeviceDescriptionUnitTests, createInstance_successCase) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);
  EXPECT_EQ(description.getDeviceId(), TT_DEVICE_HOST);
  EXPECT_FALSE(description.getDeviceKind().empty());
  EXPECT_FALSE(description.toString().empty());
  EXPECT_FALSE(description.toDebugString().empty());
}

// Tests casting DeviceDescription to raw PJRT_DeviceDescription pointer.
TEST(DeviceDescriptionUnitTests, castToPJRTDeviceDescription) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);
  PJRT_DeviceDescription *pjrt_desc = description;
  EXPECT_NE(pjrt_desc, nullptr);
  EXPECT_EQ(static_cast<void *>(&description), static_cast<void *>(pjrt_desc));
}

// Tests "unwrapping" raw PJRT_DeviceDescription pointer back to
// DeviceDescription. Verifies the unwrapped instance matches the original.
TEST(DeviceDescriptionUnitTests, unwrapPJRTDeviceDescription) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);
  PJRT_DeviceDescription *pjrt_desc = description;
  DeviceDescription *unwrapped = DeviceDescription::unwrap(pjrt_desc);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, &description);
  EXPECT_EQ(unwrapped->getDeviceId(), description.getDeviceId());
  EXPECT_EQ(unwrapped->getDeviceKind(), description.getDeviceKind());
}

// Tests getting and setting the process index.
TEST(DeviceDescriptionUnitTests, processIndex_defaultGetAndSet) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);
  EXPECT_EQ(description.getProcessIndex(), 0); // default
  description.setProcessIndex(3);
  EXPECT_EQ(description.getProcessIndex(), 3);
}

// Tests that debug string contains device and architecture information.
TEST(DeviceDescriptionUnitTests, toDebugString_containsDeviceInfo) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);
  const std::string &debug_string = description.toDebugString();
  EXPECT_FALSE(debug_string.empty());
  EXPECT_NE(debug_string.find(std::to_string(TT_DEVICE_HOST)),
            std::string::npos);
  EXPECT_NE(debug_string.find(tt::target::EnumNameArch(TT_ARCH_WH)),
            std::string::npos);
}

// Tests PJRT API for getting the device ID.
TEST(DeviceDescriptionUnitTests, API_PJRT_DeviceDescription_Id) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);

  PJRT_DeviceDescription_Id_Args args;
  args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
  args.device_description = description; // implicit pointer conversion
  args.id = -1; // intentionally set to a different value

  PJRT_Error *result = internal::onDeviceDescriptionId(&args);
  ASSERT_EQ(result, nullptr);
  EXPECT_EQ(args.id, TT_DEVICE_HOST);
}

// Tests PJRT API for getting the process index.
TEST(DeviceDescriptionUnitTests, API_PJRT_DeviceDescription_ProcessIndex) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);
  description.setProcessIndex(7);

  PJRT_DeviceDescription_ProcessIndex_Args args;
  args.struct_size = PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE;
  args.device_description = description; // implicit pointer conversion
  args.process_index = -1;               // intentionally set to different value

  PJRT_Error *result = internal::onDeviceDescriptionProcessIndex(&args);
  ASSERT_EQ(result, nullptr);
  EXPECT_EQ(args.process_index, 7);
}

// Tests PJRT API for getting device attributes.
// Currently no attributes are set, so this should return 0 attributes.
TEST(DeviceDescriptionUnitTests, API_PJRT_DeviceDescription_Attributes) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);

  PJRT_DeviceDescription_Attributes_Args args;
  args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
  args.device_description = description; // implicit pointer conversion

  PJRT_Error *result = internal::onDeviceDescriptionAttributes(&args);
  ASSERT_EQ(result, nullptr);
  EXPECT_EQ(args.num_attributes, 0);
  EXPECT_EQ(args.attributes, nullptr);
}

// Tests PJRT API for getting the device kind.
TEST(DeviceDescriptionUnitTests, API_PJRT_DeviceDescription_Kind) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);

  PJRT_DeviceDescription_Kind_Args args;
  args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
  args.device_description = description; // implicit pointer conversion
  args.device_kind = nullptr;
  args.device_kind_size = 0;

  PJRT_Error *result = internal::onDeviceDescriptionKind(&args);
  ASSERT_EQ(result, nullptr);
  ASSERT_NE(args.device_kind, nullptr);
  EXPECT_GT(args.device_kind_size, 0);
  EXPECT_EQ(std::string(args.device_kind, args.device_kind_size),
            description.getDeviceKind());
}

// Tests PJRT API for getting the debug string.
TEST(DeviceDescriptionUnitTests, API_PJRT_DeviceDescription_DebugString) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);

  PJRT_DeviceDescription_DebugString_Args args;
  args.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
  args.device_description = description; // implicit pointer conversion
  args.debug_string = nullptr;
  args.debug_string_size = 0;

  PJRT_Error *result = internal::onDeviceDescriptionDebugString(&args);
  ASSERT_EQ(result, nullptr);
  ASSERT_NE(args.debug_string, nullptr);
  EXPECT_GT(args.debug_string_size, 0);
  EXPECT_EQ(std::string(args.debug_string, args.debug_string_size),
            description.toDebugString());
}

// Tests PJRT API for getting the toString output.
TEST(DeviceDescriptionUnitTests, API_PJRT_DeviceDescription_ToString) {
  DeviceDescription description(TT_DEVICE_HOST, TT_ARCH_WH);

  PJRT_DeviceDescription_ToString_Args args;
  args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
  args.device_description = description; // implicit pointer conversion
  args.to_string = nullptr;
  args.to_string_size = 0;

  PJRT_Error *result = internal::onDeviceDescriptionToString(&args);
  ASSERT_EQ(result, nullptr);
  ASSERT_NE(args.to_string, nullptr);
  EXPECT_GT(args.to_string_size, 0);
  EXPECT_EQ(std::string(args.to_string, args.to_string_size),
            description.toString());
}

} // namespace tt::pjrt::tests
