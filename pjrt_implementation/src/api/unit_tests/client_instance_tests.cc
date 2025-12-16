// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// C++ standard library headers
#include <string>
#include <vector>

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "api/client_instance.h"

namespace tt::pjrt::tests {

// Note: Full ClientInstance testing requires hardware access.
// These tests focus on the ClientInstance class methods that don't require
// full initialization or hardware access.

// Test fixture for ClientInstance unit tests.
// Note: We cannot create a fully initialized ClientInstance without hardware,
// so these tests are limited to testing the class interface and static
// methods.
class ClientInstanceUnitTests : public ::testing::Test {};

// Tests that unwrap returns nullptr for nullptr input.
TEST_F(ClientInstanceUnitTests, unwrap_nullptr) {
  ClientInstance *client = ClientInstance::unwrap(nullptr);
  EXPECT_EQ(client, nullptr);
}

// Note: The following tests would require a fully initialized ClientInstance,
// which requires hardware access. These are documented as examples of what
// would be tested in an integration test environment.

// Example of what would be tested with hardware:
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_PlatformName) {
//   // Would test that platform name is "tt"
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_PlatformVersion) {
//   // Would test that platform version is returned
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_ProcessIndex) {
//   // Would test that process index is 0 for single-process
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_Devices) {
//   // Would test that devices are returned
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_AddressableDevices) {
//   // Would test that addressable devices are returned
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_AddressableMemories) {
//   // Would test that addressable memories are returned
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_LookupDevice) {
//   // Would test device lookup by ID
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_LookupAddressableDevice) {
//   // Would test addressable device lookup by local hardware ID
// }
//
// TEST_F(ClientInstanceUnitTests, API_PJRT_Client_DefaultDeviceAssignment) {
//   // Would test default device assignment
// }

} // namespace tt::pjrt::tests
