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
#include "api/plugin_attributes.h"

namespace tt::pjrt::tests {

// Tests that getAttributes returns a non-empty vector.
TEST(PluginAttributesUnitTests, getAttributes_returnsNonEmpty) {
  const std::vector<PJRT_NamedValue> &attrs = PluginAttributes::getAttributes();
  EXPECT_FALSE(attrs.empty());
}

// Tests that getAttributes returns expected StableHLO version attributes.
TEST(PluginAttributesUnitTests, getAttributes_containsExpectedAttributes) {
  for (const PJRT_NamedValue &attr : PluginAttributes::getAttributes()) {
    EXPECT_EQ(attr.struct_size, PJRT_NamedValue_STRUCT_SIZE);
    EXPECT_NE(attr.name, nullptr);
    EXPECT_GT(attr.name_size, 0);
    EXPECT_EQ(attr.type, PJRT_NamedValue_kInt64List);
    // Version is a 3-element array (major, minor, patch).
    EXPECT_EQ(attr.value_size, 3);
  }
}

// Tests that getAttributes returns stablehlo_current_version attribute.
TEST(PluginAttributesUnitTests, getAttributes_containsCurrentVersion) {
  bool found = false;
  for (const PJRT_NamedValue &attr : PluginAttributes::getAttributes()) {
    std::string name(attr.name, attr.name_size);
    if (name == "stablehlo_current_version") {
      found = true;
      EXPECT_NE(attr.int64_array_value, nullptr);
      EXPECT_GE(attr.int64_array_value[0], 0);
      EXPECT_GE(attr.int64_array_value[1], 0);
      EXPECT_GE(attr.int64_array_value[2], 0);
      break;
    }
  }
  EXPECT_TRUE(found);
}

// Tests that getAttributes returns stablehlo_minimum_version attribute.
TEST(PluginAttributesUnitTests, getAttributes_containsMinimumVersion) {
  bool found = false;
  for (const PJRT_NamedValue &attr : PluginAttributes::getAttributes()) {
    std::string name(attr.name, attr.name_size);
    if (name == "stablehlo_minimum_version") {
      found = true;
      EXPECT_NE(attr.int64_array_value, nullptr);
      EXPECT_GE(attr.int64_array_value[0], 0);
      EXPECT_GE(attr.int64_array_value[1], 0);
      EXPECT_GE(attr.int64_array_value[2], 0);
      break;
    }
  }
  EXPECT_TRUE(found);
}

// Tests that calling getAttributes multiple times returns the same reference.
TEST(PluginAttributesUnitTests, getAttributes_returnsSameReference) {
  const std::vector<PJRT_NamedValue> &attrs1 =
      PluginAttributes::getAttributes();
  const std::vector<PJRT_NamedValue> &attrs2 =
      PluginAttributes::getAttributes();
  EXPECT_EQ(&attrs1, &attrs2);
}

// Tests StableHLOVersionAttribute creation and conversion to named value.
TEST(PluginAttributesUnitTests, StableHLOVersionAttribute_toNamedValue) {
  mlir::vhlo::Version test_ver = mlir::vhlo::Version::getCurrentVersion();
  StableHLOVersionAttribute attr("test_version", test_ver);
  PJRT_NamedValue named_val = attr.toNamedValue();

  EXPECT_EQ(named_val.struct_size, PJRT_NamedValue_STRUCT_SIZE);
  EXPECT_EQ(named_val.extension_start, nullptr);
  EXPECT_EQ(std::string(named_val.name, named_val.name_size), "test_version");
  EXPECT_EQ(named_val.type, PJRT_NamedValue_kInt64List);
  EXPECT_EQ(named_val.value_size, 3);
  EXPECT_EQ(named_val.int64_array_value[0], test_ver.getMajor());
  EXPECT_EQ(named_val.int64_array_value[1], test_ver.getMinor());
  EXPECT_EQ(named_val.int64_array_value[2], test_ver.getPatch());
}

// Tests PJRT API for plugin initialization.
TEST(PluginAttributesUnitTests, API_PJRT_Plugin_Initialize) {
  PJRT_Plugin_Initialize_Args args;
  args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;

  PJRT_Error *error = internal::onPluginInitialize(&args);
  EXPECT_EQ(error, nullptr);
}

// Tests PJRT API for getting plugin attributes.
TEST(PluginAttributesUnitTests, API_PJRT_Plugin_Attributes) {
  PJRT_Plugin_Attributes_Args args;
  args.struct_size = PJRT_Plugin_Attributes_Args_STRUCT_SIZE;
  args.attributes = nullptr;
  args.num_attributes = 0;

  PJRT_Error *error = internal::onPluginAttributes(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_GE(args.num_attributes, 2);
  ASSERT_NE(args.attributes, nullptr);

  // Verify the attributes match what PluginAttributes::getAttributes returns.
  const std::vector<PJRT_NamedValue> &expected_attrs =
      PluginAttributes::getAttributes();
  EXPECT_EQ(args.attributes, expected_attrs.data());
  EXPECT_EQ(args.num_attributes, expected_attrs.size());
}

} // namespace tt::pjrt::tests
