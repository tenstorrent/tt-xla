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

// Tests that getAttributes returns non-empty vector.
TEST(PluginAttributesUnitTests, getAttributes_returnsNonEmpty) {
  const std::vector<PJRT_NamedValue> &attrs = PluginAttributes::getAttributes();
  EXPECT_FALSE(attrs.empty());
}

// Tests that getAttributes returns expected StableHLO version attributes.
TEST(PluginAttributesUnitTests, getAttributes_containsExpectedAttributes) {
  const std::vector<PJRT_NamedValue> &attrs = PluginAttributes::getAttributes();

  // Should contain at least current and minimum StableHLO versions.
  ASSERT_GE(attrs.size(), 2);

  // Check that we have named values with correct structure.
  for (const PJRT_NamedValue &attr : attrs) {
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
  const std::vector<PJRT_NamedValue> &attrs = PluginAttributes::getAttributes();

  bool found_current = false;
  for (const PJRT_NamedValue &attr : attrs) {
    std::string name(attr.name, attr.name_size);
    if (name == "stablehlo_current_version") {
      found_current = true;
      // Verify it's a valid version triplet.
      EXPECT_NE(attr.int64_array_value, nullptr);
      // All version components should be non-negative.
      EXPECT_GE(attr.int64_array_value[0], 0);
      EXPECT_GE(attr.int64_array_value[1], 0);
      EXPECT_GE(attr.int64_array_value[2], 0);
      break;
    }
  }
  EXPECT_TRUE(found_current);
}

// Tests that getAttributes returns stablehlo_minimum_version attribute.
TEST(PluginAttributesUnitTests, getAttributes_containsMinimumVersion) {
  const std::vector<PJRT_NamedValue> &attrs = PluginAttributes::getAttributes();

  bool found_minimum = false;
  for (const PJRT_NamedValue &attr : attrs) {
    std::string name(attr.name, attr.name_size);
    if (name == "stablehlo_minimum_version") {
      found_minimum = true;
      // Verify it's a valid version triplet.
      EXPECT_NE(attr.int64_array_value, nullptr);
      // All version components should be non-negative.
      EXPECT_GE(attr.int64_array_value[0], 0);
      EXPECT_GE(attr.int64_array_value[1], 0);
      EXPECT_GE(attr.int64_array_value[2], 0);
      break;
    }
  }
  EXPECT_TRUE(found_minimum);
}

// Tests that calling getAttributes multiple times returns the same vector.
TEST(PluginAttributesUnitTests, getAttributes_returnsSameReference) {
  const std::vector<PJRT_NamedValue> &attrs1 =
      PluginAttributes::getAttributes();
  const std::vector<PJRT_NamedValue> &attrs2 =
      PluginAttributes::getAttributes();

  // Should return the same static vector.
  EXPECT_EQ(&attrs1, &attrs2);
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
  EXPECT_NE(args.attributes, nullptr);

  // Verify the attributes match what getAttributes returns.
  const std::vector<PJRT_NamedValue> &expected_attrs =
      PluginAttributes::getAttributes();
  EXPECT_EQ(args.attributes, expected_attrs.data());
  EXPECT_EQ(args.num_attributes, expected_attrs.size());
}

// Tests StableHLOVersionAttribute creation and conversion to named value.
TEST(PluginAttributesUnitTests, StableHLOVersionAttribute_toNamedValue) {
  mlir::vhlo::Version test_version = mlir::vhlo::Version::getCurrentVersion();

  StableHLOVersionAttribute attr("test_version", test_version);
  PJRT_NamedValue named_value = attr.toNamedValue();

  EXPECT_EQ(named_value.struct_size, PJRT_NamedValue_STRUCT_SIZE);
  EXPECT_EQ(named_value.extension_start, nullptr);
  EXPECT_EQ(std::string(named_value.name, named_value.name_size),
            "test_version");
  EXPECT_EQ(named_value.type, PJRT_NamedValue_kInt64List);
  EXPECT_EQ(named_value.value_size, 3);

  // Verify version components.
  EXPECT_EQ(named_value.int64_array_value[0], test_version.getMajor());
  EXPECT_EQ(named_value.int64_array_value[1], test_version.getMinor());
  EXPECT_EQ(named_value.int64_array_value[2], test_version.getPatch());
}

} // namespace tt::pjrt::tests
