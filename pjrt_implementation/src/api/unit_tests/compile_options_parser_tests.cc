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
#include <unordered_map>

// GTest headers
#include "gtest/gtest.h"

// Protobuf headers
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/unknown_field_set.h>

// PJRT implementation headers
#include "api/compile_options.h"
#include "api/compile_options_parser.h"

namespace tt::pjrt::tests {

// ExecutableBuildOptionsProto is field #3 in CompileOptionsProto.
constexpr int kExecutableBuildOptionsFieldNumber = 3;

// Optimization level is field #24 inside ExecutableBuildOptionsProto.
constexpr int kOptimizationLevelFieldNumber = 24;

// Protobuf field for varint
static constexpr uint32_t kWireTypeVarint = 0;

// Protobuf field for length
static constexpr uint32_t kWireTypeLengthDelimited = 2;

// Helper to encode a varint field into wire-format bytes.
static std::string encodeVarint(int field_number, uint64_t value) {
  std::string output;
  google::protobuf::io::StringOutputStream sos(&output);
  google::protobuf::io::CodedOutputStream cos(&sos);

  // Shifting by 3 since protobuf has only 6 different types
  cos.WriteTag((field_number << 3) | kWireTypeVarint);
  cos.WriteVarint64(value);

  return output;
}

// Helper to encode a length field wrapping inner bytes.
static std::string encodeLength(int field_number, const std::string &data) {
  std::string output;
  google::protobuf::io::StringOutputStream sos(&output);
  google::protobuf::io::CodedOutputStream cos(&sos);

  cos.WriteTag((field_number << 3) | kWireTypeLengthDelimited);
  cos.WriteVarint32(data.size());
  cos.WriteRaw(data.data(), data.size());

  return output;
}

static std::string
buildCompileOptionsWithOptLevel(uint64_t optimization_level) {
  std::string inner =
      encodeVarint(kOptimizationLevelFieldNumber, optimization_level);
  return encodeLength(kExecutableBuildOptionsFieldNumber, inner);
}

// Helper to parse raw bytes into an UnknownFieldSet.
static bool parseToFieldSet(const std::string &data,
                            google::protobuf::UnknownFieldSet &fields) {
  google::protobuf::io::CodedInputStream cis(
      reinterpret_cast<const uint8_t *>(data.data()), data.size());
  return fields.MergeFromCodedStream(&cis);
}

TEST(CompileOptionsParserUnitTests,
     extractCustomProtobufFields_optimizationLevelPresent) {
  std::string data = buildCompileOptionsWithOptLevel(2);

  google::protobuf::UnknownFieldSet fields;
  ASSERT_TRUE(parseToFieldSet(data, fields));

  std::unordered_map<std::string, std::string> compile_options;
  tt_pjrt_status status = CompileOptionsParser::extractCustomProtobufFields(
      fields, compile_options);
  ASSERT_TRUE(tt_pjrt_status_is_ok(status));
  ASSERT_NE(compile_options.find(CompileOptions::optimization_level_key),
            compile_options.end());
  EXPECT_EQ(compile_options[CompileOptions::optimization_level_key], "2");
}

TEST(CompileOptionsParserUnitTests,
     extractCustomProtobufFields_optimizationLevelAbsent) {
  google::protobuf::UnknownFieldSet fields;

  std::unordered_map<std::string, std::string> compile_options;
  tt_pjrt_status status = CompileOptionsParser::extractCustomProtobufFields(
      fields, compile_options);
  ASSERT_TRUE(tt_pjrt_status_is_ok(status));
  EXPECT_EQ(compile_options.find(CompileOptions::optimization_level_key),
            compile_options.end());
}

// Sanity check that unrelated fields are ignored.
TEST(CompileOptionsParserUnitTests,
     extractCustomProtobufFields_unrelatedFieldsIgnored) {

  std::string unrelated = encodeVarint(5, 42);

  google::protobuf::UnknownFieldSet fields;
  ASSERT_TRUE(parseToFieldSet(unrelated, fields));

  std::unordered_map<std::string, std::string> compile_options;
  tt_pjrt_status status = CompileOptionsParser::extractCustomProtobufFields(
      fields, compile_options);
  ASSERT_TRUE(tt_pjrt_status_is_ok(status));
  EXPECT_TRUE(compile_options.empty());
}

} // namespace tt::pjrt::tests
