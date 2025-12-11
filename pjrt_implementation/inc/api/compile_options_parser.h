// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_COMPILE_OPTIONS_PARSER_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_COMPILE_OPTIONS_PARSER_H_

// c++ standard library includes
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// third-party includes
#include <google/protobuf/unknown_field_set.h>

// tt-xla includes
#include "utils/status.h"

namespace tt::pjrt {

// Utility class for parsing PJRT compile options protobuf data.
// This class provides methods to extract custom compile options and
// device assignment information from the protobuf data passed through
// the PJRT API.
//
// The protobuf layout is defined in:
// https://github.com/openxla/xla/blob/main/xla/pjrt/proto/compile_options.proto
class CompileOptionsParser {
public:
  // Parses compile options protobuf and extracts both custom compile options
  // and replica device IDs.
  static tt_pjrt_status
  parseCompileOptions(const char *compile_options_data,
                      size_t compile_options_size,
                      std::unordered_map<std::string, std::string> &out_compile_options,
                      std::optional<std::vector<int64_t>> &out_replica_device_ids);

  // Parses compile options protobuf data into UnknownFieldSet.
  // Returns true if parsing succeeded, false otherwise.
  static bool
  parseCompileOptionsProto(const char *compile_options_data,
                           size_t compile_options_size,
                           google::protobuf::UnknownFieldSet &unknown_fields);

  // Extracts custom protobuf fields from an UnknownFieldSet of all protobuf
  // fields.
  static tt_pjrt_status extractCustomProtobufFields(
      const google::protobuf::UnknownFieldSet &unknown_fields,
      std::unordered_map<std::string, std::string> &out_compile_options);

  // Extracts replica device IDs from an already-parsed UnknownFieldSet.
  static tt_pjrt_status extractReplicaDeviceIds(
      const google::protobuf::UnknownFieldSet &unknown_fields,
      std::optional<std::vector<int64_t>> &out_replica_device_ids);

  // Helper function to parse a length-delimited protobuf field into an
  // UnknownFieldSet. Returns true if parsing succeeds, false otherwise.
  static bool
  parseNestedProtobufField(const google::protobuf::UnknownField &field,
                           google::protobuf::UnknownFieldSet &out_fields);

  // Helper function to extract device IDs from a ComputationDevice protobuf
  // field. Device IDs are stored in field number 1 as a repeated field (either
  // packed or unpacked).
  static void extractDeviceIdsFromComputationDevice(
      const google::protobuf::UnknownFieldSet &comp_device_fields,
      std::set<int64_t> &out_device_ids);

  // Helper function to extract device IDs from a DeviceAssignmentProto.
  // ComputationDevice is field number 3 in DeviceAssignmentProto.
  static void extractDeviceIdsFromDeviceAssignment(
      const google::protobuf::UnknownFieldSet &device_assign_fields,
      std::set<int64_t> &out_device_ids);
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_COMPILE_OPTIONS_PARSER_H_
