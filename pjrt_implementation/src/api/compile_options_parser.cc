// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_options_parser.h"

// third-party includes
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

// tt-xla includes
#include "utils/logging.h"

namespace tt::pjrt {

tt_pjrt_status CompileOptionsParser::parseCompileOptions(
    const char *compile_options_data, size_t compile_options_size,
    std::unordered_map<std::string, std::string> &out_compile_options,
    std::optional<std::vector<int64_t>> &out_replica_device_ids) {

  google::protobuf::UnknownFieldSet unknown_fields;

  if (!parseCompileOptionsProto(compile_options_data, compile_options_size,
                                unknown_fields)) {
    return tt_pjrt_status::kInternal;
  }

  // Extract custom compile options
  tt_pjrt_status custom_options_status =
      extractCustomProtobufFields(unknown_fields, out_compile_options);
  if (!tt_pjrt_status_is_ok(custom_options_status)) {
    return custom_options_status;
  }

  // Extract replica device IDs
  tt_pjrt_status replica_ids_status =
      extractReplicaDeviceIds(unknown_fields, out_replica_device_ids);
  if (!tt_pjrt_status_is_ok(replica_ids_status)) {
    return replica_ids_status;
  }

  return tt_pjrt_status::kSuccess;
}

bool CompileOptionsParser::parseCompileOptionsProto(
    const char *compile_options_data, size_t compile_options_size,
    google::protobuf::UnknownFieldSet &unknown_fields) {
  google::protobuf::io::CodedInputStream cis(
      reinterpret_cast<const uint8_t *>(compile_options_data),
      compile_options_size);

  if (!unknown_fields.MergeFromCodedStream(&cis)) {
    DLOG_F(ERROR, "Failed to parse compile options protobuf data");
    return false;
  }

  return true;
}

bool CompileOptionsParser::parseNestedProtobufField(
    const google::protobuf::UnknownField &field,
    google::protobuf::UnknownFieldSet &out_fields) {
  if (field.type() != google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
    return false;
  }

  const std::string &data = field.length_delimited();
  google::protobuf::io::CodedInputStream input(
      reinterpret_cast<const uint8_t *>(data.data()), data.size());

  return out_fields.ParseFromCodedStream(&input);
}

void CompileOptionsParser::extractDeviceIdsFromComputationDevice(
    const google::protobuf::UnknownFieldSet &comp_device_fields,
    std::set<int64_t> &out_device_ids) {
  constexpr int kDeviceIdFieldNumber = 1;

  for (int i = 0; i < comp_device_fields.field_count(); ++i) {
    const google::protobuf::UnknownField &field = comp_device_fields.field(i);

    if (field.number() != kDeviceIdFieldNumber) {
      continue;
    }

    if (field.type() == google::protobuf::UnknownField::TYPE_VARINT) {
      // Unpacked repeated field
      out_device_ids.insert(field.varint());
    } else if (field.type() ==
               google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
      // Packed repeated field
      const std::string &packed_data = field.length_delimited();
      google::protobuf::io::CodedInputStream packed_stream(
          reinterpret_cast<const uint8_t *>(packed_data.data()),
          packed_data.size());

      uint64_t value;
      while (packed_stream.ReadVarint64(&value)) {
        out_device_ids.insert(static_cast<int64_t>(value));
      }
    }
  }
}

void CompileOptionsParser::extractDeviceIdsFromDeviceAssignment(
    const google::protobuf::UnknownFieldSet &device_assign_fields,
    std::set<int64_t> &out_device_ids) {
  constexpr int kComputationDeviceFieldNumber = 3;

  for (int i = 0; i < device_assign_fields.field_count(); ++i) {
    const google::protobuf::UnknownField &field = device_assign_fields.field(i);

    if (field.number() != kComputationDeviceFieldNumber) {
      continue;
    }

    google::protobuf::UnknownFieldSet comp_device_fields;
    if (parseNestedProtobufField(field, comp_device_fields)) {
      extractDeviceIdsFromComputationDevice(comp_device_fields, out_device_ids);
    }
  }
}

tt_pjrt_status CompileOptionsParser::extractReplicaDeviceIds(
    const google::protobuf::UnknownFieldSet &unknown_fields,
    std::optional<std::vector<int64_t>> &out_replica_device_ids) {
  std::set<int64_t> unique_device_ids;

  // The CompileOptionsProto protobuf layout is defined in
  // https://github.com/openxla/xla/blob/main/xla/pjrt/proto/compile_options.proto

  // The executable build compiler options that are defined in through the
  // jax.jit() and contain the information about devices assignment are stored
  // in the field number 3.
  constexpr int kExecutableBuildOptionsProtoFieldNumber = 3;

  // DeviceAssignmentProto is field number 9 in ExecutableBuildOptionsProto
  constexpr int kDeviceAssignmentProtoFieldNumber = 9;

  for (int i = 0; i < unknown_fields.field_count(); ++i) {
    const google::protobuf::UnknownField &field = unknown_fields.field(i);

    if (field.number() != kExecutableBuildOptionsProtoFieldNumber) {
      continue;
    }

    google::protobuf::UnknownFieldSet exec_build_fields;
    if (!parseNestedProtobufField(field, exec_build_fields)) {
      continue;
    }

    for (int j = 0; j < exec_build_fields.field_count(); ++j) {
      const google::protobuf::UnknownField &exec_field =
          exec_build_fields.field(j);

      if (exec_field.number() != kDeviceAssignmentProtoFieldNumber) {
        continue;
      }

      google::protobuf::UnknownFieldSet device_assign_fields;
      if (parseNestedProtobufField(exec_field, device_assign_fields)) {
        extractDeviceIdsFromDeviceAssignment(device_assign_fields,
                                             unique_device_ids);
      }
    }
  }

  if (!unique_device_ids.empty()) {
    out_replica_device_ids = std::vector<int64_t>(unique_device_ids.begin(),
                                                  unique_device_ids.end());
  }
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status CompileOptionsParser::extractCustomProtobufFields(
    const google::protobuf::UnknownFieldSet &unknown_fields,
    std::unordered_map<std::string, std::string> &out_compile_options) {

  // The custom compiler options that are passed in through the `jax.jit()`
  // or `torch_xla.set_custom_compile_options()` are stored in the field
  // number 7 in the UnknownFieldSet, which is defined as:
  // `env_option_overrides (map<string, OptionOverrideProto>)`.
  // Each map entry is a nested message with key/value inside.
  constexpr int kCustomCompilerOptionsFieldNumber = 7;

  // Field number corresponding to the string key of the compile options map
  // entry.
  constexpr int kMapKeyFieldNumber = 1;

  // Field number corresponding to the OptionOverrideProto value of the compile
  // options map entry.
  constexpr int kMapValueFieldNumber = 2;

  for (int i = 0; i < unknown_fields.field_count(); ++i) {
    const google::protobuf::UnknownField &field = unknown_fields.field(i);

    // Currently, we only support custom compiler options serialized in the
    // `kCustomCompilerOptionsFieldNumber` field. In case we encounter
    // options being serialized into some other field we will need to update
    // this to support them.
    if (field.number() != kCustomCompilerOptionsFieldNumber ||
        field.type() != google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
      continue;
    }

    const std::string &bytes = field.length_delimited();
    google::protobuf::io::CodedInputStream cis(
        reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());

    google::protobuf::UnknownFieldSet map_entry_fields;
    if (!map_entry_fields.MergeFromCodedStream(&cis)) {
      DLOG_F(ERROR, "Failed to parse the map entry fields from the custom "
                    "compile options protobuf data");
      return tt_pjrt_status::kInternal;
    }

    std::string key;
    std::string value;

    for (int j = 0; j < map_entry_fields.field_count(); ++j) {
      const google::protobuf::UnknownField &entry_field =
          map_entry_fields.field(j);
      // In the inner field set, first field is the key and second field is the
      // value. We expect both to be length-delimited fields (coming from a
      // dictionary).
      if (entry_field.number() == kMapKeyFieldNumber &&
          entry_field.type() ==
              google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
        key = entry_field.length_delimited();
      } else if (entry_field.number() == kMapValueFieldNumber &&
                 entry_field.type() ==
                     google::protobuf::UnknownField::TYPE_LENGTH_DELIMITED) {
        const std::string &override_bytes = entry_field.length_delimited();
        google::protobuf::io::CodedInputStream override_stream(
            reinterpret_cast<const uint8_t *>(override_bytes.data()),
            override_bytes.size());

        google::protobuf::UnknownFieldSet value_fields;
        if (!value_fields.MergeFromCodedStream(&override_stream)) {
          DLOG_F(ERROR, "Failed to parse the map entry field value from the "
                        "custom compile options protobuf data");
          return tt_pjrt_status::kInternal;
        }

        // https://github.com/openxla/xla/blob/main/xla/pjrt/proto/compile_options.proto#L151C1-L158C2
        // Field numbers and types for OptionOverrideProto
        // message OptionOverrideProto {
        //   oneof value {
        //     string string_field = 1;
        //     bool bool_field = 2;
        //     int64 int_field = 3;
        //     double double_field = 4;
        //   }
        // }
        if (value_fields.field_count() != 1) {
          DLOG_F(
              ERROR,
              "Expected exactly one field in OptionOverrideProto, but got %d",
              value_fields.field_count());
          return tt_pjrt_status::kInternal;
        }

        const google::protobuf::UnknownField &value_field =
            value_fields.field(0);
        switch (value_field.number()) {
        case 1: {
          value = value_field.length_delimited();
          break;
        }
        case 2: {
          value = value_field.varint() ? "true" : "false";
          break;
        }
        case 3: {
          value = std::to_string(value_field.varint());
          break;
        }
        case 4: {
          value = std::to_string(value_field.fixed64());
          break;
        }
        default: {
          DLOG_F(ERROR, "Unknown field number in OptionOverrideProto: %d",
                 value_field.number());
          return tt_pjrt_status::kInternal;
        }
        }
      }
    }

    if (!key.empty()) {
      out_compile_options[key] = value;
    }
  }

  return tt_pjrt_status::kSuccess;
}

} // namespace tt::pjrt
