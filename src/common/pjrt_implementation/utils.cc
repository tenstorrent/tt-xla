#include "pjrt_implementation/utils.h"

namespace tt::pjrt {

PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType) {
  switch (ElementType) {
  case tt::target::DataType::UInt8:
    return PJRT_Buffer_Type_U8;
  case tt::target::DataType::UInt16:
    return PJRT_Buffer_Type_U16;
  case tt::target::DataType::UInt32:
    return PJRT_Buffer_Type_U32;
  case tt::target::DataType::Float16:
    return PJRT_Buffer_Type_F16;
  case tt::target::DataType::Float32:
    return PJRT_Buffer_Type_F32;
  case tt::target::DataType::BFloat16:
    return PJRT_Buffer_Type_BF16;
  default:
    assert(false && "Unsupported data type");
    return PJRT_Buffer_Type_BF16;
  }
}

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type) {
  switch (buffer_type) {
  case PJRT_Buffer_Type_U8:
    return std::make_pair(tt::target::DataType::UInt8, 1);
  case PJRT_Buffer_Type_U16:
    return std::make_pair(tt::target::DataType::UInt16, 2);
  case PJRT_Buffer_Type_U32:
    return std::make_pair(tt::target::DataType::UInt32, 4);
  case PJRT_Buffer_Type_F16:
    return std::make_pair(tt::target::DataType::Float16, 2);
  case PJRT_Buffer_Type_F32:
    return std::make_pair(tt::target::DataType::Float32, 4);
  case PJRT_Buffer_Type_BF16:
    return std::make_pair(tt::target::DataType::BFloat16, 2);
  case PJRT_Buffer_Type_INVALID:
  case PJRT_Buffer_Type_S4:
  case PJRT_Buffer_Type_S8:
  case PJRT_Buffer_Type_S16:
  case PJRT_Buffer_Type_S32:
  case PJRT_Buffer_Type_S64:
  case PJRT_Buffer_Type_U4:
  case PJRT_Buffer_Type_PRED:
  case PJRT_Buffer_Type_U64:
  case PJRT_Buffer_Type_F64:
  case PJRT_Buffer_Type_C64:
  case PJRT_Buffer_Type_C128:
  default:
    assert(false && "Unsupported buffer type");
    return std::make_pair(tt::target::DataType::BFloat16, 2);
  }
}

} // namespace tt::pjrt
