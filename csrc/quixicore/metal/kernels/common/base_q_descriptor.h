// Clean-room host descriptor for the public BaseQN tensor contract.
#pragma once

#include <stdexcept>
#include <string>

namespace tk {

enum class BaseQScaleType : int {
  BF16 = 0,
  F16 = 1,
  E8M0 = 2,
  E4M3 = 3,
};

enum class BaseQLayout : int {
  MetalLaneStrided = 0,
};

struct BaseQDescriptor {
  int bits;
  int group_size;
  BaseQScaleType scale_type;
  BaseQLayout layout;
  bool symmetric;

  bool operator==(const BaseQDescriptor& other) const {
    return bits == other.bits && group_size == other.group_size &&
        scale_type == other.scale_type && layout == other.layout &&
        symmetric == other.symmetric;
  }
};

inline BaseQScaleType parse_base_q_scale_type(const std::string& value) {
  if (value == "bf16" || value == "bfloat16") return BaseQScaleType::BF16;
  if (value == "f16" || value == "float16") return BaseQScaleType::F16;
  if (value == "e8m0") return BaseQScaleType::E8M0;
  if (value == "e4m3") return BaseQScaleType::E4M3;
  throw std::invalid_argument(
      "BaseQN: scale_dtype must be bf16, f16, e8m0, or e4m3");
}

inline BaseQLayout parse_base_q_layout(const std::string& value, int bits) {
  if (value == "metal" || value == "metal_lane_strided" ||
      value == "metal_lane_strided_q" + std::to_string(bits)) {
    return BaseQLayout::MetalLaneStrided;
  }
  throw std::invalid_argument(
      "BaseQN: layout must be metal or metal_lane_strided_q" +
      std::to_string(bits));
}

inline BaseQDescriptor make_base_q_descriptor(
    int bits, int group_size, const std::string& scale_dtype,
    bool symmetric, const std::string& layout) {
  if (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 &&
      bits != 8) {
    throw std::invalid_argument("BaseQN: bits must be one of 2, 3, 4, 5, 6, or 8");
  }
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    throw std::invalid_argument("BaseQN: group_size must be 32, 64, or 128");
  }
  const auto scale_type = parse_base_q_scale_type(scale_dtype);
  if (scale_type == BaseQScaleType::E4M3 && bits != 8) {
    throw std::invalid_argument("BaseQN: e4m3 scale storage is valid only for q8");
  }
  return {bits, group_size, scale_type, parse_base_q_layout(layout, bits), symmetric};
}

inline int base_q_scale_type_code(const BaseQDescriptor& descriptor) {
  return static_cast<int>(descriptor.scale_type);
}

} // namespace tk
