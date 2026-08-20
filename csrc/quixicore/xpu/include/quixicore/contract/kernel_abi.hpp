#pragma once

#include <cstddef>
#include <cstdint>

namespace quixicore::contract {

// Stable adapter ABI shared by every backend. Native runtime handles and
// optimized typed entry points remain backend-owned behind this boundary.
enum class StatusCode : std::uint8_t {
  ok,
  invalid_argument,
  invalid_shape,
  unsupported_dtype,
  unsupported_format,
  unsupported_layout,
  insufficient_workspace,
  adapter_not_wired,
  not_implemented,
  runtime_error,
};

struct Status {
  StatusCode code;
  const char* operation;
  const char* detail;

  [[nodiscard]] constexpr bool is_ok() const noexcept {
    return code == StatusCode::ok;
  }
};

enum class DType : std::uint8_t {
  unknown,
  boolean,
  i8,
  u8,
  i16,
  u16,
  i32,
  u32,
  i64,
  u64,
  f16,
  bf16,
  f32,
  f64,
  e4m3,
  e5m2,
  packed,
};

enum class MemorySpace : std::uint8_t {
  unknown,
  host,
  device,
  shared,
};

enum TensorFlags : std::uint32_t {
  tensor_none = 0,
  tensor_read_only = 1U << 0,
  tensor_contiguous = 1U << 1,
  tensor_optional = 1U << 2,
};

struct TensorView {
  void* data;
  const std::int64_t* shape;
  const std::int64_t* strides;
  std::size_t rank;
  DType dtype;
  MemorySpace memory_space;
  std::uint32_t flags;
};

enum class AttributeKind : std::uint8_t {
  boolean,
  i64,
  f64,
  string,
  bytes,
};

struct Attribute {
  const char* name;
  AttributeKind kind;
  const void* data;
  std::size_t size;
  std::int64_t i64;
  double f64;
};

struct KernelCall {
  const TensorView* inputs;
  std::size_t input_count;
  TensorView* outputs;
  std::size_t output_count;
  const Attribute* attributes;
  std::size_t attribute_count;
  void* backend_context;
  void* stream;
  void* workspace;
  std::size_t workspace_size;
};

using KernelFunction = Status (*)(const KernelCall&) noexcept;

struct StubDescriptor {
  const char* operation;
  const char* family;
  const char* reason;
  KernelFunction function;
};

[[nodiscard]] constexpr Status not_implemented(const char* operation,
                                               const char* detail) noexcept {
  return {StatusCode::not_implemented, operation, detail};
}

[[nodiscard]] constexpr Status adapter_not_wired(const char* operation) noexcept {
  return {StatusCode::adapter_not_wired, operation,
          "native implementation evidence exists; canonical adapter not wired"};
}

}  // namespace quixicore::contract
