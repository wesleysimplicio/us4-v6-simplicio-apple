#include "core/tensor.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace us4 {

namespace {

std::uint32_t FloatToBits(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsToFloat(const std::uint32_t bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

} // namespace

std::string_view ToString(const DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return "fp32";
  case DType::kFloat16:
    return "fp16";
  case DType::kBFloat16:
    return "bf16";
  case DType::kInt8:
    return "int8";
  case DType::kInt4:
    return "int4";
  }
  return "unknown";
}

std::string_view ToString(const DeviceType device) {
  switch (device) {
  case DeviceType::kCpu:
    return "cpu";
  case DeviceType::kMlx:
    return "mlx";
  case DeviceType::kMetal:
    return "metal";
  case DeviceType::kNeon:
    return "neon";
  case DeviceType::kAne:
    return "ane";
  case DeviceType::kUnknown:
    return "unknown";
  }
  return "unknown";
}

std::size_t DTypeBitWidth(const DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return 32;
  case DType::kFloat16:
  case DType::kBFloat16:
    return 16;
  case DType::kInt8:
    return 8;
  case DType::kInt4:
    return 4;
  }
  return 0;
}

std::uint16_t EncodeFloat16(const float value) {
  const std::uint32_t bits = FloatToBits(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::int32_t exponent =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    std::uint32_t subnormal = mantissa | 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t rounded = subnormal >> shift;
    if (((subnormal >> (shift - 1U)) & 0x1U) != 0U) {
      rounded += 1U;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  if (exponent >= 31) {
    const std::uint16_t infNan = static_cast<std::uint16_t>(sign | 0x7C00U);
    if (mantissa == 0U) {
      return infNan;
    }
    return static_cast<std::uint16_t>(infNan | ((mantissa >> 13U) | 0x1U));
  }

  const std::uint32_t roundedMantissa = mantissa + 0x1000U;
  if ((roundedMantissa & 0x800000U) != 0U) {
    return static_cast<std::uint16_t>(
        sign | ((static_cast<std::uint32_t>(exponent + 1) << 10U)));
  }

  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10U) |
      ((roundedMantissa >> 13U) & 0x3FFU));
}

float DecodeFloat16(const std::uint16_t value) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U))
                             << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1FU;
  const std::uint32_t mantissa = value & 0x03FFU;

  if (exponent == 0U) {
    if (mantissa == 0U) {
      return BitsToFloat(sign);
    }

    std::uint32_t normalizedMantissa = mantissa;
    std::int32_t adjustedExponent = -14;
    while ((normalizedMantissa & 0x0400U) == 0U) {
      normalizedMantissa <<= 1U;
      --adjustedExponent;
    }
    normalizedMantissa &= 0x03FFU;
    const std::uint32_t bits =
        sign | (static_cast<std::uint32_t>(adjustedExponent + 127) << 23U) |
        (normalizedMantissa << 13U);
    return BitsToFloat(bits);
  }

  if (exponent == 0x1FU) {
    const std::uint32_t bits = sign | 0x7F800000U | (mantissa << 13U);
    return BitsToFloat(bits);
  }

  const std::uint32_t bits =
      sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  return BitsToFloat(bits);
}

std::uint16_t EncodeBFloat16(const float value) {
  const std::uint32_t bits = FloatToBits(value);
  const std::uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 0x1U);
  return static_cast<std::uint16_t>((bits + roundingBias) >> 16U);
}

float DecodeBFloat16(const std::uint16_t value) {
  return BitsToFloat(static_cast<std::uint32_t>(value) << 16U);
}

Tensor::Tensor(std::vector<std::size_t> shape, const DType dtype,
               const DeviceType device)
    : shape_(std::move(shape)), strides_(ComputeContiguousStrides(shape_)),
      dtype_(dtype), device_(device), storage_(ByteSize()) {}

const std::vector<std::size_t> &Tensor::Shape() const { return shape_; }

const std::vector<std::size_t> &Tensor::Strides() const { return strides_; }

DType Tensor::dtype() const { return dtype_; }

DeviceType Tensor::device() const { return device_; }

std::size_t Tensor::Rank() const { return shape_.size(); }

std::size_t Tensor::ElementCount() const { return ComputeElementCount(shape_); }

std::size_t Tensor::ByteSize() const {
  const std::size_t bits = ElementCount() * DTypeBitWidth(dtype_);
  return bits == 0 ? 0 : (bits + 7U) / 8U;
}

bool Tensor::Empty() const { return ElementCount() == 0; }

bool Tensor::IsContiguous() const {
  return strides_ == ComputeContiguousStrides(shape_);
}

const std::byte *Tensor::Data() const { return storage_.data(); }

std::byte *Tensor::MutableData() { return storage_.data(); }

const float *Tensor::DataAsFloat32() const {
  return dtype_ == DType::kFloat32
             ? reinterpret_cast<const float *>(storage_.data())
             : nullptr;
}

float *Tensor::MutableDataAsFloat32() {
  return dtype_ == DType::kFloat32 ? reinterpret_cast<float *>(storage_.data())
                                   : nullptr;
}

const std::uint16_t *Tensor::DataAsUInt16() const {
  return (dtype_ == DType::kFloat16 || dtype_ == DType::kBFloat16)
             ? reinterpret_cast<const std::uint16_t *>(storage_.data())
             : nullptr;
}

std::uint16_t *Tensor::MutableDataAsUInt16() {
  return (dtype_ == DType::kFloat16 || dtype_ == DType::kBFloat16)
             ? reinterpret_cast<std::uint16_t *>(storage_.data())
             : nullptr;
}

bool Tensor::Reshape(std::vector<std::size_t> shape) {
  if (ComputeElementCount(shape) != ElementCount()) {
    return false;
  }

  shape_ = std::move(shape);
  strides_ = ComputeContiguousStrides(shape_);
  return true;
}

void Tensor::FillZero() {
  std::fill(storage_.begin(), storage_.end(), std::byte{0});
}

std::vector<std::size_t>
Tensor::ComputeContiguousStrides(const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> strides(shape.size(), 1);
  for (std::size_t index = shape.size(); index > 0; --index) {
    if (index < shape.size()) {
      strides[index - 1] = strides[index] * shape[index];
    }
  }
  return strides;
}

std::size_t Tensor::ComputeElementCount(const std::vector<std::size_t> &shape) {
  if (shape.empty()) {
    return 0;
  }

  std::size_t count = 1;
  for (const std::size_t dim : shape) {
    count *= dim;
  }
  return count;
}

} // namespace us4
