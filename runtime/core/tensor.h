#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace us4 {

enum class DType {
  kFloat32,
  kFloat16,
  kBFloat16,
  kInt8,
  kInt4,
};

enum class DeviceType {
  kCpu,
  kMlx,
  kMetal,
  kNeon,
  kAne,
  kUnknown,
};

std::string_view ToString(DType dtype);
std::string_view ToString(DeviceType device);
std::size_t DTypeBitWidth(DType dtype);
std::uint16_t EncodeFloat16(float value);
float DecodeFloat16(std::uint16_t value);
std::uint16_t EncodeBFloat16(float value);
float DecodeBFloat16(std::uint16_t value);

class Tensor {
public:
  Tensor() = default;
  Tensor(std::vector<std::size_t> shape, DType dtype,
         DeviceType device = DeviceType::kCpu);

  const std::vector<std::size_t> &Shape() const;
  const std::vector<std::size_t> &Strides() const;
  DType dtype() const;
  DeviceType device() const;

  std::size_t Rank() const;
  std::size_t ElementCount() const;
  std::size_t ByteSize() const;
  bool Empty() const;
  bool IsContiguous() const;

  const std::byte *Data() const;
  std::byte *MutableData();

  const float *DataAsFloat32() const;
  float *MutableDataAsFloat32();
  const std::uint16_t *DataAsUInt16() const;
  std::uint16_t *MutableDataAsUInt16();

  bool Reshape(std::vector<std::size_t> shape);
  void FillZero();

private:
  static std::vector<std::size_t>
  ComputeContiguousStrides(const std::vector<std::size_t> &shape);
  static std::size_t ComputeElementCount(const std::vector<std::size_t> &shape);

  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
  DType dtype_ = DType::kFloat32;
  DeviceType device_ = DeviceType::kCpu;
  std::vector<std::byte> storage_;
};

} // namespace us4
