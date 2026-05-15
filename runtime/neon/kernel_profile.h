#pragma once

#include <cstddef>
#include <string_view>

#include "core/hardware_probe.h"
#include "core/tensor.h"

namespace us4 {

enum class NeonKernelFlavor {
  kScalarBridge,
  kFp32Lane4,
  kFp16Lane8,
  kBf16Lane8,
  kInt8Dot,
};

struct NeonMatmulProfile {
  NeonKernelFlavor flavor = NeonKernelFlavor::kScalarBridge;
  std::size_t tileRows = 0;
  std::size_t tileCols = 0;
  std::size_t vectorLanes = 0;
  bool usesDotProduct = false;
  bool usesAccelerateFallback = false;
};

struct NeonAttentionProfile {
  NeonKernelFlavor flavor = NeonKernelFlavor::kScalarBridge;
  std::size_t vectorLanes = 0;
  std::size_t headDimBlock = 0;
  bool fusesSoftmaxRescale = false;
  bool supportsCausalMask = false;
};

std::string_view ToString(NeonKernelFlavor flavor);
NeonMatmulProfile PlanNeonMatmul(const HardwareProbeResult &hardware,
                                 const Tensor &lhs, const Tensor &rhs);
NeonAttentionProfile PlanNeonAttention(const HardwareProbeResult &hardware,
                                       const Tensor &query, const Tensor &key,
                                       const Tensor &value, bool causalMask);

} // namespace us4
