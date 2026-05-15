#include "neon/kernel_profile.h"

namespace us4 {

std::string_view ToString(const NeonKernelFlavor flavor) {
  switch (flavor) {
  case NeonKernelFlavor::kScalarBridge:
    return "scalar-bridge";
  case NeonKernelFlavor::kFp32Lane4:
    return "fp32-lane4";
  case NeonKernelFlavor::kFp16Lane8:
    return "fp16-lane8";
  case NeonKernelFlavor::kBf16Lane8:
    return "bf16-lane8";
  case NeonKernelFlavor::kInt8Dot:
    return "int8-dot";
  }
  return "scalar-bridge";
}

NeonMatmulProfile PlanNeonMatmul(const HardwareProbeResult &hardware,
                                 const Tensor &lhs, const Tensor &rhs) {
  const bool neonReady = hardware.hasNeon && hardware.architecture == "arm64";
  if (!neonReady || lhs.Empty() || rhs.Empty()) {
    return {
        .flavor = NeonKernelFlavor::kScalarBridge,
        .usesAccelerateFallback = hardware.hasNeon,
    };
  }

  switch (lhs.dtype()) {
  case DType::kFloat32:
    return {
        .flavor = NeonKernelFlavor::kFp32Lane4,
        .tileRows = 4,
        .tileCols = 16,
        .vectorLanes = 4,
    };
  case DType::kFloat16:
    return {
        .flavor = NeonKernelFlavor::kFp16Lane8,
        .tileRows = 8,
        .tileCols = 8,
        .vectorLanes = 8,
    };
  case DType::kBFloat16:
    return {
        .flavor = NeonKernelFlavor::kBf16Lane8,
        .tileRows = 8,
        .tileCols = 8,
        .vectorLanes = 8,
    };
  case DType::kInt8:
    return {
        .flavor = NeonKernelFlavor::kInt8Dot,
        .tileRows = 8,
        .tileCols = 8,
        .vectorLanes = 16,
        .usesDotProduct = true,
    };
  case DType::kInt4:
    break;
  }

  return {
      .flavor = NeonKernelFlavor::kScalarBridge,
      .usesAccelerateFallback = true,
  };
}

NeonAttentionProfile PlanNeonAttention(const HardwareProbeResult &hardware,
                                       const Tensor &query, const Tensor &key,
                                       const Tensor &value,
                                       const bool causalMask) {
  const bool neonReady = hardware.hasNeon && hardware.architecture == "arm64";
  if (!neonReady || query.Empty() || key.Empty() || value.Empty()) {
    return {.flavor = NeonKernelFlavor::kScalarBridge};
  }

  const std::size_t headDim = query.Shape().empty() ? 0 : query.Shape().back();
  const bool usesWideLanes =
      query.dtype() == DType::kFloat16 || query.dtype() == DType::kBFloat16;

  return {
      .flavor = usesWideLanes ? NeonKernelFlavor::kFp16Lane8
                              : NeonKernelFlavor::kFp32Lane4,
      .vectorLanes = usesWideLanes ? 8U : 4U,
      .headDimBlock = headDim >= 32U ? 32U : 16U,
      .fusesSoftmaxRescale = true,
      .supportsCausalMask = causalMask,
  };
}

} // namespace us4
